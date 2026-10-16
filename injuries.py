"""Injury reports + each team's key players, so the model knows who is missing.

Sources (both allow automated reads in robots.txt):
  NFL - ESPN's public injuries feed (JSON)
  CFB - Covers.com college football injury report (one page, updated daily)
Key players come from this season's stats:
  NFL - play-by-play, last 4 games: main QB (dropbacks), top RB (carries), top 2 targets
  CFB - CFBD season player stats: main QB (attempts), top RB (carries), top 2 receivers (catches)
A key player listed Out / Doubtful / IR / Suspended counts as missing.  A QB
listed Questionable is flagged too, because the market moves hard on QBs.
"""
from __future__ import annotations
import re
from datetime import datetime, timezone
import pandas as pd
import requests

ESPN_NFL = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries'
COVERS = {'CFB': 'https://www.covers.com/sport/football/ncaaf/injuries',
          'NFL': 'https://www.covers.com/sport/football/nfl/injuries'}
HEADERS = {'User-Agent': 'Mozilla/5.0 (BornToCover personal injury reader; one request per refresh)'}

SUFFIX = {'jr', 'sr', 'ii', 'iii', 'iv', 'v'}
STATUS_RE = re.compile(r'^(Out For Season|Out|Questionable|Doubtful|Probable|Day-To-Day|Injured Reserve|IR|Suspended|'
                       r'Suspension|Game Time Decision|Inactive)\b\s*-?\s*(.*?)\s*(?:\(([^)]*)\))?\s*$', re.I)
POS_RE = re.compile(r'^(QB|RB|FB|WR|TE|OL|OT|OG|G|T|C|DL|DE|DT|NT|LB|ILB|OLB|EDGE|CB|S|FS|SS|DB|K|P|LS|ATH|KR|PR)$', re.I)


def name_key(name: str) -> str:
    """'Jalen Hurts' / 'J.Hurts' / 'J. Hurts' -> 'j hurts'.  'Amon-Ra St. Brown' -> 'a brown'."""
    s = re.sub(r"[^a-z.\s'-]", '', str(name).lower()).replace('.', ' ').replace('-', ' ')
    parts = [p for p in s.split() if p and p not in SUFFIX]
    if len(parts) < 2:
        return parts[0] if parts else ''
    return f'{parts[0][0]} {parts[-1]}'


def is_out(status: str) -> bool:
    s = str(status).lower().strip()
    return bool(re.match(r'^(out|doubtful|injured reserve|ir\b|suspen|reserve|pup|inactive)', s))


# ---------------------------------------------------------------- fetch
def _get(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r


def parse_espn(payload: dict) -> pd.DataFrame:
    from history import nfl_code
    rows = []
    for team in payload.get('injuries', []):
        for e in team.get('injuries', []):
            a = e.get('athlete') or {}
            abbr = ((a.get('team') or {}).get('abbreviation')) or ''
            pos = ((a.get('position') or {}).get('abbreviation')) or ''
            rows.append(dict(team=nfl_code(abbr) if abbr else team.get('displayName'), player=a.get('displayName'),
                             pos=pos, status=e.get('status', ''), date=e.get('date'),
                             note=e.get('shortComment') or ''))
    return pd.DataFrame(rows)


def parse_covers(page: str) -> pd.DataFrame:
    """Covers injury page -> rows (team heading 'Air Force Falcons (2)', then Player / POS / Status)."""
    from splits import html_to_text
    lines = html_to_text(page).splitlines()
    rows, team = [], None
    for i, ln in enumerate(lines):
        h = re.match(r'^(.+?)\s*\((\d+)\)$', ln)
        if h and not STATUS_RE.match(ln) and re.search(r'[A-Za-z]{3,}', h.group(1)):
            team = h.group(1).strip()
            continue
        m = STATUS_RE.match(ln) if len(ln) <= 60 else None      # long lines are news blurbs, not status cells
        if m and team and i >= 2:
            pos, player = lines[i - 1].strip(), lines[i - 2].strip()
            if not POS_RE.match(pos):                      # layout without a POS cell
                pos, player = '', lines[i - 1].strip()
            rows.append(dict(team=team, player=player, pos=pos.upper(), status=m.group(1).title(),
                             injury=m.group(2), date=m.group(3), note=''))
    return pd.DataFrame(rows)


def fetch(league: str) -> tuple[pd.DataFrame, str]:
    """Injury rows for a league and the source used."""
    errors = []
    if league == 'NFL':
        try:
            df = parse_espn(_get(ESPN_NFL).json())
            if len(df):
                return df, 'ESPN'
        except Exception as exc:
            errors.append(f'ESPN: {exc}')
    try:
        df = parse_covers(_get(COVERS[league]).text)
        if len(df):
            return df, 'Covers'
    except Exception as exc:
        errors.append(f'Covers: {exc}')
    raise RuntimeError('; '.join(errors) or 'no injury rows found')


def refresh(store, league: str, log=print):
    df, src = fetch(league)
    store.put_json(f'injuries/{league}.json', dict(at=datetime.now(timezone.utc).isoformat(), source=src,
                                                   rows=df.to_dict('records')))
    log(f'Injuries: {len(df)} report lines from {src}')


def load(store, league: str) -> tuple[pd.DataFrame, str | None, str | None]:
    v = store.get_json(f'injuries/{league}.json')
    if not v:
        return pd.DataFrame(columns=['team', 'player', 'pos', 'status']), None, None
    return pd.DataFrame(v['rows']), v.get('at'), v.get('source')


# ---------------------------------------------------------------- key players
def key_players_nfl(pbp: pd.DataFrame, last_n_weeks: int = 4) -> dict:
    """team -> list of {name, role} from recent play-by-play."""
    from history import nfl_code
    df = pbp[pbp['season_type'] == 'REG'].copy()
    if df.empty:
        return {}
    df['week'] = pd.to_numeric(df['week'], errors='coerce')
    recent = df[df['week'] > df['week'].max() - last_n_weeks]
    recent = recent[recent['posteam'].notna()]
    recent['posteam'] = recent['posteam'].map(nfl_code)
    out = {}
    for team, g in recent.groupby('posteam'):
        roles = []
        qb = g[pd.to_numeric(g.get('qb_dropback'), errors='coerce') == 1]['passer_player_name'].dropna().value_counts()
        if len(qb):
            roles.append(dict(name=qb.index[0], role='QB'))
        rb = g[g['play_type'] == 'run']['rusher_player_name'].dropna()
        rb = rb[~rb.isin([r['name'] for r in roles])].value_counts()
        if len(rb):
            roles.append(dict(name=rb.index[0], role='RB1'))
        tg = g['receiver_player_name'].dropna().value_counts()
        for k, n in enumerate(tg.index[:2]):
            roles.append(dict(name=n, role=f'Target{k + 1}'))
        out[team] = roles
    return out


def key_players_cfb(rows: list[dict]) -> dict:
    """CFBD /stats/player/season rows -> team -> key players."""
    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    df['stat'] = pd.to_numeric(df['stat'], errors='coerce')
    cat = df['category'].str.lower(); typ = df['statType'].str.upper()
    out = {}

    def top(mask, n, role):
        s = df[mask].sort_values('stat', ascending=False)
        for team, g in s.groupby('team'):
            for k, r in enumerate(g.head(n).itertuples()):
                out.setdefault(team, []).append(dict(name=r.player, role=role if n == 1 else f'{role}{k + 1}'))
    top((cat == 'passing') & (typ == 'ATT'), 1, 'QB')
    top((cat == 'rushing') & (typ == 'CAR'), 1, 'RB1')
    top((cat == 'receiving') & (typ == 'REC'), 2, 'Target')
    # a QB who also leads in carries should not double as RB1
    for team, roles in out.items():
        qb = {name_key(r['name']) for r in roles if r['role'] == 'QB'}
        out[team] = [r for r in roles if not (r['role'] == 'RB1' and name_key(r['name']) in qb)]
    return out


# ---------------------------------------------------------------- per game
def team_report(team: str, rows: pd.DataFrame, keys: list[dict]) -> dict:
    """Missing key players + other notable outs for one team (rows already filtered to the team)."""
    keymap = {name_key(k['name']): k['role'] for k in keys or []}
    missing, questionable, others = [], [], []
    for r in rows.itertuples():
        k = name_key(r.player)
        role = keymap.get(k)
        label = f"{r.player} ({r.pos or '?'}) {r.status}"
        if role and is_out(r.status):
            missing.append(dict(player=r.player, role=role, status=r.status))
        elif role and role == 'QB' and 'question' in str(r.status).lower():
            questionable.append(dict(player=r.player, role=role, status=r.status))
        elif is_out(r.status) and str(r.pos).upper() in ('QB', 'RB', 'WR', 'TE'):
            others.append(label)
    return dict(missing=missing, qb_out=any(m['role'] == 'QB' for m in missing),
                qb_questionable=bool(questionable), questionable=questionable, others=others[:6],
                known=bool(keys))


def match_team_rows(df: pd.DataFrame, team: str, league: str) -> pd.DataFrame:
    if df.empty:
        return df
    if league == 'NFL':
        return df[df['team'] == team]
    from odds import match_team
    names = df['team'].dropna().unique()
    hits = [n for n in names if match_team(n, [team]) == team]
    return df[df['team'].isin(hits)]


def describe(rep: dict, team: str) -> list[str]:
    out = []
    for m in rep.get('missing', []):
        out.append(f"{team}: {m['role']} {m['player']} {m['status'].upper()}")
    for q in rep.get('questionable', []):
        out.append(f"{team}: QB {q['player']} questionable")
    return out
