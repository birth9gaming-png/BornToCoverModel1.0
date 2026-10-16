"""Tests for injury reports, key players, QB adjustment and the Pick Score (offline)."""
import shutil, sys, tempfile, unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))

import injuries
from board import build_board, shortlist, pick_score, grade_from_score, passes_rules, DEFAULT_RULES
from model import load_model
from storage import Store
from test_backend import simulated, fake_events

ESPN = {'injuries': [{'displayName': 'Philadelphia Eagles', 'injuries': [
    {'status': 'Out', 'date': '2026-10-15T20:00Z', 'shortComment': 'Hurts (calf) will not play Sunday.',
     'athlete': {'displayName': 'Jalen Hurts', 'position': {'abbreviation': 'QB'}, 'team': {'abbreviation': 'PHI'}}},
    {'status': 'Questionable', 'date': '2026-10-15T20:00Z',
     'athlete': {'displayName': 'Saquon Barkley', 'position': {'abbreviation': 'RB'}, 'team': {'abbreviation': 'PHI'}}},
    {'status': 'Out', 'date': '2026-10-15T20:00Z',
     'athlete': {'displayName': 'DeVonta Smith', 'position': {'abbreviation': 'WR'}, 'team': {'abbreviation': 'PHI'}}}]},
    {'displayName': 'Washington Commanders', 'injuries': [
    {'status': 'Injured Reserve', 'date': '2026-10-01T20:00Z',
     'athlete': {'displayName': 'Terry McLaurin', 'position': {'abbreviation': 'WR'}, 'team': {'abbreviation': 'WSH'}}}]}]}

COVERS = '''<html><body><h2>College Football Injuries</h2>
<div><h3><img/>Air Force Falcons (2)</h3><table><tr><th>Player</th><th>POS</th><th>Status</th></tr>
<tr><td><a>G. Stalfort</a></td><td>TE</td><td><b>Questionable</b> - Undisclosed (Sun, Oct 11)</td></tr>
<tr><td><a>L. Szarka</a></td><td>QB</td><td><b>Out</b> - Leg (Mon, Oct 5)</td></tr></table></div>
<div><h3>Sample State Owls (1)</h3><table>
<tr><td><a>Z. Sample</a></td><td>QB</td><td>Out - Shoulder (Tue, Oct 13)</td></tr></table></div></body></html>'''


class TestInjuries(unittest.TestCase):
    def test_name_keys(self):
        self.assertEqual(injuries.name_key('Jalen Hurts'), injuries.name_key('J.Hurts'))
        self.assertEqual(injuries.name_key('Amon-Ra St. Brown'), injuries.name_key('A.St. Brown'))
        self.assertEqual(injuries.name_key('Michael Penix Jr.'), injuries.name_key('M.Penix'))
        self.assertTrue(injuries.is_out('Out')); self.assertTrue(injuries.is_out('Injured Reserve'))
        self.assertTrue(injuries.is_out('Doubtful')); self.assertFalse(injuries.is_out('Questionable'))
        self.assertFalse(injuries.is_out('Active'))

    def test_espn_and_covers(self):
        e = injuries.parse_espn(ESPN)
        self.assertEqual(set(e.team), {'PHI', 'WAS'})
        c = injuries.parse_covers(COVERS)
        self.assertEqual(len(c), 3)
        row = c[c.player == 'L. Szarka'].iloc[0]
        self.assertEqual((row.team, row.pos, row.status), ('Air Force Falcons', 'QB', 'Out'))
        jmu = injuries.match_team_rows(c, "Sample State", "CFB")
        self.assertEqual(list(jmu.player), ["Z. Sample"])

    def test_key_players_and_report(self):
        pbp = pd.DataFrame({
            'season_type': 'REG', 'week': [5] * 7, 'posteam': ['PHI'] * 7,
            'play_type': ['pass', 'pass', 'pass', 'run', 'run', 'pass', 'pass'],
            'qb_dropback': [1, 1, 1, 0, 0, 1, 1],
            'passer_player_name': ['J.Hurts'] * 3 + [None, None] + ['J.Hurts', 'T.McKee'],
            'rusher_player_name': [None, None, None, 'S.Barkley', 'S.Barkley', None, None],
            'receiver_player_name': ['D.Smith', 'A.Brown', 'D.Smith', None, None, 'A.Brown', 'D.Smith']})
        keys = injuries.key_players_nfl(pbp)['PHI']
        self.assertEqual({k['role'] for k in keys}, {'QB', 'RB1', 'Target1', 'Target2'})
        rep = injuries.team_report('PHI', injuries.parse_espn(ESPN).query("team == 'PHI'"), keys)
        self.assertTrue(rep['qb_out'])
        self.assertEqual({m['role'] for m in rep['missing']}, {'QB', 'Target1'})
        cfb = injuries.key_players_cfb([
            dict(team='SAMP', player='Zed Sample', category='passing', statType='ATT', stat='250'),
            dict(team='SAMP', player='Zed Sample', category='rushing', statType='CAR', stat='90'),
            dict(team='SAMP', player='Rob Runner', category='rushing', statType='CAR', stat='80'),
            dict(team='SAMP', player='Wes Wideout', category='receiving', statType='REC', stat='30'),
            dict(team='SAMP', player='Tim Tightend', category='receiving', statType='REC', stat='25')])
        roles = {k['role']: k['name'] for k in cfb['SAMP']}
        self.assertEqual(roles['QB'], 'Zed Sample')
        self.assertNotIn('RB1', [r for r, n in roles.items() if n == 'Zed Sample'])


class TestScoreAndBoard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(); self.store = Store({}, local_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_pick_score(self):
        row = dict(edge=-6.0, move_home=-1.0, split_flags=dict(sharp_side='home'), spots={'away': {'letdown': True}, 'home': {}},
                   injuries={'home': {}, 'away': {}})
        sc, parts = pick_score('home', row, 'NFL')
        self.assertEqual(sc, 50 + 12 * 3)                     # full edge + line move + money + opponent spot
        self.assertEqual(grade_from_score(sc), 'A')
        row['injuries']['home'] = dict(missing=[dict(role='QB', player='X', status='Out')])
        self.assertEqual(pick_score('home', row, 'NFL')[0], 50 + 36 - 25)
        self.assertEqual(pick_score('away', row, 'NFL')[0], 0)   # model against + 3 signals against

    def test_board_qb_out_and_shortlist_by_score(self):
        data, t = simulated('NFL', '2025-11-01T12:00:00Z')
        g = data['games']
        slate = g[(~g.completed) & (g.kickoff < t + pd.Timedelta(days=8))]
        self.store.put_json(f"odds/NFL/{(t - pd.Timedelta(hours=2)).strftime('%Y-%m-%dT%H-%M-%S')}.json.gz",
                            dict(fetched_at=(t - pd.Timedelta(hours=2)).isoformat(), league='NFL', events=fake_events(slate, 'NFL')))
        m = load_model('NFL')
        base = build_board('NFL', data, m, self.store, now=t, with_weather=False)
        top = base.sort_values('score', ascending=False).iloc[0]
        liked = top.away if top.model_side == 'away' else top.home
        # the liked team's QB is ruled out
        self.store.put_json('cache/nfl/key_players.json', {liked: [dict(name='Q.Back', role='QB')]})
        self.store.put_json('injuries/NFL.json', dict(at=str(t), source='test', rows=[
            dict(team=liked, player='Quinn Back', pos='QB', status='Out')]))
        b = build_board('NFL', data, m, self.store, now=t, with_weather=False)
        r = b[b.id == top.id].iloc[0]
        self.assertAlmostEqual(abs(r.model_home - r.model_home_raw), 2.0)
        self.assertLess(r.score, top.score)
        ok, why = passes_rules(r.to_dict(), 'NFL', DEFAULT_RULES)
        self.assertFalse(ok); self.assertIn('key player', why)
        sl = shortlist(b, 'NFL', rules=DEFAULT_RULES)
        self.assertNotIn(top.id, list(sl.id))
        self.assertTrue((sl.score.diff().dropna() <= 0).all())          # shortlist ordered by score
        ok_scores = [x.score for x in b.itertuples() if passes_rules(x._asdict(), 'NFL', DEFAULT_RULES)[0]]
        self.assertEqual(sorted(ok_scores, reverse=True)[:len(sl)], list(sl.score))


if __name__ == '__main__':
    unittest.main()
