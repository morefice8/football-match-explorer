import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.defensive_metrics import (  # noqa: E402
    calculate_ppda_data,
    calculate_ppda_profile,
    extract_ppda_key_events,
)


class PPDAMetricTests(unittest.TestCase):
    def setUp(self):
        rows = [
            # Team A pressing actions.
            dict(team_name='A', type_name='Tackle', outcome='Successful', x=40, y=30, periodId=1, timeMin=10, timeSec=0, playerName='A1', **{'Mapped Jersey Number': 4}),
            dict(team_name='A', type_name='Foul', outcome='Unsuccessful', x=50, y=50, periodId=1, timeMin=30, timeSec=0, playerName='A2', **{'Mapped Jersey Number': 8}),
            dict(team_name='A', type_name='Interception', outcome='Successful', x=70, y=60, periodId=2, timeMin=70, timeSec=0, playerName='A1', **{'Mapped Jersey Number': 4}),
            # Team B passes: only x < 60 belongs to the PPDA numerator.
            dict(team_name='B', type_name='Pass', outcome='Successful', x=10, y=20, periodId=1, timeMin=5, timeSec=0, playerName='B1'),
            dict(team_name='B', type_name='Pass', outcome='Successful', x=59, y=30, periodId=1, timeMin=20, timeSec=0, playerName='B2'),
            dict(team_name='B', type_name='Pass', outcome='Successful', x=60, y=40, periodId=1, timeMin=25, timeSec=0, playerName='B2'),
            dict(team_name='B', type_name='Pass', outcome='Successful', x=90, y=50, periodId=1, timeMin=35, timeSec=0, playerName='B3'),
            dict(team_name='B', type_name='Pass', outcome='Successful', x=20, y=60, periodId=2, timeMin=55, timeSec=0, playerName='B1'),
            dict(team_name='B', type_name='Pass', outcome='Unsuccessful', x=30, y=70, periodId=2, timeMin=80, timeSec=0, playerName='B2'),
        ]
        self.df = pd.DataFrame(rows)

    def test_ppda_uses_opponent_first_sixty_percent(self):
        ppda, defensive_actions, opponent_passes, player_stats = calculate_ppda_data(self.df, 'A', 'B')

        self.assertAlmostEqual(ppda, 4 / 3)
        self.assertEqual(len(defensive_actions), 3)
        self.assertEqual(len(opponent_passes), 4)
        self.assertTrue((opponent_passes['x'] < 60).all())
        self.assertEqual(int(player_stats['Actions'].sum()), 3)

    def test_profile_splits_halves_without_crossing_periods(self):
        profile = calculate_ppda_profile(self.df, 'A', 'B')

        self.assertAlmostEqual(profile['first_half']['ppda'], 1.0)
        self.assertAlmostEqual(profile['second_half']['ppda'], 2.0)
        self.assertEqual(profile['first_half']['opponent_passes'], 2)
        self.assertEqual(profile['second_half']['opponent_passes'], 2)
        self.assertEqual(set(profile['timeline']['period']), {'1H', '2H'})
        self.assertEqual(len(profile['timeline']), 6)
        self.assertFalse(profile['timeline']['pressure_rate'].isna().any())
        self.assertTrue((profile['timeline']['pressure_rate'] >= 0).all())
        self.assertEqual(int(profile['timeline']['opponent_passes'].sum()), 4)
        self.assertEqual(int(profile['timeline']['defensive_actions'].sum()), 3)

    def test_key_events_include_goals_and_dismissals_only(self):
        events = pd.DataFrame([
            dict(typeId=16, type_name='Goal', timeMin=23, timeSec=51, team_name='A', playerName='Scorer', **{'Red card': np.nan, 'Second yellow': np.nan}),
            dict(typeId=17, type_name='Card', timeMin=50, timeSec=0, team_name='B', playerName='Booked', **{'Red card': np.nan, 'Second yellow': np.nan}),
            dict(typeId=17, type_name='Card', timeMin=64, timeSec=0, team_name='B', playerName='Dismissed', **{'Red card': 1, 'Second yellow': np.nan}),
        ])

        key_events = extract_ppda_key_events(events)

        self.assertEqual(key_events['event_type'].tolist(), ['goal', 'red_card'])
        self.assertEqual(key_events['minute'].tolist(), [23.0, 64.0])


if __name__ == '__main__':
    unittest.main()
