import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.pass_metrics import (  # noqa: E402
    classify_progressive_passes,
    progressive_pass_player_summary,
    progressive_pass_summary,
)


def pass_event(
    event_id,
    player,
    x,
    y,
    end_x,
    end_y,
    *,
    outcome='Successful',
    **qualifiers,
):
    return {
        'id': event_id,
        'type_name': 'Pass',
        'team_name': 'A',
        'playerName': player,
        'outcome': outcome,
        'x': x,
        'y': y,
        'end_x': end_x,
        'end_y': end_y,
        **qualifiers,
    }


class ProgressivePassMetricTests(unittest.TestCase):
    def test_uses_distance_to_goal_not_x_gain_only(self):
        df = pd.DataFrame([
            # Advances in x but moves so far wide that it gets farther from goal.
            pass_event(1, 'False positive', 60, 50, 72, 0),
            # Gains only eight x units but moves infield and much closer to goal.
            pass_event(2, 'True progression', 60, 0, 68, 30),
        ])

        result = classify_progressive_passes(df).set_index('id')

        self.assertFalse(result.loc[1, 'is_progressive_attempt'])
        self.assertLess(result.loc[1, 'progressive_distance_m'], 0)
        self.assertTrue(result.loc[2, 'is_progressive_attempt'])
        self.assertTrue(result.loc[2, 'is_progressive'])
        self.assertGreaterEqual(result.loc[2, 'progressive_distance_m'], 10)

    def test_applies_phase_specific_thresholds(self):
        df = pd.DataFrame([
            pass_event(1, 'Own half', 10, 50, 40, 50),
            pass_event(2, 'Across halfway', 40, 50, 55, 50),
            pass_event(3, 'Opposition half', 60, 50, 70, 50),
        ])

        result = classify_progressive_passes(df).set_index('id')

        self.assertEqual(result.loc[1, 'progressive_threshold_m'], 30)
        self.assertEqual(result.loc[2, 'progressive_threshold_m'], 15)
        self.assertEqual(result.loc[3, 'progressive_threshold_m'], 10)
        self.assertTrue(result['is_progressive'].all())

    def test_excludes_crosses_and_restarts_but_keeps_open_play_long_balls(self):
        df = pd.DataFrame([
            pass_event(1, 'Cross', 60, 50, 85, 50, cross=1),
            pass_event(2, 'Throw in', 60, 50, 85, 50, ThrowIn='1'),
            pass_event(3, 'Free kick', 60, 50, 85, 50, **{'Free kick taken': True}),
            pass_event(4, 'Goal kick', 10, 50, 60, 50, **{'Goal kick': 1}),
            pass_event(5, 'Open-play long ball', 10, 50, 60, 50, lb=1),
        ])

        result = classify_progressive_passes(df).set_index('id')

        self.assertFalse(result.loc[[1, 2, 3, 4], 'is_progressive_attempt'].any())
        self.assertEqual(result.loc[1, 'progressive_exclusion_reason'], 'cross')
        self.assertEqual(result.loc[2, 'progressive_exclusion_reason'], 'throw_in')
        self.assertTrue(result.loc[5, 'is_progressive_attempt'])
        self.assertTrue(result.loc[5, 'progressive_is_open_play'])

    def test_keeps_attempts_and_completions_separate(self):
        df = pd.DataFrame([
            pass_event(1, 'Player A', 60, 50, 75, 50),
            pass_event(2, 'Player A', 60, 50, 75, 50, outcome='Unsuccessful'),
            pass_event(3, 'Player B', 10, 50, 45, 50),
        ])
        classified = classify_progressive_passes(df)

        summary = progressive_pass_summary(classified)
        players = progressive_pass_player_summary(classified)

        self.assertEqual(summary['attempted'], 3)
        self.assertEqual(summary['successful'], 2)
        self.assertAlmostEqual(summary['completion_pct'], 200 / 3)
        player_a = players.set_index('Player').loc['Player A']
        self.assertEqual(player_a['Successful'], 1)
        self.assertEqual(player_a['Attempted'], 2)
        self.assertEqual(player_a['Completion %'], 50)

    def test_assigns_origin_channel(self):
        df = pd.DataFrame([
            pass_event(1, 'Left', 60, 80, 80, 60),
            pass_event(2, 'Center', 60, 50, 80, 50),
            pass_event(3, 'Right', 60, 20, 80, 40),
        ])

        result = classify_progressive_passes(df).set_index('id')

        self.assertEqual(result.loc[1, 'progressive_channel'], 'Left')
        self.assertEqual(result.loc[2, 'progressive_channel'], 'Central')
        self.assertEqual(result.loc[3, 'progressive_channel'], 'Right')


if __name__ == '__main__':
    unittest.main()
