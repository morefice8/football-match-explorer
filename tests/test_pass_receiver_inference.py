import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_processing.pass_processing import (  # noqa: E402
    infer_pass_receivers,
    receiver_coverage_summary,
)
from src.metrics.pass_metrics import calculate_pass_network_data  # noqa: E402


def event(
    event_id,
    type_id,
    team,
    player,
    minute,
    second,
    x,
    y,
    *,
    outcome='Successful',
    end_x=None,
    end_y=None,
    period=1,
    player_id=None,
    jersey=None,
):
    return {
        'id': event_id,
        'eventId': event_id,
        'typeId': type_id,
        'type_name': 'Pass' if type_id == 1 else 'Technical event',
        'contestantId': team,
        'team_name': team,
        'playerId': player_id or (f'{team}-{player}' if player else None),
        'playerName': player,
        'Mapped Jersey Number': jersey,
        'periodId': period,
        'timeMin': minute,
        'timeSec': second,
        'x': x,
        'y': y,
        'end_x': end_x,
        'end_y': end_y,
        'outcome': outcome,
    }


class PassReceiverInferenceTests(unittest.TestCase):
    def test_skips_administrative_event_and_assigns_same_team_receiver(self):
        df = pd.DataFrame([
            event(1, 1, 'A', 'Passer', 10, 0, 40, 40, end_x=60, end_y=50),
            event(2, 17, 'B', 'Booked player', 10, 2, 0, 0),
            event(3, 1, 'A', 'Receiver', 10, 4, 61, 51, end_x=70, end_y=55, jersey=9),
        ])

        result = infer_pass_receivers(df)

        self.assertEqual(result.loc[0, 'receiver'], 'Receiver')
        self.assertEqual(result.loc[0, 'receiver_jersey_number'], 9)
        self.assertEqual(result.loc[0, 'receiver_confidence'], 'high')
        self.assertTrue(result.loc[0, 'receiver_is_reliable'])

    def test_does_not_assign_opponent_after_successful_pass(self):
        df = pd.DataFrame([
            event(1, 1, 'A', 'Passer', 20, 0, 55, 50, end_x=80, end_y=50),
            event(2, 7, 'B', 'Opponent', 20, 2, 20, 50),
            event(3, 1, 'A', 'Later teammate', 20, 4, 81, 51, end_x=85, end_y=52),
        ])

        result = infer_pass_receivers(df)

        self.assertTrue(pd.isna(result.loc[0, 'receiver']))
        self.assertFalse(result.loc[0, 'receiver_is_reliable'])
        self.assertEqual(result.loc[0, 'receiver_reason'], 'possession_changed_before_next_touch')

    def test_does_not_assign_receiver_to_unsuccessful_pass(self):
        df = pd.DataFrame([
            event(1, 1, 'A', 'Passer', 30, 0, 40, 40, outcome='Unsuccessful', end_x=60, end_y=50),
            event(2, 1, 'A', 'Teammate', 30, 2, 60, 50, end_x=70, end_y=50),
        ])

        result = infer_pass_receivers(df)

        self.assertTrue(pd.isna(result.loc[0, 'receiver']))
        self.assertEqual(result.loc[0, 'receiver_reason'], 'unsuccessful_pass')

    def test_marks_plausible_longer_gap_as_medium_confidence(self):
        df = pd.DataFrame([
            event(1, 1, 'A', 'Passer', 40, 0, 40, 50, end_x=60, end_y=50),
            event(2, 3, 'A', 'Receiver', 40, 7, 78, 50),
        ])

        result = infer_pass_receivers(df)

        self.assertEqual(result.loc[0, 'receiver'], 'Receiver')
        self.assertEqual(result.loc[0, 'receiver_confidence'], 'medium')
        self.assertTrue(result.loc[0, 'receiver_is_reliable'])

    def test_rejects_spatially_incoherent_next_event(self):
        df = pd.DataFrame([
            event(1, 1, 'A', 'Passer', 50, 0, 40, 50, end_x=60, end_y=50, period=2),
            event(2, 1, 'A', 'Teammate', 50, 5, 95, 5, end_x=90, end_y=10, period=2),
        ])

        result = infer_pass_receivers(df)

        self.assertTrue(pd.isna(result.loc[0, 'receiver']))
        self.assertEqual(result.loc[0, 'receiver_reason'], 'spatial_mismatch')

    def test_coverage_reports_only_successful_passes_as_eligible(self):
        passes = pd.DataFrame([
            {'outcome': 'Successful', 'receiver': 'A', 'receiver_is_reliable': True, 'receiver_confidence': 'high'},
            {'outcome': 'Successful', 'receiver': 'B', 'receiver_is_reliable': True, 'receiver_confidence': 'medium'},
            {'outcome': 'Successful', 'receiver': pd.NA, 'receiver_is_reliable': False, 'receiver_confidence': pd.NA},
            {'outcome': 'Unsuccessful', 'receiver': pd.NA, 'receiver_is_reliable': False, 'receiver_confidence': pd.NA},
        ])

        summary = receiver_coverage_summary(passes)

        self.assertEqual(summary['eligible'], 3)
        self.assertEqual(summary['resolved'], 2)
        self.assertEqual(summary['high'], 1)
        self.assertEqual(summary['medium'], 1)
        self.assertAlmostEqual(summary['coverage_pct'], 200 / 3)

    def test_pass_network_excludes_unreliable_receiver_links(self):
        passes = pd.DataFrame([
            {
                'id': 1, 'team_name': 'A', 'playerName': 'Passer',
                'receiver': 'Reliable receiver', 'receiver_is_reliable': True,
                'x': 40, 'y': 50, 'Mapped Jersey Number': 8,
            },
            {
                'id': 2, 'team_name': 'A', 'playerName': 'Passer',
                'receiver': 'Opponent', 'receiver_is_reliable': False,
                'x': 45, 'y': 52, 'Mapped Jersey Number': 8,
            },
            {
                'id': 3, 'team_name': 'A', 'playerName': 'Reliable receiver',
                'receiver': 'Passer', 'receiver_is_reliable': True,
                'x': 60, 'y': 50, 'Mapped Jersey Number': 9,
            },
        ])

        links, _ = calculate_pass_network_data(passes, 'A')

        self.assertEqual(len(links), 1)
        self.assertEqual(int(links.iloc[0]['pass_count']), 2)
        self.assertNotIn('Opponent', links[['player1', 'player2']].to_numpy())


if __name__ == '__main__':
    unittest.main()
