import unittest

import pandas as pd

from src.metrics.restart_metrics import (
    classify_restart_event,
    extract_restart_sequences,
)
from src.metrics.set_piece_metrics import analyze_and_summarize_set_pieces


def event(
    event_id,
    team='Home',
    event_type='Pass',
    outcome='Successful',
    *,
    second=0.0,
    x=70.0,
    y=50.0,
    end_x=80.0,
    end_y=50.0,
    corner=0,
    free_kick=0,
    throw_in=0,
    goal_kick=0,
    direct_free_kick=0,
    cross=0,
):
    return {
        'id': event_id,
        'eventId': event_id,
        'team_name': team,
        'type_name': event_type,
        'outcome': outcome,
        'playerName': f'{team} Player {event_id}',
        'Mapped Jersey Number': event_id,
        'periodId': 1,
        'timeMin': 10,
        'timeSec': float(second),
        'x': x,
        'y': y,
        'end_x': end_x,
        'end_y': end_y,
        'Corner taken': corner,
        'Free kick taken': free_kick,
        'Freekick taken': 0,
        'ThrowIn': throw_in,
        'Goal kick': goal_kick,
        'Free kick': direct_free_kick,
        'cross': cross,
        'Penalty': 0,
        'Own goal': 0,
        'Goal mouth y co-ordinate': 50.0,
        'Right footed': 1,
        'Left footed': 0,
        'In-swinger': 0,
        'Out-swinger': 0,
        'Straight': 0,
    }


class RestartTaxonomyTests(unittest.TestCase):

    def test_classifier_separates_four_restart_types(self):
        cases = [
            (event(1, corner=1), 'Corner'),
            (event(2, free_kick=1), 'Free Kick'),
            (event(3, throw_in=1), 'Throw-in'),
            (event(4, goal_kick=1), 'Goal Kick'),
        ]

        for row, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    classify_restart_event(pd.Series(row)),
                    expected,
                )

    def test_generic_out_is_not_a_restart(self):
        row = pd.Series(
            event(1, event_type='Out', outcome='Unsuccessful')
        )
        self.assertIsNone(classify_restart_event(row))
        self.assertEqual(
            extract_restart_sequences(pd.DataFrame([row]), 'Home'),
            [],
        )

    def test_out_followed_by_throw_in_is_classified_from_delivery(self):
        df = pd.DataFrame([
            event(1, team='Away', event_type='Out', outcome='Unsuccessful'),
            event(2, throw_in=1, second=2.0),
            event(3, event_type='Dispossessed', second=4.0),
        ])
        sequences = extract_restart_sequences(df, 'Home')
        self.assertEqual(len(sequences), 1)
        self.assertEqual(
            sequences[0].iloc[0]['type_of_initial_trigger'],
            'Throw-in',
        )

    def test_out_followed_by_goal_kick_is_goal_kick(self):
        df = pd.DataFrame([
            event(1, team='Away', event_type='Out', outcome='Unsuccessful'),
            event(2, goal_kick=1, second=2.0, x=5.0, end_x=35.0),
            event(3, team='Away', event_type='Interception', second=5.0),
        ])
        sequences = extract_restart_sequences(df, 'Home')
        analyzed, _ = analyze_and_summarize_set_pieces(sequences)
        self.assertEqual(
            analyzed.iloc[0]['Action Type'],
            'Goal Kick',
        )

    def test_all_restart_types_remain_separate_in_summary(self):
        df = pd.DataFrame([
            event(1, corner=1, second=1.0),
            event(2, team='Away', event_type='Interception', second=2.0),
            event(3, free_kick=1, second=10.0),
            event(4, team='Away', event_type='Interception', second=11.0),
            event(5, throw_in=1, second=20.0),
            event(6, team='Away', event_type='Interception', second=21.0),
            event(7, goal_kick=1, second=30.0, x=5.0, end_x=30.0),
            event(8, team='Away', event_type='Interception', second=31.0),
        ])
        sequences = extract_restart_sequences(df, 'Home')
        analyzed, stats = analyze_and_summarize_set_pieces(sequences)

        self.assertEqual(len(sequences), 4)
        self.assertEqual(
            set(analyzed['Action Type']),
            {'Corner', 'Free Kick', 'Throw-in', 'Goal Kick'},
        )
        self.assertEqual(
            stats['action_types'],
            {
                'Corner': 1,
                'Free Kick': 1,
                'Throw-in': 1,
                'Goal Kick': 1,
            },
        )

    def test_unsuccessful_delivery_is_lost_possession_for_each_restart(self):
        for flag_name in ('corner', 'free_kick', 'throw_in', 'goal_kick'):
            df = pd.DataFrame([
                event(
                    1,
                    outcome='Unsuccessful',
                    **{flag_name: 1},
                )
            ])
            seq = extract_restart_sequences(df, 'Home')[0]
            with self.subTest(restart=flag_name):
                self.assertEqual(
                    seq.iloc[-1]['sequence_outcome_type'],
                    'Lost Possessions',
                )

    def test_restart_does_not_follow_possession_to_later_shot(self):
        df = pd.DataFrame([
            event(1, corner=1),
            event(
                2,
                event_type='Miss',
                outcome='Unsuccessful',
                second=4.0,
                x=90.0,
                end_x=100.0,
            ),
        ])
        seq = extract_restart_sequences(df, 'Home')[0]
        self.assertEqual(seq['eventId'].tolist(), [1])
        self.assertEqual(
            seq.iloc[-1]['restart_execution_outcome'],
            'Successful Delivery',
        )
        self.assertEqual(
            seq.iloc[-1]['sequence_outcome_type'],
            'Possession Retained',
        )

    def test_restart_does_not_follow_possession_to_later_goal(self):
        df = pd.DataFrame([
            event(1, throw_in=1),
            event(
                2,
                event_type='Goal',
                second=5.0,
                x=90.0,
                end_x=100.0,
            ),
        ])
        seq = extract_restart_sequences(df, 'Home')[0]
        self.assertEqual(seq['eventId'].tolist(), [1])
        self.assertEqual(
            seq.iloc[-1]['restart_execution_outcome'],
            'Successful Delivery',
        )
        self.assertEqual(
            seq.iloc[-1]['sequence_outcome_type'],
            'Possession Retained',
        )

    def test_direct_free_kick_shot_is_supported(self):
        df = pd.DataFrame([
            event(
                1,
                event_type='Goal',
                x=82.0,
                end_x=100.0,
                direct_free_kick=1,
            ),
        ])
        sequences = extract_restart_sequences(df, 'Home')
        analyzed, _ = analyze_and_summarize_set_pieces(sequences)

        self.assertEqual(
            sequences[0].iloc[0]['type_of_initial_trigger'],
            'Free Kick',
        )
        self.assertEqual(analyzed.iloc[0]['Action Type'], 'Free Kick')
        self.assertEqual(analyzed.iloc[0]['Delivery'], 'Direct Shot')
        self.assertEqual(analyzed.iloc[0]['Outcome'], 'Goal')

    def test_nearby_restarts_do_not_merge(self):
        df = pd.DataFrame([
            event(1, throw_in=1),
            event(2, free_kick=1, second=3.0),
            event(3, event_type='Dispossessed', second=5.0),
        ])
        sequences = extract_restart_sequences(df, 'Home')
        self.assertEqual(
            [seq.iloc[0]['restart_type'] for seq in sequences],
            ['Throw-in', 'Free Kick'],
        )


if __name__ == '__main__':
    unittest.main()
