import unittest

import pandas as pd

from src.metrics.sequence_outcome_metrics import (
    summarize_sequences,
)


def buildup_event(
    event_id,
    sequence_id,
    event_type='Pass',
    outcome='Successful',
    x=20,
    y=50,
    end_x=30,
    end_y=50,
    minute=1,
    second=0,
    final_outcome='Lost Possessions',
    pass_count=1,
):
    return {
        'id': event_id,
        'eventId': event_id,
        'trigger_sequence_id': sequence_id,
        'team_name': 'Home',
        'type_name': event_type,
        'outcome': outcome,
        'x': x,
        'y': y,
        'end_x': end_x,
        'end_y': end_y,
        'timeMin': minute,
        'timeSec': second,
        'trigger_zone': 'Defensive Third',
        'type_of_initial_trigger': 'Keeper pick-up',
        'timeMin_at_trigger': 1,
        'timeSec_at_trigger': 0,
        'buildup_pass_count': pass_count,
        'sequence_outcome_type': final_outcome,
    }


def transition_event(
    event_id,
    sequence_id,
    event_type='Pass',
    outcome='Successful',
    x=45,
    y=50,
    end_x=55,
    end_y=50,
    minute=10,
    second=0,
    final_outcome='Lost Possessions',
    pass_count=1,
):
    return {
        'id': event_id,
        'eventId': event_id,
        'loss_sequence_id': sequence_id,
        'team_name': 'Away',
        'type_name': event_type,
        'outcome': outcome,
        'x': x,
        'y': y,
        'end_x': end_x,
        'end_y': end_y,
        'timeMin': minute,
        'timeSec': second,
        'loss_zone': 'Middle Third',
        'type_of_initial_loss': 'Pass Interception',
        'timeMin_at_loss': 10,
        'timeSec_at_loss': 0,
        'opponent_pass_count': pass_count,
        'sequence_outcome_type': final_outcome,
    }


class SequenceSummaryTests(unittest.TestCase):

    def test_empty_input_returns_empty_summary(self):
        result = summarize_sequences(
            pd.DataFrame(),
            sequence_kind='buildup',
        )

        self.assertTrue(result.empty)

    def test_buildup_is_collapsed_to_one_row(self):
        df = pd.DataFrame([
            buildup_event(
                1,
                0,
                x=20,
                end_x=45,
                second=2,
                pass_count=2,
            ),
            buildup_event(
                2,
                0,
                x=45,
                end_x=72,
                second=7,
                pass_count=2,
            ),
        ])

        result = summarize_sequences(
            df,
            sequence_kind='buildup',
        )

        self.assertEqual(len(result), 1)

        row = result.iloc[0]

        self.assertEqual(
            row['sequence_id'],
            0,
        )

        self.assertEqual(
            row['pass_count'],
            2,
        )

        self.assertEqual(
            row['event_count'],
            2,
        )

        self.assertEqual(
            row['duration_seconds'],
            5.0,
        )

        self.assertEqual(
            row['restart_delay_seconds'],
            2.0,
        )

        self.assertTrue(
            row['reached_final_third']
        )

    def test_multiple_transitions_create_multiple_rows(self):
        df = pd.DataFrame([
            transition_event(
                1,
                0,
                second=2,
            ),
            transition_event(
                2,
                0,
                second=5,
            ),
            transition_event(
                3,
                1,
                second=20,
            ),
        ])

        # Second sequence starts at 10:00 in the helper too;
        # duration itself is irrelevant for this test.
        result = summarize_sequences(
            df,
            sequence_kind='offensive_transition',
        )

        self.assertEqual(
            len(result),
            2,
        )

        self.assertEqual(
            result['sequence_id'].tolist(),
            [0, 1],
        )

    def test_failed_pass_into_box_does_not_create_box_milestone(self):
        df = pd.DataFrame([
            transition_event(
                1,
                0,
                outcome='Unsuccessful',
                x=60,
                end_x=90,
                end_y=50,
                second=5,
                final_outcome='Lost Possessions',
                pass_count=0,
            ),
        ])

        result = summarize_sequences(
            df,
            sequence_kind='offensive_transition',
        )

        row = result.iloc[0]

        self.assertFalse(
            row['entered_penalty_area']
        )

        self.assertFalse(
            row['reached_final_third']
        )

        self.assertEqual(
            row['final_outcome'],
            'Lost Possessions',
        )

    def test_shot_sequence_preserves_outcome_and_milestones(self):
        df = pd.DataFrame([
            transition_event(
                1,
                0,
                x=45,
                end_x=72,
                second=3,
                pass_count=1,
                final_outcome='Shots',
            ),
            transition_event(
                2,
                0,
                event_type='Miss',
                outcome='Unsuccessful',
                x=88,
                y=50,
                end_x=100,
                end_y=50,
                second=8,
                pass_count=1,
                final_outcome='Shots',
            ),
        ])

        result = summarize_sequences(
            df,
            sequence_kind='offensive_transition',
        )

        row = result.iloc[0]

        self.assertEqual(
            row['final_outcome'],
            'Shots',
        )

        self.assertTrue(
            row['reached_final_third']
        )

        self.assertTrue(
            row['entered_penalty_area']
        )

        self.assertTrue(
            row['produced_shot']
        )

        self.assertFalse(
            row['produced_goal']
        )

        self.assertEqual(
            row['duration_seconds'],
            8.0,
        )

    def test_goal_sequence_is_marked_as_goal(self):
        df = pd.DataFrame([
            transition_event(
                1,
                0,
                event_type='Goal',
                x=88,
                y=50,
                end_x=100,
                end_y=50,
                second=6,
                final_outcome='Goals',
                pass_count=0,
            ),
        ])

        result = summarize_sequences(
            df,
            sequence_kind='offensive_transition',
        )

        row = result.iloc[0]

        self.assertTrue(
            row['produced_shot']
        )

        self.assertTrue(
            row['produced_goal']
        )

    def test_failed_first_buildup_pass_is_valid_zero_pass_sequence(
        self
    ):
        df = pd.DataFrame([
            buildup_event(
                1,
                0,
                outcome='Unsuccessful',
                x=25,
                end_x=70,
                minute=1,
                second=30,
                pass_count=0,
                final_outcome='Lost Possessions',
            ),
        ])

        result = summarize_sequences(
            df,
            sequence_kind='buildup',
        )

        self.assertEqual(
            len(result),
            1,
        )

        row = result.iloc[0]

        self.assertEqual(
            row['pass_count'],
            0,
        )

        self.assertEqual(
            row['event_count'],
            1,
        )

        self.assertEqual(
            row['duration_seconds'],
            0.0,
        )

        self.assertEqual(
            row['restart_delay_seconds'],
            30.0,
        )

        self.assertEqual(
            row['final_outcome'],
            'Lost Possessions',
        )

if __name__ == '__main__':
    unittest.main()