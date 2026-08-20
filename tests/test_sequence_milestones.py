import unittest

import pandas as pd

from src.metrics.sequence_outcome_metrics import (
    calculate_sequence_milestones,
)


def event(
    event_id,
    event_type='Pass',
    outcome='Successful',
    x=20,
    y=50,
    end_x=30,
    end_y=50,
):
    return {
        'id': event_id,
        'eventId': event_id,
        'team_name': 'Home',
        'type_name': event_type,
        'outcome': outcome,
        'x': x,
        'y': y,
        'end_x': end_x,
        'end_y': end_y,
    }


class SequenceMilestoneTests(
    unittest.TestCase
):

    def test_empty_sequence_returns_no_milestones(
        self
    ):
        result = calculate_sequence_milestones(
            pd.DataFrame()
        )

        self.assertFalse(
            result['reached_final_third']
        )
        self.assertFalse(
            result['entered_penalty_area']
        )
        self.assertFalse(
            result['produced_shot']
        )

    def test_successful_progression_reaches_final_third(
        self
    ):
        df = pd.DataFrame([
            event(
                1,
                x=20,
                end_x=45,
            ),
            event(
                2,
                x=45,
                end_x=70,
            ),
        ])

        result = calculate_sequence_milestones(
            df
        )

        self.assertTrue(
            result['reached_middle_third']
        )
        self.assertTrue(
            result['reached_opposition_half']
        )
        self.assertTrue(
            result['reached_final_third']
        )

        self.assertEqual(
            result['max_controlled_x'],
            70.0,
        )

    def test_failed_long_pass_does_not_count_as_reached_zone(
        self
    ):
        df = pd.DataFrame([
            event(
                1,
                outcome='Unsuccessful',
                x=40,
                end_x=90,
                end_y=50,
            ),
        ])

        result = calculate_sequence_milestones(
            df
        )

        self.assertEqual(
            result['max_controlled_x'],
            40.0,
        )

        self.assertFalse(
            result['reached_opposition_half']
        )

        self.assertFalse(
            result['reached_final_third']
        )

        self.assertFalse(
            result['entered_penalty_area']
        )

    def test_successful_pass_into_box_counts_as_box_entry(
        self
    ):
        df = pd.DataFrame([
            event(
                1,
                outcome='Successful',
                x=70,
                end_x=90,
                end_y=50,
            ),
        ])

        result = calculate_sequence_milestones(
            df
        )

        self.assertTrue(
            result['reached_final_third']
        )

        self.assertTrue(
            result['entered_penalty_area']
        )

    def test_action_starting_inside_box_counts_as_box_entry(
        self
    ):
        df = pd.DataFrame([
            event(
                1,
                event_type='Miss',
                outcome='Unsuccessful',
                x=90,
                y=45,
                end_x=100,
                end_y=50,
            ),
        ])

        result = calculate_sequence_milestones(
            df
        )

        self.assertTrue(
            result['entered_penalty_area']
        )

        self.assertTrue(
            result['produced_shot']
        )

        self.assertFalse(
            result['produced_goal']
        )

    def test_goal_is_shot_and_goal(
        self
    ):
        df = pd.DataFrame([
            event(
                1,
                event_type='Goal',
                outcome='Successful',
                x=88,
                y=50,
                end_x=100,
                end_y=50,
            ),
        ])

        result = calculate_sequence_milestones(
            df
        )

        self.assertTrue(
            result['produced_shot']
        )

        self.assertTrue(
            result['produced_goal']
        )


if __name__ == '__main__':
    unittest.main()