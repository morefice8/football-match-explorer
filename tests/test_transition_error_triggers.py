import unittest

import pandas as pd

from src.metrics.transition_metrics import (
    find_buildup_after_possession_loss,
)


def event(
    event_id,
    team,
    event_type,
    outcome='Successful',
    *,
    second=0.0,
    period=2,
    x=50.0,
    y=50.0,
    end_x=60.0,
    end_y=50.0,
):
    return {
        'id': event_id,
        'eventId': event_id,
        'team_name': team,
        'type_name': event_type,
        'outcome': outcome,
        'x': x,
        'y': y,
        'end_x': end_x,
        'end_y': end_y,
        'playerName': f'{team} Player',
        'Mapped Jersey Number': 1,
        'timeMin': 80,
        'timeSec': float(second),
        'periodId': period,
        'Own goal': 0,
        'Penalty': 0,
        'From corner': 0,
        'Goal mouth y co-ordinate': 50.0,
        'Out of play': 0,
    }


def offensive_transitions_after_error(rows):
    df = pd.DataFrame(rows)

    return find_buildup_after_possession_loss(
        df,
        team_that_lost_possession='Genoa',
        possession_loss_types=['Error'],
        shot_types=['Goal', 'Miss', 'Attempt Saved', 'Post'],
        metric_to_analyze='offensive_transitions',
    )


class TransitionErrorTriggerTests(unittest.TestCase):

    def test_defensive_error_during_opponent_possession_is_not_new_transition(self):
        # Mirrors Vergara pass -> Vasquez Error -> De Bruyne goal.
        # Napoli already owns the possession before the defensive Error.
        result = offensive_transitions_after_error([
            event(
                1,
                'Napoli',
                'Pass',
                second=0.0,
                x=78.1,
                y=37.8,
                end_x=91.1,
                end_y=44.9,
            ),
            event(
                2,
                'Genoa',
                'Error',
                second=0.3,
                x=10.6,
                y=62.9,
            ),
            event(
                3,
                'Napoli',
                'Goal',
                second=1.8,
                x=91.1,
                y=44.9,
                end_x=100.0,
                end_y=52.7,
            ),
        ])

        self.assertTrue(result.empty)

    def test_error_after_same_team_control_is_valid_transition_trigger(self):
        result = offensive_transitions_after_error([
            event(
                1,
                'Genoa',
                'Pass',
                second=0.0,
                x=35.0,
                end_x=48.0,
            ),
            event(
                2,
                'Genoa',
                'Error',
                second=1.0,
                x=48.0,
                y=50.0,
            ),
            event(
                3,
                'Napoli',
                'Ball recovery',
                second=1.5,
                x=52.0,
                y=50.0,
            ),
            event(
                4,
                'Napoli',
                'Pass',
                second=4.0,
                x=55.0,
                end_x=78.0,
            ),
            event(
                5,
                'Napoli',
                'Goal',
                second=7.0,
                x=88.0,
                end_x=100.0,
            ),
        ])

        self.assertFalse(result.empty)
        self.assertEqual(
            result['sequence_outcome_type'].iloc[-1],
            'Goals',
        )
        self.assertEqual(
            result['terminal_outcome'].iloc[-1],
            'goal',
        )
        self.assertEqual(
            result['type_of_initial_loss'].iloc[0],
            'Error',
        )

    def test_failed_opponent_challenge_does_not_erase_prior_same_team_control(self):
        result = offensive_transitions_after_error([
            event(
                1,
                'Genoa',
                'Pass',
                second=0.0,
            ),
            event(
                2,
                'Napoli',
                'Tackle',
                'Unsuccessful',
                second=0.5,
            ),
            event(
                3,
                'Genoa',
                'Error',
                second=1.0,
            ),
            event(
                4,
                'Napoli',
                'Ball recovery',
                second=1.5,
            ),
            event(
                5,
                'Napoli',
                'Goal',
                second=5.0,
                x=90.0,
                end_x=100.0,
            ),
        ])

        self.assertFalse(result.empty)
        self.assertEqual(
            result['sequence_outcome_type'].iloc[-1],
            'Goals',
        )


if __name__ == '__main__':
    unittest.main()
