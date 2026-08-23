import inspect
import unittest

import pandas as pd

from src.metrics import transition_metrics


def make_event(
    event_id,
    team,
    event_type,
    outcome='Successful',
    *,
    x=50.0,
    y=50.0,
    end_x=50.0,
    end_y=50.0,
    minute=1,
    second=0.0,
    period=1,
    red_card=0,
    second_yellow=0,
    out_of_play=0,
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
        'playerName': f'Player {event_id}',
        'Mapped Jersey Number': event_id,
        'timeMin': minute,
        'timeSec': second,
        'periodId': period,
        'Penalty': 0,
        'Own goal': 0,
        'From corner': 0,
        'Goal mouth y co-ordinate': 50.0,
        'Red card': red_card,
        'Second yellow': second_yellow,
        'Out of play': out_of_play,
    }


def find_transition(
    rows,
    *,
    losing_team='Home',
    metric='offensive_transitions',
    **kwargs,
):
    return transition_metrics.find_buildup_after_possession_loss(
        pd.DataFrame(rows),
        team_that_lost_possession=losing_team,
        metric_to_analyze=metric,
        **kwargs,
    )


class TransitionEdgeCaseTests(unittest.TestCase):
    def test_default_window_is_backed_by_module_constant(self):
        self.assertEqual(
            transition_metrics.TRANSITION_WINDOW_SECONDS,
            12.0,
        )

        default_value = inspect.signature(
            transition_metrics.find_buildup_after_possession_loss
        ).parameters['max_transition_seconds'].default

        self.assertEqual(
            default_value,
            transition_metrics.TRANSITION_WINDOW_SECONDS,
        )

    def test_immediate_recovery_and_regain_is_zero_pass_turnover(self):
        rows = [
            make_event(
                1,
                'Home',
                'Pass',
                'Unsuccessful',
                second=0.0,
            ),
            make_event(
                2,
                'Away',
                'Ball recovery',
                second=0.5,
            ),
            make_event(
                3,
                'Home',
                'Tackle',
                second=1.0,
            ),
        ]

        offensive = find_transition(rows)
        defensive = find_transition(
            rows,
            metric='defensive_transitions',
        )

        self.assertFalse(offensive.empty)
        self.assertFalse(defensive.empty)
        self.assertEqual(
            offensive['sequence_outcome_type'].iloc[-1],
            'Lost Possessions',
        )
        self.assertEqual(
            defensive['sequence_outcome_type'].iloc[-1],
            'Regained Possessions',
        )
        self.assertEqual(
            offensive['terminal_outcome'].iloc[-1],
            'turnover',
        )
        self.assertEqual(
            offensive['opponent_pass_count'].iloc[-1],
            0,
        )
        self.assertIn(
            'Ball recovery',
            offensive['type_name'].tolist(),
        )

    def test_tackle_in_play_confirms_transition(self):
        result = find_transition([
            make_event(
                1,
                'Home',
                'Pass',
                'Unsuccessful',
                second=0.0,
            ),
            make_event(
                2,
                'Away',
                'Tackle',
                'Successful',
                second=0.5,
                out_of_play=0,
            ),
            make_event(
                3,
                'Away',
                'Corner Awarded',
                'Successful',
                second=1.0,
            ),
        ])

        self.assertFalse(result.empty)
        self.assertIn(
            2,
            result['eventId'].tolist(),
        )
        self.assertEqual(
            result['sequence_outcome_type'].iloc[-1],
            'Corner',
        )

    def test_tackle_out_of_play_does_not_confirm_transition(self):
        result = find_transition([
            make_event(
                1,
                'Home',
                'Pass',
                'Unsuccessful',
                second=0.0,
            ),
            make_event(
                2,
                'Away',
                'Tackle',
                'Successful',
                second=0.5,
                out_of_play=1,
            ),
            make_event(
                3,
                'Away',
                'Corner Awarded',
                'Successful',
                second=1.0,
            ),
        ])

        self.assertTrue(result.empty)

    def test_double_possession_change_creates_two_separate_transitions(self):
        rows = [
            make_event(1, 'Home', 'Pass', 'Unsuccessful', second=0.0),
            make_event(2, 'Away', 'Ball recovery', second=1.0),
            make_event(3, 'Away', 'Pass', 'Unsuccessful', second=2.0),
            make_event(4, 'Home', 'Ball recovery', second=3.0),
            make_event(
                5,
                'Home',
                'Goal',
                second=5.0,
                x=85.0,
                end_x=100.0,
            ),
        ]

        first_transition = find_transition(rows)
        second_transition = find_transition(rows, losing_team='Away')

        self.assertFalse(first_transition.empty)
        self.assertEqual(
            first_transition['sequence_outcome_type'].iloc[-1],
            'Lost Possessions',
        )
        self.assertNotIn(5, first_transition['eventId'].tolist())

        self.assertFalse(second_transition.empty)
        self.assertEqual(
            second_transition['sequence_outcome_type'].iloc[-1],
            'Goals',
        )
        self.assertIn(5, second_transition['eventId'].tolist())

    def test_keeper_claim_closes_transition_as_turnover(self):
        rows = [
            make_event(
                1,
                'Home',
                'Pass',
                'Unsuccessful',
                second=0.0,
            ),
            make_event(
                2,
                'Away',
                'Ball recovery',
                second=1.0,
            ),
            make_event(
                3,
                'Home',
                'Claim',
                'Successful',
                second=3.0,
            ),
        ]

        offensive = find_transition(rows)

        defensive = find_transition(
            rows,
            metric='defensive_transitions',
        )

        self.assertFalse(offensive.empty)
        self.assertFalse(defensive.empty)

        self.assertEqual(
            offensive[
                'sequence_outcome_type'
            ].iloc[-1],
            'Lost Possessions',
        )

        self.assertEqual(
            defensive[
                'sequence_outcome_type'
            ].iloc[-1],
            'Regained Possessions',
        )

        self.assertEqual(
            offensive[
                'terminal_outcome'
            ].iloc[-1],
            'turnover',
        )

    def test_foul_requires_confirmed_recovery(self):
        immediate = find_transition([
            make_event(
                1,
                'Home',
                'Pass',
                'Unsuccessful',
                second=0.0,
            ),
            make_event(
                2,
                'Home',
                'Foul',
                'Unsuccessful',
                second=1.0,
            ),
            make_event(
                3,
                'Away',
                'Unknown',
                second=2.0,
            ),
        ])

        self.assertTrue(immediate.empty)

        result = find_transition([
            make_event(
                1,
                'Home',
                'Pass',
                'Unsuccessful',
                second=0.0,
            ),
            make_event(
                2,
                'Away',
                'Ball recovery',
                second=0.5,
            ),
            make_event(
                3,
                'Home',
                'Foul',
                'Unsuccessful',
                second=1.0,
            ),
        ])

        self.assertFalse(result.empty)
        self.assertEqual(
            result['sequence_outcome_type'].iloc[-1],
            'Foul',
        )
        self.assertEqual(
            result['terminal_outcome'].iloc[-1],
            'foul',
        )
        self.assertEqual(
            result['termination_reason'].iloc[-1],
            'foul',
        )
        self.assertEqual(
            result['eventId'].tolist(),
            [2, 3],
        )

    def test_out_requires_confirmed_recovery(self):
        immediate = find_transition([
            make_event(
                1,
                'Home',
                'Pass',
                'Unsuccessful',
                second=0.0,
            ),
            make_event(
                2,
                'Away',
                'Out',
                'Unsuccessful',
                second=1.0,
            ),
        ])

        self.assertTrue(immediate.empty)

        result = find_transition([
            make_event(
                1,
                'Home',
                'Pass',
                'Unsuccessful',
                second=0.0,
            ),
            make_event(
                2,
                'Away',
                'Ball recovery',
                second=0.5,
            ),
            make_event(
                3,
                'Away',
                'Out',
                'Unsuccessful',
                second=1.0,
            ),
        ])

        self.assertFalse(result.empty)
        self.assertEqual(
            result['sequence_outcome_type'].iloc[-1],
            'Out',
        )
        self.assertEqual(
            result['terminal_outcome'].iloc[-1],
            'out',
        )
        self.assertEqual(
            result['termination_reason'].iloc[-1],
            'ball_out',
        )
        self.assertEqual(
            result['eventId'].tolist(),
            [2, 3],
        )

    def test_dismissal_card_does_not_change_possession(self):
        result = find_transition([
            make_event(1, 'Home', 'Pass', 'Unsuccessful', second=0.0),
            make_event(2, 'Away', 'Ball recovery', second=1.0),
            make_event(3, 'Home', 'Card', second=2.0, red_card=1),
            make_event(4, 'Away', 'Pass', second=3.0, end_x=70.0),
            make_event(
                5,
                'Away',
                'Miss',
                'Unsuccessful',
                second=5.0,
                x=80.0,
                end_x=100.0,
            ),
        ])

        self.assertFalse(result.empty)
        self.assertEqual(result['sequence_outcome_type'].iloc[-1], 'Shots')
        self.assertEqual(result['terminal_outcome'].iloc[-1], 'shot')
        self.assertNotIn(3, result['eventId'].tolist())

    def test_period_boundary_stops_sequence_and_is_explicit_reason(self):
        result = find_transition([
            make_event(
                1, 'Home', 'Pass', 'Unsuccessful', second=0.0, period=1
            ),
            make_event(2, 'Away', 'Ball recovery', second=1.0, period=1),
            make_event(
                3,
                'Away',
                'Pass',
                second=2.0,
                period=1,
                end_x=65.0,
            ),
            make_event(
                4,
                'Away',
                'Goal',
                second=3.0,
                period=2,
                x=85.0,
                end_x=100.0,
            ),
        ])

        self.assertFalse(result.empty)
        self.assertNotIn(4, result['eventId'].tolist())
        self.assertEqual(
            result['termination_reason'].iloc[-1],
            'period_boundary',
        )
        self.assertNotEqual(result['terminal_outcome'].iloc[-1], 'goal')

    def test_shot_just_inside_default_window_counts(self):
        result = find_transition([
            make_event(1, 'Home', 'Pass', 'Unsuccessful', second=0.0),
            make_event(2, 'Away', 'Ball recovery', second=0.5),
            make_event(
                3,
                'Away',
                'Miss',
                'Unsuccessful',
                second=11.9,
                x=80.0,
                end_x=100.0,
            ),
        ])

        self.assertFalse(result.empty)
        self.assertEqual(result['sequence_outcome_type'].iloc[-1], 'Shots')
        self.assertEqual(result['terminal_outcome'].iloc[-1], 'shot')
        self.assertIn(3, result['eventId'].tolist())

    def test_shot_just_outside_default_window_is_excluded(self):
        result = find_transition([
            make_event(1, 'Home', 'Pass', 'Unsuccessful', second=0.0),
            make_event(2, 'Away', 'Ball recovery', second=0.5),
            make_event(3, 'Away', 'Pass', second=5.0, end_x=70.0),
            make_event(
                4,
                'Away',
                'Miss',
                'Unsuccessful',
                second=12.1,
                x=80.0,
                end_x=100.0,
            ),
        ])

        self.assertFalse(result.empty)
        self.assertEqual(
            result['sequence_outcome_type'].iloc[-1],
            'Possession Consolidated',
        )
        self.assertEqual(result['terminal_outcome'].iloc[-1], 'consolidated')
        self.assertEqual(
            result['termination_reason'].iloc[-1],
            'time_window_elapsed',
        )
        self.assertNotIn(4, result['eventId'].tolist())

    def test_custom_window_still_overrides_default_constant(self):
        result = find_transition(
            [
                make_event(1, 'Home', 'Pass', 'Unsuccessful', second=0.0),
                make_event(2, 'Away', 'Ball recovery', second=1.0),
                make_event(
                    3,
                    'Away',
                    'Goal',
                    second=15.0,
                    x=85.0,
                    end_x=100.0,
                ),
            ],
            max_transition_seconds=20.0,
        )

        self.assertFalse(result.empty)
        self.assertEqual(result['sequence_outcome_type'].iloc[-1], 'Goals')
        self.assertEqual(result['terminal_outcome'].iloc[-1], 'goal')

    def test_unknown_type_does_not_end_transition(self):
        result = find_transition([
            make_event(
                1,
                'Home',
                'Pass',
                'Unsuccessful',
                second=0.0,
            ),
            make_event(
                2,
                'Away',
                'Ball recovery',
                second=0.5,
            ),
            make_event(
                3,
                'Away',
                'Unknown Type',
                'Unsuccessful',
                second=1.0,
            ),
            make_event(
                4,
                'Away',
                'Pass',
                'Successful',
                second=2.0,
                end_x=70.0,
            ),
            make_event(
                5,
                'Away',
                'Miss',
                'Unsuccessful',
                second=4.0,
                x=82.0,
                end_x=100.0,
            ),
        ])

        self.assertFalse(result.empty)

        self.assertEqual(
            result['sequence_outcome_type'].iloc[-1],
            'Shots',
        )

        self.assertEqual(
            result['terminal_outcome'].iloc[-1],
            'shot',
        )

        self.assertNotIn(
            3,
            result['eventId'].tolist(),
        )

        self.assertIn(
            5,
            result['eventId'].tolist(),
        )

    def test_opponent_terminal_event_is_rotated_into_transition_frame(self):
        result = find_transition([
            make_event(
                1,
                'Home',
                'Pass',
                'Unsuccessful',
                second=0.0,
            ),
            make_event(
                2,
                'Away',
                'Ball recovery',
                'Successful',
                x=20.0,
                y=30.0,
                second=1.0,
            ),
            make_event(
                3,
                'Home',
                'Foul',
                'Unsuccessful',
                x=80.0,
                y=70.0,
                second=2.0,
            ),
        ])

        self.assertFalse(result.empty)

        recovery = result[
            result['eventId'].eq(2)
        ].iloc[0]

        foul = result[
            result['eventId'].eq(3)
        ].iloc[0]

        self.assertEqual(
            float(recovery['x']),
            20.0,
        )
        self.assertEqual(
            float(recovery['y']),
            30.0,
        )

        # Home x=80/y=70 and Away x=20/y=30
        # describe the same physical location.
        self.assertEqual(
            float(foul['x']),
            20.0,
        )
        self.assertEqual(
            float(foul['y']),
            30.0,
        )

        self.assertEqual(
            result[
                'sequence_outcome_type'
            ].iloc[-1],
            'Foul',
        )


if __name__ == '__main__':
    unittest.main()
