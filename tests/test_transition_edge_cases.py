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

    def test_shot_just_outside_base_window_completes_advanced_transition(self):
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
            'Shots',
        )
        self.assertEqual(result['terminal_outcome'].iloc[-1], 'shot')
        self.assertEqual(
            result['termination_reason'].iloc[-1],
            'terminal_action_grace',
        )
        self.assertIn(4, result['eventId'].tolist())

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

    def test_failed_opponent_tackle_after_dispossessed_keeps_transition_alive(self):
        rows = [
            make_event(
                1,
                'Home',
                'Pass',
                'Unsuccessful',
                second=0.0,
                end_x=65.0,
                end_y=18.0,
            ),
            make_event(
                2,
                'Away',
                'Interception',
                second=0.8,
                x=35.0,
                y=82.0,
            ),
            make_event(
                3,
                'Away',
                'Ball recovery',
                second=1.5,
                x=34.0,
                y=78.0,
            ),
            make_event(
                4,
                'Away',
                'Dispossessed',
                second=3.000,
                x=42.6,
                y=65.2,
            ),
            # Same physical duel in Home coordinates:
            # 100 - 57.4 = 42.6, 100 - 34.8 = 65.2.
            make_event(
                5,
                'Home',
                'Tackle',
                'Unsuccessful',
                second=3.001,
                x=57.4,
                y=34.8,
            ),
            make_event(
                6,
                'Away',
                'Pass',
                second=5.0,
                x=46.2,
                y=50.1,
                end_x=75.5,
                end_y=4.8,
            ),
            make_event(
                7,
                'Away',
                'Pass',
                second=8.0,
                x=81.7,
                y=9.4,
                end_x=89.3,
                end_y=66.8,
            ),
            make_event(
                8,
                'Away',
                'Goal',
                second=11.0,
                x=89.3,
                y=66.8,
                end_x=100.0,
                end_y=55.0,
            ),
        ]

        result = find_transition(rows)

        self.assertFalse(result.empty)
        self.assertEqual(
            result['sequence_outcome_type'].iloc[-1],
            'Goals',
        )
        self.assertEqual(
            result['terminal_outcome'].iloc[-1],
            'goal',
        )
        self.assertIn(4, result['eventId'].tolist())
        self.assertNotIn(5, result['eventId'].tolist())
        self.assertIn(8, result['eventId'].tolist())

    def test_successful_opponent_tackle_still_ends_dispossessed_transition(self):
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
                'Away',
                'Dispossessed',
                second=3.000,
                x=42.6,
                y=65.2,
            ),
            make_event(
                4,
                'Home',
                'Tackle',
                'Successful',
                second=3.001,
                x=57.4,
                y=34.8,
            ),
            make_event(
                5,
                'Away',
                'Goal',
                second=6.0,
                x=90.0,
                end_x=100.0,
            ),
        ]

        result = find_transition(rows)

        self.assertFalse(result.empty)
        self.assertEqual(
            result['sequence_outcome_type'].iloc[-1],
            'Lost Possessions',
        )
        self.assertEqual(
            result['terminal_outcome'].iloc[-1],
            'turnover',
        )
        self.assertNotIn(5, result['eventId'].tolist())

    def test_spatially_unrelated_failed_tackle_does_not_cancel_dispossessed(self):
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
                'Away',
                'Dispossessed',
                second=3.000,
                x=42.6,
                y=65.2,
            ),
            make_event(
                4,
                'Home',
                'Tackle',
                'Unsuccessful',
                second=3.001,
                # Rotates to 10, 10: far from the Dispossessed point.
                x=90.0,
                y=90.0,
            ),
            make_event(
                5,
                'Away',
                'Goal',
                second=6.0,
                x=90.0,
                end_x=100.0,
            ),
        ]

        result = find_transition(rows)

        self.assertFalse(result.empty)
        self.assertEqual(
            result['sequence_outcome_type'].iloc[-1],
            'Lost Possessions',
        )
        self.assertEqual(
            result['terminal_outcome'].iloc[-1],
            'turnover',
        )
        self.assertNotIn(5, result['eventId'].tolist())

    def test_terminal_grace_requires_final_third_progress_inside_base_window(self):
        result = find_transition([
            make_event(1, 'Home', 'Pass', 'Unsuccessful', second=0.0),
            make_event(2, 'Away', 'Ball recovery', second=0.5, x=30.0),
            make_event(
                3,
                'Away',
                'Pass',
                second=5.0,
                x=35.0,
                end_x=60.0,
            ),
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
        self.assertEqual(
            result['terminal_outcome'].iloc[-1],
            'consolidated',
        )
        self.assertEqual(
            result['termination_reason'].iloc[-1],
            'time_window_elapsed',
        )
        self.assertNotIn(4, result['eventId'].tolist())

    def test_terminal_grace_expires_after_four_seconds(self):
        result = find_transition([
            make_event(1, 'Home', 'Pass', 'Unsuccessful', second=0.0),
            make_event(2, 'Away', 'Ball recovery', second=0.5),
            make_event(3, 'Away', 'Pass', second=5.0, end_x=75.0),
            make_event(
                4,
                'Away',
                'Goal',
                second=16.1,
                x=88.0,
                end_x=100.0,
            ),
        ])

        self.assertFalse(result.empty)
        self.assertEqual(
            result['sequence_outcome_type'].iloc[-1],
            'Possession Consolidated',
        )
        self.assertEqual(
            result['terminal_outcome'].iloc[-1],
            'consolidated',
        )
        self.assertEqual(
            result['termination_reason'].iloc[-1],
            'time_window_elapsed',
        )
        self.assertNotIn(4, result['eventId'].tolist())

    def test_vergara_style_counterattack_goal_can_complete_in_terminal_grace(self):
        # Timing mirrors the real Sow -> Lucca recovery -> Vergara goal case:
        # the attack reaches the final third inside 12 seconds and the goal
        # arrives at ~15.3s without an intervening change of possession.
        result = find_transition([
            make_event(
                1,
                'Home',
                'Pass',
                'Unsuccessful',
                second=0.000,
                end_x=65.9,
                end_y=18.5,
            ),
            make_event(
                2,
                'Away',
                'Interception',
                second=0.787,
                x=22.8,
                y=83.3,
            ),
            make_event(
                3,
                'Away',
                'Ball recovery',
                second=1.595,
                x=34.1,
                y=78.1,
            ),
            make_event(
                4,
                'Away',
                'Dispossessed',
                second=5.331,
                x=42.6,
                y=65.2,
            ),
            make_event(
                5,
                'Home',
                'Tackle',
                'Unsuccessful',
                second=5.332,
                x=57.4,
                y=34.8,
            ),
            make_event(
                6,
                'Away',
                'Pass',
                second=6.987,
                x=46.2,
                y=50.1,
                end_x=75.5,
                end_y=4.8,
            ),
            make_event(
                7,
                'Away',
                'Pass',
                second=11.299,
                x=81.7,
                y=9.4,
                end_x=89.3,
                end_y=66.8,
            ),
            make_event(
                8,
                'Away',
                'Goal',
                second=15.307,
                x=89.3,
                y=66.8,
                end_x=100.0,
                end_y=54.9,
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
            result['termination_reason'].iloc[-1],
            'terminal_action_grace',
        )
        self.assertIn(8, result['eventId'].tolist())

    def test_unsuccessful_ball_touch_can_trigger_goal_transition(self):
        result = find_transition([
            make_event(
                2958077549,
                'Home',
                'Ball touch',
                'Unsuccessful',
                minute=27,
                second=54.0,
                x=52.9,
                y=54.4,
            ),
            make_event(
                2958077575,
                'Away',
                'Pass',
                'Successful',
                minute=27,
                second=55.0,
                x=36.3,
                y=34.9,
                end_x=55.3,
                end_y=43.3,
            ),
            make_event(
                2958077589,
                'Away',
                'Ball recovery',
                'Successful',
                minute=27,
                second=56.0,
                x=55.3,
                y=43.5,
            ),
            make_event(
                2958077623,
                'Away',
                'Pass',
                'Successful',
                minute=27,
                second=57.0,
                x=55.3,
                y=43.5,
                end_x=58.1,
                end_y=64.8,
            ),
            make_event(
                2958077665,
                'Away',
                'Pass',
                'Successful',
                minute=27,
                second=58.0,
                x=58.1,
                y=64.8,
                end_x=73.5,
                end_y=62.4,
            ),
            make_event(
                2958077669,
                'Away',
                'Goal',
                'Successful',
                minute=28,
                second=2.0,
                x=86.1,
                y=61.1,
                end_x=100.0,
                end_y=50.0,
            ),
        ])

        self.assertFalse(result.empty)
        self.assertEqual(result['sequence_outcome_type'].iloc[-1], 'Goals')
        self.assertEqual(result['terminal_outcome'].iloc[-1], 'goal')
        self.assertIn(2958077669, result['eventId'].tolist())
        self.assertEqual(
            result.iloc[0]['type_of_initial_loss'],
            'Ball touch',
        )



if __name__ == '__main__':
    unittest.main()
