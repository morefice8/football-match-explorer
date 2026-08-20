import unittest

import pandas as pd

from src.metrics import transition_metrics


def make_event(
    event_id,
    team,
    event_type,
    outcome='Successful',
    x=50,
    y=50,
    end_x=50,
    end_y=50,
    minute=1,
    second=0,
    period=1,
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
        'Own goal': 0,
        'From corner': 0,
        'Goal mouth y co-ordinate': 50,
    }


class TransitionPhaseBoundaryTests(
    unittest.TestCase
):

    def test_offensive_transition_consolidates_after_12_seconds(
        self
    ):
        df = pd.DataFrame([
            make_event(
                1,
                'Home',
                'Pass',
                outcome='Unsuccessful',
                second=0,
            ),
            make_event(
                2,
                'Away',
                'Pass',
                outcome='Successful',
                x=50,
                end_x=60,
                second=4,
            ),
            make_event(
                3,
                'Away',
                'Pass',
                outcome='Successful',
                x=60,
                end_x=65,
                second=9,
            ),
            make_event(
                4,
                'Away',
                'Pass',
                outcome='Successful',
                x=65,
                end_x=70,
                second=13,
            ),
        ])

        result = (
            transition_metrics
            .find_buildup_after_possession_loss(
                df,
                team_that_lost_possession='Home',
                metric_to_analyze=(
                    'offensive_transitions'
                ),
            )
        )

        self.assertFalse(result.empty)

        self.assertEqual(
            result[
                'sequence_outcome_type'
            ].iloc[-1],
            'Possession Consolidated',
        )

        self.assertNotIn(
            4,
            result['eventId'].tolist(),
        )

    def test_defensive_transition_consolidates_after_12_seconds(
        self
    ):
        df = pd.DataFrame([
            make_event(
                1,
                'Home',
                'Pass',
                outcome='Unsuccessful',
                second=0,
            ),
            make_event(
                2,
                'Away',
                'Pass',
                outcome='Successful',
                second=5,
            ),
            make_event(
                3,
                'Away',
                'Pass',
                outcome='Successful',
                second=13,
            ),
        ])

        result = (
            transition_metrics
            .find_buildup_after_possession_loss(
                df,
                team_that_lost_possession='Home',
                metric_to_analyze=(
                    'defensive_transitions'
                ),
            )
        )

        self.assertFalse(result.empty)

        self.assertEqual(
            result[
                'sequence_outcome_type'
            ].iloc[-1],
            'Opponent Possession Consolidated',
        )

    def test_shot_inside_window_is_transition_outcome(
        self
    ):
        df = pd.DataFrame([
            make_event(
                1,
                'Home',
                'Pass',
                outcome='Unsuccessful',
                second=0,
            ),
            make_event(
                2,
                'Away',
                'Pass',
                outcome='Successful',
                x=50,
                end_x=75,
                second=4,
            ),
            make_event(
                3,
                'Away',
                'Miss',
                outcome='Unsuccessful',
                x=80,
                end_x=100,
                second=11,
            ),
        ])

        result = (
            transition_metrics
            .find_buildup_after_possession_loss(
                df,
                team_that_lost_possession='Home',
                metric_to_analyze=(
                    'offensive_transitions'
                ),
            )
        )

        self.assertFalse(result.empty)

        self.assertEqual(
            result[
                'sequence_outcome_type'
            ].iloc[-1],
            'Shots',
        )

        self.assertIn(
            3,
            result['eventId'].tolist(),
        )

    def test_shot_after_window_is_not_transition_outcome(
        self
    ):
        df = pd.DataFrame([
            make_event(
                1,
                'Home',
                'Pass',
                outcome='Unsuccessful',
                second=0,
            ),
            make_event(
                2,
                'Away',
                'Pass',
                outcome='Successful',
                x=50,
                end_x=65,
                second=5,
            ),
            make_event(
                3,
                'Away',
                'Pass',
                outcome='Successful',
                x=65,
                end_x=75,
                second=10,
            ),
            make_event(
                4,
                'Away',
                'Goal',
                x=85,
                end_x=100,
                second=15,
            ),
        ])

        result = (
            transition_metrics
            .find_buildup_after_possession_loss(
                df,
                team_that_lost_possession='Home',
                metric_to_analyze=(
                    'offensive_transitions'
                ),
            )
        )

        self.assertFalse(result.empty)

        self.assertEqual(
            result[
                'sequence_outcome_type'
            ].iloc[-1],
            'Possession Consolidated',
        )

        self.assertNotIn(
            4,
            result['eventId'].tolist(),
        )

    def test_transition_window_is_configurable(
        self
    ):
        df = pd.DataFrame([
            make_event(
                1,
                'Home',
                'Pass',
                outcome='Unsuccessful',
                second=0,
            ),
            make_event(
                2,
                'Away',
                'Pass',
                outcome='Successful',
                second=5,
            ),
            make_event(
                3,
                'Away',
                'Goal',
                x=85,
                end_x=100,
                second=15,
            ),
        ])

        result = (
            transition_metrics
            .find_buildup_after_possession_loss(
                df,
                team_that_lost_possession='Home',
                metric_to_analyze=(
                    'offensive_transitions'
                ),
                max_transition_seconds=20,
            )
        )

        self.assertFalse(result.empty)

        self.assertEqual(
            result[
                'sequence_outcome_type'
            ].iloc[-1],
            'Goals',
        )


if __name__ == '__main__':
    unittest.main()