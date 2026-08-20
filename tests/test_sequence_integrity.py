import unittest

import pandas as pd

from src.metrics import (
    buildup_metrics,
    transition_metrics,
)


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
    player='Player',
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
        'playerName': player,
        'Mapped Jersey Number': 1,
        'timeMin': minute,
        'timeSec': second,
        'periodId': period,

        # Required by buildup_metrics
        'lb': 0,
        'Length': 10,
        'cross': 0,
        'Corner taken': 0,

        # Useful optional qualifiers
        'positional_role': 'CM',
        'Own goal': 0,
        'From corner': 0,
        'Goal mouth y co-ordinate': 50,
    }


class TransitionSequenceIntegrityTests(
    unittest.TestCase
):

    def test_transition_does_not_cross_period_boundary(
        self
    ):
        df = pd.DataFrame([
            make_event(
                1,
                'Home',
                'Pass',
                outcome='Unsuccessful',
                x=60,
                end_x=55,
                period=1,
            ),
            make_event(
                2,
                'Away',
                'Pass',
                outcome='Successful',
                x=45,
                end_x=60,
                period=2,
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

        self.assertTrue(result.empty)

    def test_goal_from_other_team_is_not_loss_trigger(
        self
    ):
        df = pd.DataFrame([
            make_event(
                1,
                'Away',
                'Goal',
                x=90,
                end_x=100,
            ),
            make_event(
                2,
                'Home',
                'Pass',
                outcome='Successful',
            ),
        ])

        result = (
            transition_metrics
            .find_buildup_after_possession_loss(
                df,
                team_that_lost_possession='Home',
                possession_loss_types=['Goal'],
            )
        )

        self.assertTrue(result.empty)

    def test_failed_pass_into_box_is_not_big_chance(
        self
    ):
        df = pd.DataFrame([
            make_event(
                1,
                'Home',
                'Pass',
                outcome='Unsuccessful',
                x=55,
                end_x=50,
            ),
            make_event(
                2,
                'Away',
                'Pass',
                outcome='Unsuccessful',
                x=70,
                end_x=90,
                end_y=50,
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
            'Regained Possessions',
        )

    def test_defensive_transition_records_regain(
        self
    ):
        df = pd.DataFrame([
            make_event(
                1,
                'Home',
                'Pass',
                outcome='Unsuccessful',
                x=55,
                end_x=50,
            ),
            make_event(
                2,
                'Away',
                'Pass',
                outcome='Successful',
                x=50,
                end_x=65,
            ),
            make_event(
                3,
                'Home',
                'Tackle',
                outcome='Successful',
                x=65,
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
            'Regained Possessions',
        )

    def test_offensive_transition_records_loss(
        self
    ):
        df = pd.DataFrame([
            make_event(
                1,
                'Home',
                'Pass',
                outcome='Unsuccessful',
                x=55,
                end_x=50,
            ),
            make_event(
                2,
                'Away',
                'Pass',
                outcome='Successful',
                x=50,
                end_x=65,
            ),
            make_event(
                3,
                'Home',
                'Tackle',
                outcome='Successful',
                x=65,
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
            'Lost Possessions',
        )


class BuildupSequenceIntegrityTests(
    unittest.TestCase
):

    def test_buildup_does_not_cross_period_boundary(
        self
    ):
        df = pd.DataFrame([
            make_event(
                1,
                'Home',
                'Keeper pick-up',
                x=10,
                end_x=10,
                period=1,
                player='GK',
            ),
            make_event(
                2,
                'Home',
                'Pass',
                x=10,
                end_x=30,
                period=2,
                player='GK',
            ),
            make_event(
                3,
                'Away',
                'Unknown',
                period=2,
            ),
        ])

        result = buildup_metrics.find_buildup_sequences(
            df,
            attacking_team='Home',
            defending_team='Away',
            metric_to_analyze='buildup_phase',
            triggers_buildups=['Keeper pick-up'],
        )

        self.assertTrue(result.empty)

    def test_failed_buildup_pass_into_box_is_loss(
        self
    ):
        df = pd.DataFrame([
            make_event(
                1,
                'Home',
                'Keeper pick-up',
                x=10,
                end_x=10,
                player='GK',
            ),
            make_event(
                2,
                'Home',
                'Pass',
                outcome='Unsuccessful',
                x=70,
                end_x=90,
                end_y=50,
            ),
        ])

        result = buildup_metrics.find_buildup_sequences(
            df,
            attacking_team='Home',
            defending_team='Away',
            metric_to_analyze='buildup_phase',
            triggers_buildups=['Keeper pick-up'],
        )

        self.assertFalse(result.empty)

        self.assertEqual(
            result[
                'sequence_outcome_type'
            ].iloc[-1],
            'Lost Possessions',
        )

    def test_buildup_records_opponent_regain(
        self
    ):
        df = pd.DataFrame([
            make_event(
                1,
                'Home',
                'Keeper pick-up',
                x=10,
                end_x=10,
                player='GK',
            ),
            make_event(
                2,
                'Home',
                'Pass',
                outcome='Successful',
                x=10,
                end_x=30,
            ),
            make_event(
                3,
                'Away',
                'Tackle',
                outcome='Successful',
                x=30,
            ),
        ])

        result = buildup_metrics.find_buildup_sequences(
            df,
            attacking_team='Home',
            defending_team='Away',
            metric_to_analyze='buildup_phase',
            triggers_buildups=['Keeper pick-up'],
        )

        self.assertFalse(result.empty)

        self.assertEqual(
            result[
                'sequence_outcome_type'
            ].iloc[-1],
            'Lost Possessions',
        )


if __name__ == '__main__':
    unittest.main()