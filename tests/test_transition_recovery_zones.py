import unittest

import numpy as np
import pandas as pd

from src.metrics.transition_metrics import (
    find_buildup_after_possession_loss,
)


def event(
    event_id,
    team,
    event_type,
    outcome,
    x,
    y=50,
    end_x=np.nan,
    end_y=np.nan,
    minute=1,
    second=0,
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
        'playerName': 'Player',
        'Mapped Jersey Number': 1,
        'timeMin': minute,
        'timeSec': second,
        'periodId': 1,
    }


class TransitionRecoveryZoneTests(
    unittest.TestCase
):

    def test_save_uses_recovering_team_location(
        self
    ):
        df = pd.DataFrame([
            event(
                1,
                'Away',
                'Save',
                'Successful',
                x=10,
                second=0,
            ),
            event(
                2,
                'Away',
                'Pass',
                'Successful',
                x=12,
                end_x=25,
                second=2,
            ),
            event(
                3,
                'Away',
                'Pass',
                'Unsuccessful',
                x=25,
                end_x=40,
                second=4,
            ),
        ])

        result = (
            find_buildup_after_possession_loss(
                df,
                team_that_lost_possession='Home',
                metric_to_analyze=
                    'offensive_transitions',
            )
        )

        self.assertFalse(result.empty)

        self.assertEqual(
            result.iloc[0]['loss_zone'],
            'Defensive Third',
        )


    def test_goal_does_not_start_transition(
        self
    ):
        df = pd.DataFrame([
            event(
                1,
                'Home',
                'Goal',
                'Successful',
                x=90,
                second=0,
            ),
            event(
                2,
                'Away',
                'Pass',
                'Successful',
                x=50,
                end_x=60,
                second=5,
            ),
        ])

        result = (
            find_buildup_after_possession_loss(
                df,
                team_that_lost_possession='Home',
                metric_to_analyze=
                    'offensive_transitions',
            )
        )

        self.assertTrue(result.empty)


    def test_clearance_without_end_x_uses_first_recovery_location(
        self
    ):
        df = pd.DataFrame([
            event(
                1,
                'Home',
                'Clearance',
                'Successful',
                x=20,
                end_x=np.nan,
                second=0,
            ),
            event(
                2,
                'Away',
                'Pass',
                'Successful',
                x=25,
                end_x=35,
                second=2,
            ),
            event(
                3,
                'Away',
                'Pass',
                'Unsuccessful',
                x=35,
                end_x=45,
                second=4,
            ),
        ])

        result = (
            find_buildup_after_possession_loss(
                df,
                team_that_lost_possession='Home',
                metric_to_analyze=
                    'offensive_transitions',
            )
        )

        self.assertFalse(result.empty)

        self.assertEqual(
            result.iloc[0]['loss_zone'],
            'Defensive Third',
        )

    def test_defensive_failed_pass_stores_loss_destination(
        self
    ):
        df = pd.DataFrame([
            event(
                1,
                'Home',
                'Pass',
                'Unsuccessful',
                x=30,
                y=40,
                end_x=45,
                end_y=60,
                second=0,
            ),
            event(
                2,
                'Away',
                'Pass',
                'Successful',
                x=45,
                y=60,
                end_x=55,
                end_y=60,
                second=2,
            ),
            event(
                3,
                'Away',
                'Pass',
                'Unsuccessful',
                x=55,
                y=60,
                end_x=65,
                end_y=60,
                second=4,
            ),
        ])

        result = (
            find_buildup_after_possession_loss(
                df,
                team_that_lost_possession='Home',
                metric_to_analyze=
                    'defensive_transitions',
            )
        )

        self.assertFalse(result.empty)

        self.assertTrue(
            result['loss_x'].eq(45).all()
        )

        self.assertTrue(
            result['loss_y'].eq(60).all()
        )

        self.assertTrue(
            result['loss_zone']
            .eq('Middle Third')
            .all()
        )


    def test_defensive_point_loss_stores_event_location(
        self
    ):
        df = pd.DataFrame([
            event(
                1,
                'Home',
                'Dispossessed',
                'Successful',
                x=28,
                y=42,
                second=0,
            ),
            event(
                2,
                'Away',
                'Pass',
                'Successful',
                x=28,
                y=42,
                end_x=40,
                end_y=45,
                second=2,
            ),
            event(
                3,
                'Away',
                'Pass',
                'Unsuccessful',
                x=40,
                y=45,
                end_x=50,
                end_y=45,
                second=4,
            ),
        ])

        result = (
            find_buildup_after_possession_loss(
                df,
                team_that_lost_possession='Home',
                metric_to_analyze=
                    'defensive_transitions',
            )
        )

        self.assertFalse(result.empty)

        self.assertTrue(
            result['loss_x'].eq(28).all()
        )

        self.assertTrue(
            result['loss_y'].eq(42).all()
        )

        self.assertTrue(
            result['loss_zone']
            .eq('Defensive Third')
            .all()
        )


if __name__ == '__main__':
    unittest.main()