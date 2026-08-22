import unittest

import pandas as pd

from src.metrics import buildup_metrics
from src.metrics.sequence_outcome_metrics import summarize_sequences


def make_event(
    event_id,
    team,
    event_type,
    *,
    outcome='Successful',
    x=20,
    y=50,
    end_x=30,
    end_y=50,
    minute=1,
    second=0,
    period=1,
    player='Player',
    role='CM',
    lb=0,
    goal_kick=0,
    penalty=0,
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
        'lb': lb,
        'Length': 10,
        'cross': 0,
        'Corner taken': 0,
        'positional_role': role,
        'Own goal': 0,
        'Penalty': penalty,
        'Goal mouth y co-ordinate': 50,
        'Goal kick': goal_kick,
    }


def find_home_buildup(events, **kwargs):
    return buildup_metrics.find_buildup_sequences(
        pd.DataFrame(events),
        attacking_team='Home',
        defending_team='Away',
        metric_to_analyze='buildup_phase',
        **kwargs,
    )


class ReliableBuildupDefinitionTests(unittest.TestCase):

    def test_keeper_recovery_separates_trigger_and_active_start(self):
        result = find_home_buildup([
            make_event(
                1,
                'Home',
                'Keeper pick-up',
                x=8,
                end_x=8,
                minute=1,
                second=0,
                player='GK',
                role='GK',
            ),
            make_event(
                2,
                'Home',
                'Pass',
                x=8,
                end_x=28,
                minute=1,
                second=5,
                player='GK',
                role='GK',
            ),
            make_event(
                3,
                'Home',
                'Pass',
                x=28,
                end_x=55,
                minute=1,
                second=10,
            ),
        ])

        self.assertFalse(result.empty)
        row = result.iloc[-1]

        self.assertEqual(row['type_of_initial_trigger'], 'Keeper pick-up')
        self.assertEqual(row['first_active_action_type'], 'Pass')
        self.assertEqual(row['timeSec_at_trigger'], 0)
        self.assertEqual(row['timeSec_at_active_start'], 5)
        self.assertEqual(row['terminal_outcome'], 'consolidated')
        self.assertEqual(
            row['termination_reason'],
            'opposition_half_reached',
        )

        summary = summarize_sequences(
            result,
            sequence_kind='buildup',
        ).iloc[0]

        self.assertEqual(summary['restart_delay_seconds'], 5.0)
        self.assertEqual(summary['duration_seconds'], 5.0)

    def test_goal_kick_is_trigger_and_first_active_action(self):
        result = find_home_buildup([
            make_event(
                1,
                'Home',
                'Pass',
                x=6,
                end_x=62,
                minute=2,
                second=0,
                player='GK',
                role='GK',
                lb=1,
                goal_kick=1,
            ),
        ])

        self.assertFalse(result.empty)
        row = result.iloc[-1]

        self.assertEqual(row['type_of_initial_trigger'], 'Goal kick')
        self.assertEqual(row['first_active_action_type'], 'Goal kick')
        self.assertEqual(row['buildup_type'], 'Long Ball')
        self.assertEqual(row['terminal_outcome'], 'consolidated')
        self.assertEqual(
            row['termination_reason'],
            'opposition_half_reached',
        )

        summary = summarize_sequences(
            result,
            sequence_kind='buildup',
        ).iloc[0]

        self.assertEqual(summary['restart_delay_seconds'], 0.0)
        self.assertEqual(summary['duration_seconds'], 0.0)

    def test_low_outfield_recovery_does_not_start_buildup(self):
        result = find_home_buildup([
            make_event(
                1,
                'Home',
                'Ball recovery',
                x=18,
                minute=3,
                second=0,
                role='CB',
            ),
            make_event(
                2,
                'Home',
                'Pass',
                x=18,
                end_x=32,
                minute=3,
                second=2,
                role='CB',
            ),
            make_event(
                3,
                'Home',
                'Pass',
                x=32,
                end_x=54,
                minute=3,
                second=6,
            ),
        ])

        self.assertTrue(result.empty)

    def test_short_long_classification_uses_first_phase_only(self):
        result = find_home_buildup([
            make_event(
                1,
                'Home',
                'Keeper pick-up',
                x=8,
                minute=4,
                second=0,
                role='GK',
            ),
            make_event(
                2,
                'Home',
                'Pass',
                x=8,
                end_x=24,
                minute=4,
                second=2,
                lb=0,
            ),
            make_event(
                3,
                'Home',
                'Pass',
                x=24,
                end_x=61,
                minute=4,
                second=7,
                lb=1,
            ),
        ])

        self.assertFalse(result.empty)
        self.assertEqual(
            result.iloc[-1]['buildup_type'],
            'Short-Long',
        )

    def test_unsuccessful_first_pass_is_turnover(self):
        result = find_home_buildup([
            make_event(
                1,
                'Home',
                'Keeper pick-up',
                x=10,
                minute=5,
                second=0,
                role='GK',
            ),
            make_event(
                2,
                'Home',
                'Pass',
                outcome='Unsuccessful',
                x=10,
                end_x=38,
                minute=5,
                second=3,
            ),
        ])

        self.assertFalse(result.empty)
        row = result.iloc[-1]

        self.assertEqual(row['terminal_outcome'], 'turnover')
        self.assertEqual(row['termination_reason'], 'unsuccessful_pass')
        self.assertEqual(row['buildup_pass_count'], 0)
        self.assertEqual(row['buildup_type'], 'Short-Short')

    def test_foul_terminates_active_buildup(self):
        result = find_home_buildup([
            make_event(
                1,
                'Home',
                'Keeper pick-up',
                x=20,
                minute=6,
                second=0,
                player='GK',
                role='GK',
            ),
            make_event(
                2,
                'Home',
                'Pass',
                x=20,
                end_x=35,
                minute=6,
                second=2,
            ),
            make_event(
                3,
                'Home',
                'Foul',
                outcome='Unsuccessful',
                x=35,
                minute=6,
                second=6,
            ),
        ])

        self.assertFalse(result.empty)
        row = result.iloc[-1]

        self.assertEqual(row['terminal_outcome'], 'foul')
        self.assertEqual(row['termination_reason'], 'foul')
        self.assertEqual(row['sequence_outcome_type'], 'Foul')

    def test_shot_terminates_active_buildup(self):
        result = find_home_buildup([
            make_event(
                1,
                'Home',
                'Keeper pick-up',
                x=18,
                minute=7,
                second=0,
                player='GK',
                role='GK',
            ),
            make_event(
                2,
                'Home',
                'Pass',
                x=18,
                end_x=42,
                minute=7,
                second=2,
            ),
            make_event(
                3,
                'Home',
                'Miss',
                outcome='Unsuccessful',
                x=48,
                end_x=100,
                minute=7,
                second=6,
            ),
        ])

        self.assertFalse(result.empty)
        row = result.iloc[-1]

        self.assertEqual(row['terminal_outcome'], 'shot')
        self.assertEqual(row['termination_reason'], 'shot')

        summary = summarize_sequences(
            result,
            sequence_kind='buildup',
        ).iloc[0]

        self.assertTrue(summary['produced_shot'])

    def test_period_change_is_hard_boundary(self):
        result = find_home_buildup([
            make_event(
                1,
                'Home',
                'Keeper pick-up',
                x=10,
                minute=45,
                second=0,
                period=1,
                role='GK',
            ),
            make_event(
                2,
                'Home',
                'Pass',
                x=10,
                end_x=30,
                minute=45,
                second=2,
                period=1,
            ),
            make_event(
                3,
                'Home',
                'Pass',
                x=30,
                end_x=55,
                minute=46,
                second=0,
                period=2,
            ),
        ])

        self.assertFalse(result.empty)
        self.assertEqual(len(result), 1)

        row = result.iloc[-1]

        self.assertEqual(row['terminal_outcome'], 'retained')
        self.assertEqual(row['termination_reason'], 'period_boundary')
        self.assertEqual(row['periodId'], 1)

    def test_long_restart_delay_is_not_active_duration(self):
        result = find_home_buildup([
            make_event(
                1,
                'Home',
                'Keeper pick-up',
                x=8,
                minute=8,
                second=0,
                role='GK',
            ),
            make_event(
                2,
                'Home',
                'Pass',
                x=8,
                end_x=28,
                minute=8,
                second=30,
                role='GK',
            ),
            make_event(
                3,
                'Home',
                'Pass',
                x=28,
                end_x=55,
                minute=8,
                second=35,
            ),
        ])

        summary = summarize_sequences(
            result,
            sequence_kind='buildup',
        ).iloc[0]

        self.assertEqual(summary['restart_delay_seconds'], 30.0)
        self.assertEqual(summary['duration_seconds'], 5.0)

    def test_active_window_separates_first_phase_from_normal_possession(self):
        result = find_home_buildup([
            make_event(
                1,
                'Home',
                'Keeper pick-up',
                x=15,
                minute=9,
                second=0,
                player='GK',
                role='GK',
            ),
            make_event(
                2,
                'Home',
                'Pass',
                x=15,
                end_x=28,
                minute=9,
                second=2,
            ),
            make_event(
                3,
                'Home',
                'Pass',
                x=28,
                end_x=38,
                minute=9,
                second=25,
            ),
        ])

        self.assertFalse(result.empty)
        self.assertEqual(len(result), 1)

        row = result.iloc[-1]

        self.assertEqual(row['terminal_outcome'], 'consolidated')
        self.assertEqual(
            row['termination_reason'],
            'time_window_elapsed',
        )
        self.assertEqual(
            row['buildup_active_duration_seconds'],
            20.0,
        )

        summary = summarize_sequences(
            result,
            sequence_kind='buildup',
        ).iloc[0]

        self.assertEqual(summary['duration_seconds'], 20.0)


if __name__ == '__main__':
    unittest.main()
