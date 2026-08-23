import unittest

import pandas as pd

from src.metrics.player_metrics import (
    calculate_player_stats,
)
from src.metrics.shot_sequence_metrics import (
    build_shot_sequences,
    calculate_shot_sequence_player_stats,
)


def event(
    event_id,
    *,
    team='Home',
    player='Player A',
    event_type='Pass',
    outcome='Successful',
    period=1,
    minute=1,
    second=None,
    key_pass=False,
    assist=False,
    penalty=0,
    direct_free_kick=0,
    corner=0,
    free_kick=0,
    throw_in=0,
    goal_kick=0,
    own_goal=0,
):
    return {
        'id': event_id,
        'eventId': event_id,
        'team_name': team,
        'playerName': player,
        'type_name': event_type,
        'outcome': outcome,
        'periodId': period,
        'timeMin': minute,
        'timeSec': (
            float(event_id)
            if second is None
            else float(second)
        ),
        'x': 60.0,
        'y': 50.0,
        'end_x': 70.0,
        'end_y': 50.0,
        'is_key_pass': key_pass,
        'is_assist': assist,
        'Penalty': penalty,
        'Free kick': direct_free_kick,
        'Own goal': own_goal,
        'Corner taken': corner,
        'Free kick taken': free_kick,
        'ThrowIn': throw_in,
        'Goal kick': goal_kick,
        'cross': 0,
    }


class ShotSequenceContractTests(unittest.TestCase):

    def test_basic_chain_assigns_three_roles(self):
        df = pd.DataFrame([
            event(1, player='Pre'),
            event(2, player='Creator', key_pass=True),
            event(3, player='Shooter', event_type='Miss'),
        ])

        result = build_shot_sequences(df)
        roles = result.set_index('playerName')['sequence_role'].to_dict()

        self.assertEqual(roles['Pre'], 'pre_assist')
        self.assertEqual(roles['Creator'], 'shot_assist')
        self.assertEqual(roles['Shooter'], 'shooter')

    def test_key_pass_assist_overlap_counts_once(self):
        df = pd.DataFrame([
            event(1, player='Creator', key_pass=True, assist=True),
            event(2, player='Shooter', event_type='Goal'),
        ])

        stats = calculate_shot_sequence_player_stats(df)

        self.assertEqual(
            stats.loc['Creator', 'Shot Sequence Assists'],
            1,
        )
        self.assertEqual(
            stats.loc['Creator', 'Shot Sequence Involvements'],
            1,
        )

    def test_opponent_control_breaks_possession(self):
        df = pd.DataFrame([
            event(1, player='Old Creator', key_pass=True),
            event(2, team='Away', player='Opponent', event_type='Pass'),
            event(3, player='Recovery', event_type='Ball recovery'),
            event(4, player='Creator', key_pass=True),
            event(5, player='Shooter', event_type='Goal'),
        ])

        result = build_shot_sequences(df)

        self.assertNotIn(1, result['id'].tolist())
        self.assertNotIn(2, result['id'].tolist())
        self.assertEqual(set(result['team_name']), {'Home'})

    def test_period_boundary_breaks_chain(self):
        df = pd.DataFrame([
            event(1, player='Old Passer', period=1),
            event(2, player='Creator', period=2, key_pass=True),
            event(3, player='Shooter', period=2, event_type='Miss'),
        ])

        result = build_shot_sequences(df)

        self.assertEqual(set(result['periodId']), {2})
        self.assertNotIn(1, result['id'].tolist())

    def test_unsuccessful_opponent_event_is_skipped(self):
        df = pd.DataFrame([
            event(1, player='Pre'),
            event(
                2,
                team='Away',
                player='Opponent',
                event_type='Tackle',
                outcome='Unsuccessful',
            ),
            event(3, player='Creator', key_pass=True),
            event(4, player='Shooter', event_type='Miss'),
        ])

        result = build_shot_sequences(df)

        self.assertIn(1, result['id'].tolist())
        self.assertNotIn(2, result['id'].tolist())
        self.assertTrue(
            pd.isna(
                result.loc[
                    result['id'] == 1,
                    'sequence_role',
                ].iloc[0]
            )
        )

        self.assertEqual(
            result.loc[
                result['id'] == 3,
                'sequence_role',
            ].iloc[0],
            'shot_assist',
        )

    def test_recovery_marks_start_of_possession(self):
        df = pd.DataFrame([
            event(1, player='Stale Pass'),
            event(2, player='Recovery', event_type='Ball recovery'),
            event(3, player='Creator', key_pass=True),
            event(4, player='Shooter', event_type='Goal'),
        ])

        result = build_shot_sequences(df)

        self.assertNotIn(1, result['id'].tolist())
        self.assertIn(2, result['id'].tolist())
        self.assertTrue(
            result['sequence_boundary_reason'].eq('possession_start').all()
        )

    def test_previous_shot_prevents_role_reuse(self):
        df = pd.DataFrame([
            event(1, player='Creator', key_pass=True),
            event(2, player='Shooter A', event_type='Miss'),
            event(3, player='Shooter B', event_type='Miss'),
        ])

        result = build_shot_sequences(df)
        second = result[result['shot_sequence_id'] == 2]

        self.assertEqual(second['id'].tolist(), [3])
        self.assertEqual(second['sequence_role'].iloc[0], 'shooter')

    def test_failed_flagged_pass_is_not_assist(self):
        df = pd.DataFrame([
            event(
                1,
                player='Failed Creator',
                outcome='Unsuccessful',
                key_pass=True,
                assist=True,
            ),
            event(2, player='Shooter', event_type='Goal'),
        ])

        result = build_shot_sequences(df)

        self.assertEqual(result['id'].tolist(), [2])
        self.assertFalse(result['sequence_role'].eq('shot_assist').any())

    def test_penalty_is_standalone_sequence(self):
        df = pd.DataFrame([
            event(1, player='Stale Creator', key_pass=True),
            event(2, player='Penalty Taker', event_type='Goal', penalty=1),
        ])

        result = build_shot_sequences(df)

        self.assertEqual(result['id'].tolist(), [2])
        self.assertEqual(
            result['sequence_boundary_reason'].iloc[0],
            'direct_restart_shot',
        )

    def test_every_sequence_has_one_team_period_and_terminal_shot(self):
        df = pd.DataFrame([
            event(1, player='Creator A', key_pass=True),
            event(2, player='Shooter A', event_type='Miss'),
            event(3, team='Away', player='Creator B', key_pass=True, period=2),
            event(4, team='Away', player='Shooter B', event_type='Goal', period=2),
        ])

        result = build_shot_sequences(df)

        for _, sequence in result.groupby('shot_sequence_id'):
            self.assertEqual(sequence['team_name'].nunique(), 1)
            self.assertEqual(sequence['periodId'].nunique(), 1)
            self.assertIn(
                sequence.iloc[-1]['type_name'],
                {'Goal', 'Miss', 'Attempt Saved', 'Post'},
            )
            self.assertEqual(
                (sequence['sequence_role'] == 'shooter').sum(),
                1,
            )

    def test_player_stats_keep_passing_and_sequence_assists_separate(self):
        df = pd.DataFrame([
            # REL-07 passing contribution, but outside the eventual shot
            # possession after the opponent takes controlled possession.
            event(1, player='Old Creator', key_pass=True),
            event(2, team='Away', player='Opponent', event_type='Pass'),
            event(3, player='Recovery', event_type='Ball recovery'),
            event(4, player='Pre'),
            event(5, player='Creator', key_pass=True),
            event(6, player='Shooter', event_type='Goal'),
        ])

        stats = calculate_player_stats(df)

        self.assertEqual(stats.loc['Old Creator', 'Shot Assists'], 1)
        self.assertEqual(
            stats.loc['Old Creator', 'Shot Sequence Assists'],
            0,
        )
        self.assertEqual(
            stats.loc['Creator', 'Shot Sequence Assists'],
            1,
        )
        self.assertEqual(
            stats.loc['Pre', 'Shot Sequence Pre-Assists'],
            1,
        )
        self.assertEqual(
            stats.loc['Shooter', 'Shot Sequence Shots'],
            1,
        )

        # Compatibility aliases remain until the graph redesign.
        self.assertEqual(stats.loc['Pre', 'Buildup to Shot'], 1)
        self.assertEqual(
            stats.loc['Pre', 'Shooting Seq Total'],
            stats.loc['Pre', 'Shot Sequence Involvements'],
        )

    def test_pre_assist_requires_receiver_to_be_shot_creator(self):
        df = pd.DataFrame([
            event(
                1,
                player='Candidate Pre',
            ),
            # This player, not the eventual creator, receives event 1.
            event(
                2,
                player='Other Receiver',
                event_type='Ball touch',
            ),
            event(
                3,
                player='Creator',
                key_pass=True,
            ),
            event(
                4,
                player='Shooter',
                event_type='Miss',
            ),
        ])

        result = build_shot_sequences(df)

        self.assertFalse(
            result['sequence_role']
            .eq('pre_assist')
            .any()
        )
        self.assertEqual(
            result.loc[
                result['id'] == 3,
                'sequence_role',
            ].iloc[0],
            'shot_assist',
        )

    def test_player_can_be_pre_assist_and_shooter_on_different_events(self):
        df = pd.DataFrame([
            event(
                1,
                player='Shooter',
            ),
            event(
                2,
                player='Creator',
                key_pass=True,
            ),
            event(
                3,
                player='Shooter',
                event_type='Goal',
            ),
        ])

        stats = calculate_shot_sequence_player_stats(df)

        self.assertEqual(
            stats.loc[
                'Shooter',
                'Shot Sequence Shots',
            ],
            1,
        )
        self.assertEqual(
            stats.loc[
                'Shooter',
                'Shot Sequence Pre-Assists',
            ],
            1,
        )
        self.assertEqual(
            stats.loc[
                'Shooter',
                'Shot Sequence Involvements',
            ],
            2,
        )

    def test_opponent_error_does_not_break_shot_creating_pass(self):
        df = pd.DataFrame([
            event(
                1,
                player='Vergara',
                key_pass=True,
                assist=True,
            ),
            event(
                2,
                team='Away',
                player='Defender',
                event_type='Error',
                outcome='Successful',
            ),
            event(
                3,
                player='De Bruyne',
                event_type='Goal',
            ),
        ])

        result = build_shot_sequences(df)

        self.assertNotIn(
            2,
            result['id'].tolist(),
        )
        self.assertEqual(
            result.loc[
                result['id'] == 1,
                'sequence_role',
            ].iloc[0],
            'shot_assist',
        )
        self.assertEqual(
            result.loc[
                result['id'] == 3,
                'sequence_role',
            ].iloc[0],
            'shooter',
        )


if __name__ == '__main__':
    unittest.main()
