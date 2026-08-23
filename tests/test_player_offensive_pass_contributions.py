import unittest

import pandas as pd

from src.data_processing.pass_processing import get_passes_df
from src.metrics.player_metrics import (
    build_player_passing_event_sets,
    calculate_offensive_pass_contributions,
    calculate_player_stats,
)


def pass_event(
    event_id,
    player='Player A',
    *,
    outcome='Successful',
    x=60.0,
    y=50.0,
    end_x=65.0,
    end_y=50.0,
    is_key_pass=False,
    is_assist=False,
    cross=0,
    throw_in=0,
    corner=0,
    free_kick=0,
    goal_kick=0,
):
    return {
        'id': event_id,
        'eventId': event_id,
        'playerName': player,
        'team_name': 'Home',
        'type_name': 'Pass',
        'outcome': outcome,
        'x': x,
        'y': y,
        'end_x': end_x,
        'end_y': end_y,
        'is_key_pass': is_key_pass,
        'is_assist': is_assist,
        'cross': cross,
        'ThrowIn': throw_in,
        'Corner taken': corner,
        'Free kick taken': free_kick,
        'Goal kick': goal_kick,
        'timeMin': 1,
        'timeSec': float(event_id),
    }


class PlayerPassingEventSetTests(unittest.TestCase):

    def test_key_pass_and_assist_overlap_counts_once_in_unions(self):
        df = pd.DataFrame([
            pass_event(1, is_key_pass=True, is_assist=True),
        ])
        event_sets = build_player_passing_event_sets(df)
        self.assertEqual(len(event_sets['key_pass']), 1)
        self.assertEqual(len(event_sets['assist']), 1)
        self.assertEqual(len(event_sets['shot_assist']), 1)
        self.assertEqual(len(event_sets['unique_offensive_contribution']), 1)

    def test_assist_without_key_pass_is_still_a_unique_contribution(self):
        df = pd.DataFrame([
            pass_event(1, is_assist=True),
        ])
        event_sets = build_player_passing_event_sets(df)
        self.assertFalse(event_sets['key_pass'])
        self.assertEqual(event_sets['assist'], {('id', 1)})
        self.assertEqual(event_sets['shot_assist'], {('id', 1)})
        self.assertEqual(
            event_sets['unique_offensive_contribution'],
            {('id', 1)},
        )

    def test_restart_key_pass_and_assist_remain_valid_contributions(self):
        df = pd.DataFrame([
            pass_event(
                1,
                x=60.0,
                end_x=90.0,
                corner=1,
                is_key_pass=True,
                is_assist=True,
            ),
        ])
        event_sets = build_player_passing_event_sets(df)
        self.assertEqual(event_sets['key_pass'], {('id', 1)})
        self.assertEqual(event_sets['assist'], {('id', 1)})
        self.assertFalse(event_sets['completed_pass_into_box'])
        self.assertFalse(event_sets['progressive_pass'])
        self.assertEqual(
            event_sets['unique_offensive_contribution'],
            {('id', 1)},
        )

    def test_failed_key_pass_and_assist_are_excluded(self):
        df = pd.DataFrame([
            pass_event(1, outcome='Unsuccessful', is_key_pass=True, is_assist=True),
        ])
        event_sets = build_player_passing_event_sets(df)
        self.assertFalse(event_sets['key_pass'])
        self.assertFalse(event_sets['assist'])
        self.assertFalse(event_sets['shot_assist'])
        self.assertFalse(event_sets['unique_offensive_contribution'])

    def test_completed_into_box_requires_success_and_open_play(self):
        df = pd.DataFrame([
            pass_event(1, end_x=90.0, end_y=50.0),
            pass_event(2, outcome='Unsuccessful', end_x=90.0, end_y=50.0),
            pass_event(3, end_x=90.0, end_y=50.0, corner=1),
        ])
        event_sets = build_player_passing_event_sets(df)
        self.assertEqual(event_sets['completed_pass_into_box'], {('id', 1)})

    def test_progressive_pass_uses_completed_open_play_contract(self):
        df = pd.DataFrame([
            pass_event(1, x=60.0, end_x=80.0),
            pass_event(2, outcome='Unsuccessful', x=60.0, end_x=80.0),
            pass_event(3, x=60.0, end_x=80.0, free_kick=1),
        ])
        event_sets = build_player_passing_event_sets(df)
        self.assertEqual(event_sets['progressive_pass'], {('id', 1)})

    def test_all_restart_types_are_excluded_from_open_play_metrics(self):
        df = pd.DataFrame([
            pass_event(1, x=60.0, end_x=90.0, corner=1),
            pass_event(2, x=60.0, end_x=90.0, free_kick=1),
            pass_event(3, x=60.0, end_x=90.0, throw_in=1),
            pass_event(4, x=60.0, end_x=90.0, goal_kick=1),
        ])
        event_sets = build_player_passing_event_sets(df)
        self.assertFalse(event_sets['completed_pass_into_box'])
        self.assertFalse(event_sets['progressive_pass'])
        self.assertFalse(event_sets['unique_offensive_contribution'])

    def test_single_event_in_all_components_counts_once(self):
        df = pd.DataFrame([
            pass_event(
                1,
                x=60.0,
                end_x=90.0,
                is_key_pass=True,
                is_assist=True,
            ),
        ])
        event_sets = build_player_passing_event_sets(df)
        event_key = {('id', 1)}
        self.assertEqual(event_sets['progressive_pass'], event_key)
        self.assertEqual(event_sets['completed_pass_into_box'], event_key)
        self.assertEqual(event_sets['key_pass'], event_key)
        self.assertEqual(event_sets['assist'], event_key)
        self.assertEqual(
            event_sets['unique_offensive_contribution'],
            event_key,
        )

    def test_unique_contribution_deduplicates_component_overlap(self):
        df = pd.DataFrame([
            pass_event(1, x=60.0, end_x=90.0, is_key_pass=True, is_assist=True),
            pass_event(2, x=70.0, end_x=75.0, is_assist=True),
        ])
        event_sets = build_player_passing_event_sets(df)
        self.assertEqual(len(event_sets['unique_offensive_contribution']), 2)

    def test_counts_are_per_player(self):
        df = pd.DataFrame([
            pass_event(1, player='Player A', end_x=90.0),
            pass_event(2, player='Player B', is_key_pass=True),
        ])
        result = calculate_offensive_pass_contributions(df)
        self.assertEqual(result['Player A'], 1)
        self.assertEqual(result['Player B'], 1)

    def test_regular_pass_is_not_counted(self):
        df = pd.DataFrame([pass_event(1)])
        result = calculate_offensive_pass_contributions(df)
        self.assertTrue(result.empty)

    def test_player_stats_keep_components_separate(self):
        df = pd.DataFrame([
            pass_event(1, x=60.0, end_x=90.0, is_key_pass=True, is_assist=True),
            pass_event(2, x=70.0, end_x=75.0, is_assist=True),
        ])
        stats = calculate_player_stats(df)
        player = stats.loc['Player A']
        self.assertEqual(player['Key Passes'], 1)
        self.assertEqual(player['Assists'], 2)
        self.assertEqual(player['Shot Assists'], 2)
        self.assertEqual(player['Progressive Passes'], 1)
        self.assertEqual(player['Passes into Box'], 1)
        self.assertEqual(player['Offensive Pass Contributions'], 2)

    def test_get_passes_df_into_box_flag_matches_contract(self):
        df = pd.DataFrame([
            pass_event(1, end_x=90.0, end_y=50.0, is_key_pass=True),
            pass_event(2, outcome='Unsuccessful', end_x=90.0, end_y=50.0, is_key_pass=True),
            pass_event(3, end_x=90.0, end_y=50.0, corner=1),
        ])
        passes = get_passes_df(df).set_index('id')
        self.assertTrue(bool(passes.loc[1, 'is_into_box']))
        self.assertFalse(bool(passes.loc[2, 'is_into_box']))
        self.assertFalse(bool(passes.loc[3, 'is_into_box']))
        self.assertTrue(bool(passes.loc[1, 'is_key_pass']))
        self.assertFalse(bool(passes.loc[2, 'is_key_pass']))

    def test_get_passes_df_excludes_all_restarts_from_into_box(self):
        df = pd.DataFrame([
            pass_event(1, end_x=90.0, end_y=50.0, corner=1),
            pass_event(2, end_x=90.0, end_y=50.0, free_kick=1),
            pass_event(3, end_x=90.0, end_y=50.0, throw_in=1),
            pass_event(4, end_x=90.0, end_y=50.0, goal_kick=1),
        ])
        passes = get_passes_df(df).set_index('id')
        self.assertFalse(bool(passes.loc[1, 'is_into_box']))
        self.assertFalse(bool(passes.loc[2, 'is_into_box']))
        self.assertFalse(bool(passes.loc[3, 'is_into_box']))
        self.assertFalse(bool(passes.loc[4, 'is_into_box']))


if __name__ == '__main__':
    unittest.main()
