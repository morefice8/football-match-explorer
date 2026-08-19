import unittest

import pandas as pd

from src.metrics.player_metrics import (
    calculate_offensive_pass_contributions,
)


class OffensivePassContributionTests(unittest.TestCase):

    def test_single_category_counts_once(self):
        df = pd.DataFrame([
            {
                'id': 1,
                'playerName': 'Player A',
                'type_name': 'Pass',
                'end_x': 70,
                'end_y': 50,
                'is_key_pass': False,
                'is_assist': False,
            },
        ])

        progressive = df.copy()

        result = calculate_offensive_pass_contributions(
            df,
            progressive,
        )

        self.assertEqual(result['Player A'], 1)

    def test_overlapping_categories_count_once(self):
        df = pd.DataFrame([
            {
                'id': 1,
                'playerName': 'Player A',
                'type_name': 'Pass',
                'end_x': 90,
                'end_y': 50,
                'is_key_pass': True,
                'is_assist': False,
            },
        ])

        # Same event is progressive + into box + key pass.
        progressive = df.copy()

        result = calculate_offensive_pass_contributions(
            df,
            progressive,
        )

        self.assertEqual(result['Player A'], 1)

    def test_distinct_offensive_passes_count_separately(self):
        df = pd.DataFrame([
            {
                'id': 1,
                'playerName': 'Player A',
                'type_name': 'Pass',
                'end_x': 75,
                'end_y': 50,
                'is_key_pass': False,
                'is_assist': False,
            },
            {
                'id': 2,
                'playerName': 'Player A',
                'type_name': 'Pass',
                'end_x': 90,
                'end_y': 50,
                'is_key_pass': False,
                'is_assist': False,
            },
            {
                'id': 3,
                'playerName': 'Player A',
                'type_name': 'Pass',
                'end_x': 70,
                'end_y': 50,
                'is_key_pass': True,
                'is_assist': False,
            },
        ])

        progressive = df[df['id'] == 1].copy()

        result = calculate_offensive_pass_contributions(
            df,
            progressive,
        )

        self.assertEqual(result['Player A'], 3)

    def test_regular_pass_is_not_counted(self):
        df = pd.DataFrame([
            {
                'id': 1,
                'playerName': 'Player A',
                'type_name': 'Pass',
                'end_x': 60,
                'end_y': 50,
                'is_key_pass': False,
                'is_assist': False,
            },
        ])

        result = calculate_offensive_pass_contributions(
            df,
            pd.DataFrame(),
        )

        self.assertTrue(result.empty)

    def test_counts_are_per_player(self):
        df = pd.DataFrame([
            {
                'id': 1,
                'playerName': 'Player A',
                'type_name': 'Pass',
                'end_x': 90,
                'end_y': 50,
                'is_key_pass': False,
                'is_assist': False,
            },
            {
                'id': 2,
                'playerName': 'Player B',
                'type_name': 'Pass',
                'end_x': 70,
                'end_y': 50,
                'is_key_pass': True,
                'is_assist': False,
            },
        ])

        result = calculate_offensive_pass_contributions(
            df,
            pd.DataFrame(),
        )

        self.assertEqual(result['Player A'], 1)
        self.assertEqual(result['Player B'], 1)


if __name__ == '__main__':
    unittest.main()