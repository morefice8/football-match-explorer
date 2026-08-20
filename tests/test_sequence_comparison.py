import unittest

import pandas as pd

from src.metrics.sequence_outcome_metrics import (
    build_sequence_comparison,
)


class SequenceComparisonTests(
    unittest.TestCase
):

    def test_comparison_preserves_team_names_and_totals(
        self
    ):
        home = pd.DataFrame({
            'reached_final_third': [
                True,
                False,
            ],
        })

        away = pd.DataFrame({
            'reached_final_third': [
                True,
                True,
                False,
            ],
        })

        result = build_sequence_comparison(
            home,
            away,
            home_team='Napoli',
            away_team='Udinese',
            milestone_keys=[
                'reached_final_third',
            ],
        )

        self.assertEqual(
            result['home_team'],
            'Napoli',
        )

        self.assertEqual(
            result['away_team'],
            'Udinese',
        )

        self.assertEqual(
            result['home_total'],
            2,
        )

        self.assertEqual(
            result['away_total'],
            3,
        )

    def test_funnel_contains_counts_and_percentages(
        self
    ):
        home = pd.DataFrame({
            'reached_final_third': [
                True,
                False,
                False,
                False,
            ],
        })

        away = pd.DataFrame({
            'reached_final_third': [
                True,
                True,
                False,
                False,
            ],
        })

        result = build_sequence_comparison(
            home,
            away,
            home_team='Home',
            away_team='Away',
            milestone_keys=[
                'reached_final_third',
            ],
        )

        sequence_row = result[
            'funnel'
        ][0]

        final_third_row = result[
            'funnel'
        ][1]

        self.assertEqual(
            sequence_row['home_percentage'],
            100.0,
        )

        self.assertEqual(
            final_third_row['home_count'],
            1,
        )

        self.assertEqual(
            final_third_row[
                'home_percentage'
            ],
            25.0,
        )

        self.assertEqual(
            final_third_row['away_count'],
            2,
        )

        self.assertEqual(
            final_third_row[
                'away_percentage'
            ],
            50.0,
        )

    def test_missing_outcome_on_one_side_is_zero_filled(
        self
    ):
        home = pd.DataFrame({
            'final_outcome': [
                'Shots',
                'Lost Possessions',
            ],
        })

        away = pd.DataFrame({
            'final_outcome': [
                'Lost Possessions',
                'Lost Possessions',
            ],
        })

        result = build_sequence_comparison(
            home,
            away,
            home_team='Home',
            away_team='Away',
        )

        shot_row = next(
            row
            for row in result['outcomes']
            if row['outcome'] == 'Shots'
        )

        self.assertEqual(
            shot_row['home_count'],
            1,
        )

        self.assertEqual(
            shot_row['away_count'],
            0,
        )

        self.assertEqual(
            shot_row['away_percentage'],
            0.0,
        )

    def test_empty_side_is_supported(
        self
    ):
        home = pd.DataFrame({
            'reached_final_third': [
                True,
            ],
            'duration_seconds': [
                8,
            ],
            'pass_count': [
                2,
            ],
        })

        away = pd.DataFrame()

        result = build_sequence_comparison(
            home,
            away,
            home_team='Home',
            away_team='Away',
            milestone_keys=[
                'reached_final_third',
            ],
        )

        self.assertEqual(
            result['home_total'],
            1,
        )

        self.assertEqual(
            result['away_total'],
            0,
        )

        self.assertEqual(
            result[
                'funnel'
            ][0][
                'away_percentage'
            ],
            0.0,
        )

        self.assertEqual(
            result[
                'profile'
            ][
                'home'
            ][
                'avg_duration_seconds'
            ],
            8.0,
        )


if __name__ == '__main__':
    unittest.main()