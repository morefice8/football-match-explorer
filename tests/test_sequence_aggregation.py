import unittest

import pandas as pd

from src.metrics.sequence_outcome_metrics import (
    aggregate_sequence_outcomes,
)


class SequenceAggregationTests(
    unittest.TestCase
):

    def test_empty_summary_returns_zero_counts(
        self
    ):
        result = aggregate_sequence_outcomes(
            pd.DataFrame()
        )

        self.assertEqual(
            result['total_sequences'],
            0,
        )

        self.assertEqual(
            result[
                'milestones'
            ][
                'reached_final_third'
            ]['count'],
            0,
        )

        self.assertEqual(
            result[
                'milestones'
            ][
                'reached_final_third'
            ]['percentage'],
            0.0,
        )

        self.assertEqual(
            result['outcomes'],
            {},
        )

    def test_milestone_counts_and_percentages(
        self
    ):
        df = pd.DataFrame({
            'reached_middle_third': [
                True,
                True,
                True,
                False,
            ],
            'reached_opposition_half': [
                True,
                True,
                False,
                False,
            ],
            'reached_final_third': [
                True,
                False,
                False,
                False,
            ],
            'entered_penalty_area': [
                True,
                False,
                False,
                False,
            ],
            'produced_shot': [
                False,
                False,
                False,
                False,
            ],
            'produced_goal': [
                False,
                False,
                False,
                False,
            ],
        })

        result = aggregate_sequence_outcomes(
            df
        )

        self.assertEqual(
            result['total_sequences'],
            4,
        )

        self.assertEqual(
            result[
                'milestones'
            ][
                'reached_middle_third'
            ]['count'],
            3,
        )

        self.assertEqual(
            result[
                'milestones'
            ][
                'reached_opposition_half'
            ]['count'],
            2,
        )

        self.assertEqual(
            result[
                'milestones'
            ][
                'reached_final_third'
            ]['count'],
            1,
        )

        self.assertEqual(
            result[
                'milestones'
            ][
                'reached_final_third'
            ]['percentage'],
            25.0,
        )

    def test_terminal_outcomes_are_aggregated(
        self
    ):
        df = pd.DataFrame({
            'final_outcome': [
                'Lost Possessions',
                'Lost Possessions',
                'Shots',
                'Possession Consolidated',
            ],
        })

        result = aggregate_sequence_outcomes(
            df
        )

        self.assertEqual(
            result[
                'outcomes'
            ][
                'Lost Possessions'
            ]['count'],
            2,
        )

        self.assertEqual(
            result[
                'outcomes'
            ][
                'Lost Possessions'
            ]['percentage'],
            50.0,
        )

        self.assertEqual(
            result[
                'outcomes'
            ][
                'Shots'
            ]['count'],
            1,
        )

    def test_duration_and_pass_statistics(
        self
    ):
        df = pd.DataFrame({
            'duration_seconds': [
                0,
                10,
                20,
                30,
            ],
            'pass_count': [
                0,
                2,
                4,
                6,
            ],
        })

        result = aggregate_sequence_outcomes(
            df
        )

        self.assertEqual(
            result[
                'avg_duration_seconds'
            ],
            15.0,
        )

        self.assertEqual(
            result[
                'median_duration_seconds'
            ],
            15.0,
        )

        self.assertEqual(
            result[
                'avg_completed_passes'
            ],
            3.0,
        )

        self.assertEqual(
            result[
                'median_completed_passes'
            ],
            3.0,
        )

    def test_milestones_are_independent_of_terminal_outcome(
        self
    ):
        df = pd.DataFrame({
            'final_outcome': [
                'Lost Possessions',
            ],
            'reached_middle_third': [
                True,
            ],
            'reached_opposition_half': [
                True,
            ],
            'reached_final_third': [
                True,
            ],
            'entered_penalty_area': [
                True,
            ],
            'produced_shot': [
                False,
            ],
            'produced_goal': [
                False,
            ],
        })

        result = aggregate_sequence_outcomes(
            df
        )

        self.assertEqual(
            result[
                'outcomes'
            ][
                'Lost Possessions'
            ]['count'],
            1,
        )

        self.assertEqual(
            result[
                'milestones'
            ][
                'entered_penalty_area'
            ]['count'],
            1,
        )

        self.assertEqual(
            result[
                'milestones'
            ][
                'produced_shot'
            ]['count'],
            0,
        )


if __name__ == '__main__':
    unittest.main()