import unittest

import numpy as np
import pandas as pd

from src.metrics.data_quality import (
    carry_candidate_coverage,
    carry_candidate_coverage_from_stats,
    coordinate_coverage,
    coverage_item,
    outcome_coverage,
    receiver_coverage,
    sequence_outcome_coverage,
    sequence_retention,
    threshold_status,
)


class DataQualityTests(unittest.TestCase):

    def test_receiver_coverage_uses_successful_passes_only(self):
        df = pd.DataFrame([
            {
                "outcome": "Successful",
                "receiver_is_reliable": True,
                "receiver_confidence": "high",
            },
            {
                "outcome": "Successful",
                "receiver_is_reliable": False,
                "receiver_confidence": None,
            },
            {
                "outcome": "Unsuccessful",
                "receiver_is_reliable": False,
                "receiver_confidence": None,
            },
        ])

        coverage = receiver_coverage(df)

        self.assertEqual(coverage["eligible"], 2)
        self.assertEqual(coverage["resolved"], 1)
        self.assertEqual(coverage["unresolved"], 1)
        self.assertEqual(coverage["high"], 1)
        self.assertAlmostEqual(coverage["coverage_pct"], 50.0)

    def test_coordinate_coverage_requires_all_requested_coordinates(self):
        df = pd.DataFrame([
            {"x": 10, "y": 20, "end_x": 30, "end_y": 40},
            {"x": 10, "y": 20, "end_x": np.nan, "end_y": 40},
            {"x": "bad", "y": 20, "end_x": 30, "end_y": 40},
        ])

        coverage = coordinate_coverage(
            df,
            ("x", "y", "end_x", "end_y"),
        )

        self.assertEqual(coverage["total"], 3)
        self.assertEqual(coverage["valid"], 1)
        self.assertEqual(coverage["invalid"], 2)
        self.assertAlmostEqual(coverage["coverage_pct"], 100 / 3)

    def test_coordinate_coverage_missing_column_is_not_silently_ignored(self):
        df = pd.DataFrame([
            {"x": 10, "y": 20},
            {"x": 30, "y": 40},
        ])

        coverage = coordinate_coverage(
            df,
            ("x", "y", "end_x", "end_y"),
        )

        self.assertEqual(coverage["valid"], 0)
        self.assertEqual(coverage["invalid"], 2)
        self.assertIn("end_x", coverage["missing_columns"])

    def test_outcome_coverage_treats_blank_and_unknown_as_unknown(self):
        df = pd.DataFrame({
            "outcome": [
                "Successful",
                "Unsuccessful",
                "Unknown",
                "",
                None,
            ]
        })

        coverage = outcome_coverage(df)

        self.assertEqual(coverage["known"], 2)
        self.assertEqual(coverage["unknown"], 3)
        self.assertAlmostEqual(coverage["known_pct"], 40.0)

    def test_carry_candidates_report_included_and_excluded(self):
        coverage = carry_candidate_coverage(
            candidates=10,
            included=3,
            excluded=7,
        )

        self.assertEqual(coverage["included"], 3)
        self.assertEqual(coverage["excluded"], 7)
        self.assertAlmostEqual(coverage["inclusion_pct"], 30.0)

    def test_carry_coverage_reads_existing_final_third_stats(self):
        coverage = carry_candidate_coverage_from_stats({
            "carry_entry_candidates": 8,
            "carry_entries": 2,
            "carry_entries_excluded_total": 6,
        })

        self.assertEqual(coverage["candidates"], 8)
        self.assertEqual(coverage["included"], 2)
        self.assertEqual(coverage["excluded"], 6)

    def test_sequence_retention_distinguishes_discarded_candidates(self):
        coverage = sequence_retention(
            candidates=20,
            built=15,
        )

        self.assertEqual(coverage["discarded"], 5)
        self.assertAlmostEqual(coverage["retention_pct"], 75.0)

    def test_sequence_unknown_outcomes_are_counted_once_per_sequence(self):
        df = pd.DataFrame([
            {
                "loss_sequence_id": 1,
                "sequence_outcome_type": "Shots",
            },
            {
                "loss_sequence_id": 1,
                "sequence_outcome_type": "Shots",
            },
            {
                "loss_sequence_id": 2,
                "sequence_outcome_type": "Unknown",
            },
        ])

        coverage = sequence_outcome_coverage(
            df,
            sequence_id_column="loss_sequence_id",
        )

        self.assertEqual(coverage["total"], 2)
        self.assertEqual(coverage["unknown"], 1)
        self.assertAlmostEqual(coverage["known_pct"], 50.0)

    def test_warning_is_only_emitted_when_threshold_is_configured(self):
        self.assertEqual(
            threshold_status(
                "receiver_coverage_pct",
                20.0,
                {},
            ),
            "neutral",
        )

        self.assertEqual(
            threshold_status(
                "receiver_coverage_pct",
                69.9,
                {
                    "receiver_coverage_pct": {
                        "min": 70.0,
                    }
                },
            ),
            "warning",
        )

        self.assertEqual(
            threshold_status(
                "receiver_coverage_pct",
                70.0,
                {
                    "receiver_coverage_pct": {
                        "min": 70.0,
                    }
                },
            ),
            "ok",
        )

    def test_neutral_carry_metric_stays_neutral(self):
        item = coverage_item(
            key="carry_inclusion_pct",
            label="Carry candidates",
            value="2 / 10",
            detail="8 excluded",
            threshold_value=20.0,
            thresholds={
                "carry_inclusion_pct": None,
            },
        )

        self.assertEqual(item["status"], "neutral")


if __name__ == "__main__":
    unittest.main()
