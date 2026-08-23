import unittest

from src import config
from src.metrics.data_quality import threshold_status


class DataCoverageThresholdTests(unittest.TestCase):

    def test_buildup_retention_warns_below_configured_threshold(self):
        self.assertEqual(
            threshold_status(
                "buildup_sequence_retention_pct",
                74.9,
                config.DATA_COVERAGE_THRESHOLDS,
            ),
            "warning",
        )
        self.assertEqual(
            threshold_status(
                "buildup_sequence_retention_pct",
                75.0,
                config.DATA_COVERAGE_THRESHOLDS,
            ),
            "ok",
        )

    def test_transition_retention_is_neutral_by_default(self):
        for value in (45.3, 1.0):
            with self.subTest(value=value):
                self.assertEqual(
                    threshold_status(
                        "transition_sequence_retention_pct",
                        value,
                        config.DATA_COVERAGE_THRESHOLDS,
                    ),
                    "neutral",
                )

    def test_other_rel10_thresholds_remain_configured(self):
        expectations = (
            ("receiver_coverage_pct", 69.9, "warning"),
            ("valid_coordinate_pct", 94.9, "warning"),
            ("known_outcome_pct", 97.9, "warning"),
            ("carry_inclusion_pct", 10.0, "neutral"),
        )

        for key, value, expected in expectations:
            with self.subTest(key=key):
                self.assertEqual(
                    threshold_status(
                        key,
                        value,
                        config.DATA_COVERAGE_THRESHOLDS,
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
