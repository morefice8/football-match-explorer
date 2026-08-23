import unittest

import pandas as pd

from src.metrics.data_quality import attach_sequence_coverage


class DataCoverageMetadataTests(unittest.TestCase):

    def test_attaches_candidate_built_discarded_metadata(self):
        df = pd.DataFrame([
            {"loss_sequence_id": 1},
            {"loss_sequence_id": 1},
            {"loss_sequence_id": 2},
        ])

        result = attach_sequence_coverage(
            df,
            candidates=5,
            sequence_id_column="loss_sequence_id",
        )

        coverage = result.attrs["data_coverage"]

        self.assertEqual(coverage["sequence_candidates"], 5)
        self.assertEqual(coverage["sequence_built"], 2)
        self.assertEqual(coverage["sequence_discarded"], 3)
        self.assertAlmostEqual(
            coverage["sequence_retention_pct"],
            40.0,
        )

    def test_empty_result_keeps_discarded_candidates_visible(self):
        result = attach_sequence_coverage(
            pd.DataFrame(),
            candidates=4,
            sequence_id_column="trigger_sequence_id",
        )

        coverage = result.attrs["data_coverage"]

        self.assertEqual(coverage["sequence_candidates"], 4)
        self.assertEqual(coverage["sequence_built"], 0)
        self.assertEqual(coverage["sequence_discarded"], 4)
        self.assertEqual(coverage["sequence_retention_pct"], 0.0)

    def test_zero_candidates_is_valid_metadata(self):
        result = attach_sequence_coverage(
            pd.DataFrame(),
            candidates=0,
            sequence_id_column="trigger_sequence_id",
        )

        coverage = result.attrs["data_coverage"]

        self.assertEqual(coverage["sequence_candidates"], 0)
        self.assertEqual(coverage["sequence_built"], 0)
        self.assertEqual(coverage["sequence_discarded"], 0)


if __name__ == "__main__":
    unittest.main()
