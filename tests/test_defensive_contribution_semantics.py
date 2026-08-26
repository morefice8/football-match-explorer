import unittest

import pandas as pd

from src.metrics import (
    defensive_contribution_metrics,
)


class DefensiveContributionSemanticTests(
    unittest.TestCase
):
    def test_committed_fouls_exclude_fouls_suffered(
        self,
    ):
        frame = pd.DataFrame([
            {
                "id": 1,
                "typeId": 4,
                "type_name": "Foul",
                "outcome": "Unsuccessful",
            },
            {
                "id": 2,
                "typeId": 4,
                "type_name": "Foul",
                "outcome": "Successful",
            },
        ])

        profile = (
            defensive_contribution_metrics
            .player_defensive_profile(
                frame
            )
        )

        self.assertEqual(
            profile["fouls"],
            1,
        )

    def test_blocked_pass_type_id_counts_as_block(
        self,
    ):
        frame = pd.DataFrame([
            {
                "id": 1,
                "typeId": 74,
                "type_name": "Blocked Pass",
                "outcome": "Successful",
            }
        ])

        profile = (
            defensive_contribution_metrics
            .player_defensive_profile(
                frame
            )
        )

        self.assertEqual(
            profile["blocks"],
            1,
        )
        self.assertEqual(
            profile["unique"],
            1,
        )

    def test_event10_requires_def_block_qualifier(
        self,
    ):
        frame = pd.DataFrame([
            {
                "id": 1,
                "typeId": 10,
                "type_name": "Save",
                "outcome": "Successful",
                "Def block": 1,
            },
            {
                "id": 2,
                "typeId": 10,
                "type_name": "Save",
                "outcome": "Successful",
                "Def block": 0,
            },
        ])

        profile = (
            defensive_contribution_metrics
            .player_defensive_profile(
                frame
            )
        )

        self.assertEqual(
            profile["blocks"],
            1,
        )
        self.assertEqual(
            profile["unique"],
            1,
        )

    def test_raw_type_fallbacks_keep_core_metrics_stable(
        self,
    ):
        frame = pd.DataFrame([
            {
                "id": 1,
                "typeId": 7,
                "type_name": "Something",
                "outcome": "Successful",
            },
            {
                "id": 2,
                "typeId": 8,
                "type_name": "Something",
                "outcome": "Successful",
            },
            {
                "id": 3,
                "typeId": 49,
                "type_name": "Something",
                "outcome": "Successful",
            },
            {
                "id": 4,
                "typeId": 12,
                "type_name": "Something",
                "outcome": "Successful",
            },
        ])

        profile = (
            defensive_contribution_metrics
            .player_defensive_profile(
                frame
            )
        )

        self.assertEqual(
            profile["unique"],
            4,
        )
        self.assertEqual(
            profile["tackles_won"],
            1,
        )
        self.assertEqual(
            profile["interceptions"],
            1,
        )
        self.assertEqual(
            profile["recoveries"],
            1,
        )
        self.assertEqual(
            profile["clearances"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
