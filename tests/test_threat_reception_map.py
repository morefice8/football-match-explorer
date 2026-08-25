import unittest
from unittest.mock import patch

import pandas as pd

from src.metrics import (
    threat_reception_metrics,
)
from src.visualization import (
    threat_reception_map,
)


def passes_frame():
    return pd.DataFrame([
        {
            "id": 1,
            "playerName": "Passer A",
            "receiver": "Target",
            "receiver_is_reliable": True,
            "team_name": "Home",
            "outcome": "Successful",
            "type_name": "Pass",
            "x": 20.0,
            "y": 40.0,
            "end_x": 70.0,
            "end_y": 45.0,
            "timeMin": 10,
            "timeSec": 2,
        },
        {
            "id": 2,
            "playerName": "Passer B",
            "receiver": "Target",
            "receiver_is_reliable": True,
            "team_name": "Home",
            "outcome": "Successful",
            "type_name": "Pass",
            "x": 42.0,
            "y": 55.0,
            "end_x": 86.0,
            "end_y": 50.0,
            "timeMin": 22,
            "timeSec": 4,
        },
    ])


def sequence_frame():
    return pd.DataFrame([
        {
            "id": 1,
            "playerName": "Passer A",
            "team_name": "Home",
            "outcome": "Successful",
            "type_name": "Pass",
            "sequence_role": "",
            "sequence_id": "A",
            "x": 20.0,
            "y": 40.0,
            "end_x": 70.0,
            "end_y": 45.0,
            "timeMin": 10,
            "timeSec": 2,
        },
        {
            "id": 9,
            "playerName": "Target",
            "team_name": "Home",
            "outcome": "Successful",
            "type_name": "Attempt Saved",
            "sequence_role": "shooter",
            "sequence_id": "A",
            "x": 82.0,
            "y": 48.0,
            "end_x": 100.0,
            "end_y": 48.0,
            "timeMin": 10,
            "timeSec": 10,
        },
    ])


class ThreatReceptionMetricsTests(
    unittest.TestCase
):
    @patch(
        "src.metrics.threat_reception_metrics."
        "shot_sequence_metrics.build_shot_sequences"
    )
    @patch(
        "src.metrics.threat_reception_metrics."
        "pass_processing.get_passes_df"
    )
    def test_flags_received_pass_by_shared_event_id(
        self,
        get_passes_df,
        build_shot_sequences,
    ):
        get_passes_df.return_value = (
            passes_frame()
        )
        build_shot_sequences.return_value = (
            sequence_frame()
        )

        profile = (
            threat_reception_metrics
            .received_passes_for_player(
                pd.DataFrame([
                    {
                        "dummy":
                            1,
                    }
                ]),
                "Target",
                "Home",
            )
        )

        flagged = (
            profile[
                profile[
                    "_in_shot_sequence"
                ]
            ]
        )

        self.assertEqual(
            flagged["id"].tolist(),
            [1],
        )


class ThreatReceptionPlotTests(
    unittest.TestCase
):
    def test_typical_zone_is_robust_not_convex_hull(
        self,
    ):
        profile = pd.DataFrame([
            {
                "playerName": "A",
                "x": 50,
                "y": 40,
                "end_x": 70,
                "end_y": 50,
                "_in_shot_sequence": False,
            },
            {
                "playerName": "B",
                "x": 55,
                "y": 42,
                "end_x": 72,
                "end_y": 52,
                "_in_shot_sequence": True,
            },
            {
                "playerName": "C",
                "x": 58,
                "y": 45,
                "end_x": 74,
                "end_y": 48,
                "_in_shot_sequence": False,
            },
            {
                "playerName": "D",
                "x": 5,
                "y": 5,
                "end_x": 10,
                "end_y": 95,
                "_in_shot_sequence": False,
            },
        ])

        summary = (
            threat_reception_metrics
            .reception_summary(
                profile
            )
        )

        fig = (
            threat_reception_map
            .plot_reception_profile(
                profile,
                selected_player=
                    "Target",
                jersey=
                    "9",
                team_color=
                    "#1597c2",
                summary=
                    summary,
            )
        )

        zone = next(
            trace
            for trace in fig.data
            if trace.name
            == "Typical reception zone"
        )

        self.assertLess(
            max(
                zone.x
            )
            - min(
                zone.x
            ),
            30,
        )

        self.assertLess(
            max(
                zone.y
            )
            - min(
                zone.y
            ),
            30,
        )

        names = {
            trace.name
            for trace in fig.data
            if trace.name
        }

        self.assertIn(
            "Shot-linked reception",
            names,
        )


if __name__ == "__main__":
    unittest.main()
