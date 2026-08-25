import unittest
from pathlib import Path

import pandas as pd

from src.metrics import restart_panel_metrics
from src.visualization import restart_map


def _restart_sequence():
    return pd.DataFrame(
        [
            {
                "trigger_sequence_id":
                    "restart-1",
                "restart_type":
                    "Corner",
                "restart_delivery_type":
                    "Direct Cross",
                "restart_execution_outcome":
                    "Successful Delivery",
                "x": 100.0,
                "y": 2.0,
                "end_x": 91.0,
                "end_y": 44.0,
                "timeMin": 12,
                "timeSec": 34,
                "playerName":
                    "Taker",
                "Mapped Jersey Number":
                    10,
                "type_name":
                    "Pass",
                "outcome":
                    "Successful",
                "sequence_outcome_type":
                    "Possession Retained",
            }
        ]
    )


class RestartPanelTests(
    unittest.TestCase
):
    def test_rel09_fields_override_legacy_presentation_fields(self):
        analyzed = pd.DataFrame(
            [
                {
                    "sequence_id":
                        "restart-1",
                    "Action Type":
                        "Legacy Type",
                    "Side":
                        "Right",
                    "Delivery":
                        "Legacy Delivery",
                    "Destination":
                        "Penalty Area",
                    "Outcome":
                        "Legacy Outcome",
                    "Player":
                        "Legacy Taker",
                    "x_start":
                        100.0,
                    "y_start":
                        2.0,
                    "x_end":
                        91.0,
                    "y_end":
                        44.0,
                }
            ]
        )

        records = (
            restart_panel_metrics
            .build_restart_records(
                analyzed,
                [
                    _restart_sequence()
                ],
            )
        )

        self.assertEqual(
            records[0][
                "restart_type"
            ],
            "Corner",
        )
        self.assertEqual(
            records[0][
                "delivery"
            ],
            "Direct Cross",
        )
        self.assertEqual(
            records[0][
                "outcome"
            ],
            "Successful Delivery",
        )
        self.assertEqual(
            records[0][
                "side"
            ],
            "Right",
        )
        self.assertEqual(
            records[0][
                "destination"
            ],
            "Penalty Area",
        )

    def test_filters_cover_requested_restart_dimensions(self):
        records = [
            {
                "sequence_id":
                    "a",
                "restart_type":
                    "Corner",
                "side":
                    "Right",
                "delivery":
                    "Direct Cross",
                "destination":
                    "Penalty Area",
                "outcome":
                    "Successful Delivery",
            },
            {
                "sequence_id":
                    "b",
                "restart_type":
                    "Free Kick",
                "side":
                    "Left",
                "delivery":
                    "Long Pass",
                "destination":
                    "Final Third",
                "outcome":
                    "Unsuccessful Delivery",
            },
        ]

        filtered = (
            restart_panel_metrics
            .apply_restart_filters(
                records,
                {
                    "action":
                        "Corner",
                    "delivery":
                        "Direct Cross",
                },
            )
        )

        self.assertEqual(
            [
                record[
                    "sequence_id"
                ]
                for record
                in filtered
            ],
            ["a"],
        )

    def test_aggregated_map_exposes_sequence_id_for_click_selection(self):
        records = [
            {
                "sequence_id":
                    "restart-1",
                "restart_type":
                    "Corner",
                "side":
                    "Right",
                "delivery":
                    "Direct Cross",
                "destination":
                    "Penalty Area",
                "outcome":
                    "Successful Delivery",
                "player_name":
                    "Taker",
                "jersey_number":
                    10,
                "start_x":
                    100.0,
                "start_y":
                    2.0,
                "end_x":
                    91.0,
                "end_y":
                    44.0,
                "match_second":
                    754.0,
            }
        ]

        fig = (
            restart_map
            .plot_restart_map(
                records,
                team_color=
                    "#e96a4a",
            )
        )

        trace = next(
            trace
            for trace in fig.data
            if trace.type
            == "scatter"
        )

        self.assertEqual(
            trace.customdata[0][0],
            "restart-1",
        )

    def test_active_restart_view_replaces_carousel_and_legacy_map(self):
        source = Path(
            "app.py"
        ).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            '"restart-map-graph"',
            source,
        )
        self.assertIn(
            '"restart-selected-sequence"',
            source,
        )
        self.assertIn(
            "select_restart_from_map",
            source,
        )

        self.assertNotIn(
            "plot_set_piece_map(",
            source,
        )
        self.assertNotIn(
            "set-piece-carousel-content",
            source,
        )
        self.assertNotIn(
            "set-piece-prev-button",
            source,
        )
        self.assertNotIn(
            "set-piece-next-button",
            source,
        )


    def test_missing_destination_gets_spatial_fallback(self):
        sequence = _restart_sequence()

        analyzed = pd.DataFrame(
            [
                {
                    "sequence_id": "restart-1",
                    "Action Type": "Corner",
                    "Side": "Right",
                    "Delivery": "Direct Cross",
                    "Destination": "N/A",
                    "Outcome": "Successful Delivery",
                    "Player": "Taker",
                    "x_start": 100.0,
                    "y_start": 2.0,
                    "x_end": 91.0,
                    "y_end": 44.0,
                }
            ]
        )

        records = (
            restart_panel_metrics
            .build_restart_records(
                analyzed,
                [sequence],
            )
        )

        self.assertEqual(
            records[0]["destination"],
            "Penalty Area",
        )

    def test_spatial_destination_uses_pitch_thirds(self):
        self.assertEqual(
            restart_panel_metrics
            ._spatial_destination(20, 50),
            "Own Third",
        )
        self.assertEqual(
            restart_panel_metrics
            ._spatial_destination(50, 50),
            "Middle Third",
        )
        self.assertEqual(
            restart_panel_metrics
            ._spatial_destination(75, 10),
            "Final Third",
        )

    def test_restart_map_explains_line_style_semantics(self):
        fig = (
            restart_map
            .plot_restart_map(
                [
                    {
                        "sequence_id": "restart-1",
                        "restart_type": "Corner",
                        "side": "Right",
                        "delivery": "Direct Cross",
                        "destination": "Penalty Area",
                        "outcome": "Successful Delivery",
                        "player_name": "Taker",
                        "jersey_number": 10,
                        "start_x": 100.0,
                        "start_y": 2.0,
                        "end_x": 91.0,
                        "end_y": 44.0,
                        "match_second": 754.0,
                    }
                ],
                team_color="#e96a4a",
            )
        )

        annotation_text = " ".join(
            str(annotation.text)
            for annotation
            in fig.layout.annotations
        )

        self.assertIn("Solid", annotation_text)
        self.assertIn("Dotted", annotation_text)

if __name__ == "__main__":
    unittest.main()
