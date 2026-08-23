from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import pandas as pd

import app
from src.components import final_third_view
from src.visualization import pass_plotly


def sample_entries():
    return pd.DataFrame([
        {
            "entry_type": "Pass",
            "x": 61.0,
            "y": 20.0,
            "end_x": 72.0,
            "end_y": 20.0,
            "playerName": "Passer A",
            "final_third_channel": "Right",
            "destination_zone": "Right Half-Space",
        },
        {
            "entry_type": "Pass",
            "x": 60.0,
            "y": 50.0,
            "end_x": 75.0,
            "end_y": 50.0,
            "playerName": "Passer B",
            "final_third_channel": "Central",
            "destination_zone": "Zone 14",
        },
        {
            "entry_type": "Pass",
            "x": 63.0,
            "y": 80.0,
            "end_x": 78.0,
            "end_y": 79.0,
            "playerName": "Passer C",
            "final_third_channel": "Left",
            "destination_zone": "Left Half-Space",
        },
        {
            "entry_type": "Carry",
            "x": 62.0,
            "y": 78.0,
            "end_x": 71.0,
            "end_y": 82.0,
            "playerName": "Carrier",
            "final_third_channel": "Left",
            "destination_zone": "Other",
            "carry_confidence": "high",
        },
    ])


BASE_STATS = {
    "total_final_third": 4,
    "pass_entries": 3,
    "carry_entries": 1,
    "carry_entry_candidates": 3,
    "carry_entries_excluded_confidence": 1,
    "carry_entries_excluded_boundary": 1,
    "carry_entries_excluded_total": 2,
    "channel_left": 2,
    "channel_central": 1,
    "channel_right": 1,
    "zone14": 1,
    "hs_left": 1,
    "hs_right": 1,
    "other": 1,
}


class FinalThirdSummaryViewTests(unittest.TestCase):
    def setUp(self):
        self.entries = sample_entries()

    def test_filter_separates_passes_and_inferred_carries(self):
        passes = final_third_view.filter_entries(
            self.entries,
            final_third_view.ENTRY_PASSES,
        )
        carries = final_third_view.filter_entries(
            self.entries,
            final_third_view.ENTRY_CARRIES,
        )

        self.assertEqual(len(passes), 3)
        self.assertEqual(len(carries), 1)
        self.assertTrue(passes["entry_type"].eq("Pass").all())
        self.assertTrue(carries["entry_type"].eq("Carry").all())

    def test_summary_counts_channels_and_destinations(self):
        summary = final_third_view.summarize_entries(
            self.entries,
            BASE_STATS,
        )

        self.assertEqual(summary["total_final_third"], 4)
        self.assertEqual(summary["channel_left"], 2)
        self.assertEqual(summary["channel_central"], 1)
        self.assertEqual(summary["channel_right"], 1)
        self.assertEqual(summary["zone14"], 1)
        self.assertEqual(summary["hs_left"], 1)
        self.assertEqual(summary["hs_right"], 1)
        self.assertEqual(summary["other"], 1)

    def test_filtered_summary_preserves_quality_diagnostics(self):
        carries = final_third_view.filter_entries(
            self.entries,
            final_third_view.ENTRY_CARRIES,
        )
        summary = final_third_view.summarize_entries(
            carries,
            BASE_STATS,
        )

        self.assertEqual(summary["total_final_third"], 1)
        self.assertEqual(summary["carry_entry_candidates"], 3)
        self.assertEqual(summary["carry_entries_excluded_total"], 2)

    def test_summary_renderer_has_no_individual_trajectory_lines(self):
        figure = pass_plotly.plot_final_third_summary_plotly(
            self.entries,
            BASE_STATS,
            "Team A",
            "#e96a4a",
            is_away=False,
        )

        self.assertEqual(len(figure.data), 0)
        self.assertGreater(len(figure.layout.shapes or []), 0)

    def test_summary_renderer_keeps_canonical_left_right_labels(self):
        figure = (
            pass_plotly
            .plot_final_third_summary_plotly(
                self.entries,
                BASE_STATS,
                "Team A",
                "#e96a4a",
                is_away=False,
            )
        )

        text = " ".join(
            str(
                annotation.text
            )
            for annotation
            in (
                figure.layout.annotations
                or []
            )
        )

        self.assertIn(
            "LEFT",
            text,
        )

        self.assertIn(
            "CENTRAL",
            text,
        )

        self.assertIn(
            "RIGHT",
            text,
        )

        # Destination-zone counts are intentionally kept out of
        # the pitch and live only in the sidebar.
        self.assertNotIn(
            "LEFT HS",
            text,
        )

        self.assertNotIn(
            "RIGHT HS",
            text,
        )

        self.assertNotIn(
            "ZONE 14",
            text,
        )

        self.assertNotIn(
            "OTHER",
            text,
        )

    def test_controls_default_to_all_and_hide_entries(self):
        component = final_third_view.controls()
        found = {}

        def walk(node):
            node_id = getattr(node, "id", None)
            if node_id:
                found[node_id] = node

            children = getattr(node, "children", None)
            if isinstance(children, (list, tuple)):
                for child in children:
                    walk(child)
            elif children is not None:
                walk(children)

        walk(component)

        self.assertEqual(
            found["final-third-entry-type"].value,
            final_third_view.ENTRY_ALL,
        )
        self.assertEqual(
            found["final-third-show-entries"].value,
            [],
        )

    def test_show_entries_switch_contract(self):
        self.assertFalse(
            final_third_view.show_entries_enabled([])
        )
        self.assertTrue(
            final_third_view.show_entries_enabled(
                [final_third_view.SHOW_ENTRIES]
            )
        )

    def test_active_callback_uses_summary_workspace(self):
        source = inspect.getsource(
            app.show_final_third_content_callback
        )

        self.assertIn("final_third_view", source)
        self.assertIn(".workspace(", source)
        self.assertIn('"final-third-view-content"', source)
        self.assertIn("render_data_coverage_panel", source)
        self.assertIn("_carry_coverage_item", source)

    def test_dynamic_callback_uses_entry_type_and_show_entries(self):
        source = inspect.getsource(
            app.update_final_third_view
        )

        self.assertIn("_build_final_third_view", source)
        self.assertIn("entry_type", source)
        self.assertIn("show_entries_enabled", source)

    def test_analyst_note_ids_are_preserved(self):
        source = inspect.getsource(
            app.show_final_third_content_callback
        )

        for component_id in (
            "comment-final-third",
            "save-comment-final-third",
            "save-status-final-third",
        ):
            self.assertIn(component_id, source)

    def test_legacy_static_image_helper_is_not_active_callback(self):
        source = Path("app.py").read_text(encoding="utf-8")

        callback_start = source.find(
            "def show_final_third_content_callback"
        )
        callback_end = source.find(
            "def save_final_third_comment",
            callback_start,
        )
        branch = source[callback_start:callback_end]

        self.assertNotIn("generate_final_third_plot(", branch)
        self.assertNotIn("plt.subplots(", branch)


    def test_summary_pitch_keeps_destination_counts_out_of_annotations(self):
        figure = (
            pass_plotly
            .plot_final_third_summary_plotly(
                self.entries,
                BASE_STATS,
                "Team A",
                "#e96a4a",
                is_away=False,
            )
        )

        text = " ".join(
            str(
                annotation.text
            )
            for annotation
            in (
                figure.layout.annotations
                or []
            )
        )

        self.assertIn(
            "LEFT",
            text,
        )
        self.assertIn(
            "CENTRAL",
            text,
        )
        self.assertIn(
            "RIGHT",
            text,
        )

        self.assertNotIn(
            "ZONE 14",
            text,
        )
        self.assertNotIn(
            "LEFT HS",
            text,
        )
        self.assertNotIn(
            "RIGHT HS",
            text,
        )
        self.assertNotIn(
            "OTHER",
            text,
        )

    def test_detail_wrapper_removes_legacy_internal_heading(self):
        figure = (
            pass_plotly
            .plot_final_third_entries_detail_plotly(
                self.entries,
                BASE_STATS,
                "Team A",
                "#e96a4a",
                is_away=False,
            )
        )

        title_text = (
            getattr(
                figure.layout.title,
                "text",
                None,
            )
            if figure.layout.title
            else None
        )

        self.assertFalse(
            title_text
        )

        annotation_text = " ".join(
            str(
                annotation.text
            )
            for annotation
            in (
                figure.layout.annotations
                or []
            )
        ).lower()

        self.assertNotIn(
            "final third entries",
            annotation_text,
        )

        self.assertFalse(
            (
                " total"
                in annotation_text
                and " pass"
                in annotation_text
                and "carry"
                in annotation_text
            )
        )



if __name__ == "__main__":
    unittest.main()
