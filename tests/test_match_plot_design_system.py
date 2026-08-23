import unittest

import pandas as pd
import plotly.graph_objects as go

from src.visualization.plotly_branding import (
    MATCH_AWAY_CYAN,
    MATCH_CARD_BG,
    MATCH_COMPARE_HEIGHT,
    MATCH_DISPLAY_FONT,
    MATCH_HOME_CORAL,
    MATCH_PITCH_BG,
    add_attacking_direction,
    add_plot_header,
    add_zero_state,
    apply_match_pitch_layout,
    get_team_palette,
)
from src.visualization.pass_plotly import (
    plot_progressive_passes_plotly,
)


class MatchPlotDesignSystemTests(unittest.TestCase):

    def test_home_away_palette_contract(self):
        self.assertEqual(
            get_team_palette(
                is_away=False
            )["primary"],
            MATCH_HOME_CORAL,
        )
        self.assertEqual(
            get_team_palette(
                is_away=True
            )["primary"],
            MATCH_AWAY_CYAN,
        )

    def test_pitch_layout_uses_white_paper_and_dark_pitch(self):
        fig = go.Figure()

        apply_match_pitch_layout(
            fig,
            pitch_shapes=[],
        )

        self.assertEqual(
            fig.layout.paper_bgcolor,
            MATCH_CARD_BG,
        )
        self.assertEqual(
            fig.layout.plot_bgcolor,
            MATCH_PITCH_BG,
        )
        self.assertEqual(
            fig.layout.height,
            MATCH_COMPARE_HEIGHT,
        )
        self.assertIsNone(
            fig.layout.title.text
        )

    def test_comparison_figures_have_fixed_equal_height(self):
        home = go.Figure()
        away = go.Figure()

        apply_match_pitch_layout(
            home,
            pitch_shapes=[],
        )
        apply_match_pitch_layout(
            away,
            pitch_shapes=[],
        )

        self.assertEqual(
            home.layout.height,
            away.layout.height,
        )

    def test_legend_has_reserved_above_pitch_position(self):
        fig = go.Figure()

        apply_match_pitch_layout(
            fig,
            pitch_shapes=[],
            showlegend=True,
        )

        self.assertEqual(
            fig.layout.legend.orientation,
            "h",
        )
        self.assertGreater(
            float(fig.layout.legend.y),
            1.0,
        )
        self.assertGreaterEqual(
            int(fig.layout.margin.t),
            52,
        )
        self.assertLess(
            int(fig.layout.margin.t),
            90,
        )

    def test_shared_legend_font_is_readable(self):
        fig = go.Figure()

        apply_match_pitch_layout(
            fig,
            pitch_shapes=[],
            showlegend=True,
        )

        self.assertGreaterEqual(
            int(fig.layout.legend.font.size),
            13,
        )

    def test_header_uses_display_font(self):
        fig = go.Figure()

        add_plot_header(
            fig,
            "Example",
            dark=False,
        )

        title_annotation = (
            fig.layout.annotations[0]
        )

        self.assertIn(
            "IBM Plex Sans",
            title_annotation.font.family,
        )
        self.assertEqual(
            title_annotation.font.family,
            MATCH_DISPLAY_FONT,
        )

    def test_zero_state_and_attacking_direction_are_shared_annotations(self):
        fig = go.Figure()

        add_zero_state(
            fig,
            "No data",
        )
        add_attacking_direction(
            fig,
            dark=True,
        )

        texts = [
            annotation.text
            for annotation
            in fig.layout.annotations
        ]

        self.assertIn(
            "<b>No data</b>",
            texts,
        )
        self.assertTrue(
            any(
                "ATTACK" in text.upper()
                for text in texts
            )
        )

    def test_demo_progressive_plot_keeps_metric_traces_but_uses_brand_palette(self):
        df = pd.DataFrame([
            {
                "playerName": "Home A",
                "timeMin": 10,
                "x": 20.0,
                "y": 40.0,
                "end_x": 55.0,
                "end_y": 42.0,
                "is_progressive_attempt": True,
                "is_progressive": True,
                "progressive_distance_m": 31.0,
                "progressive_threshold_m": 30.0,
                "progressive_phase": "Own half",
                "progressive_channel": "Central",
            },
            {
                "playerName": "Home B",
                "timeMin": 20,
                "x": 55.0,
                "y": 60.0,
                "end_x": 68.0,
                "end_y": 62.0,
                "is_progressive_attempt": True,
                "is_progressive": False,
                "progressive_distance_m": 8.0,
                "progressive_threshold_m": 10.0,
                "progressive_phase": "Opposition half",
                "progressive_channel": "Right",
            },
        ])

        source = df.copy(
            deep=True
        )

        home_fig = (
            plot_progressive_passes_plotly(
                df,
                "Home",
                "#000000",
                is_away=False,
            )
        )

        away_fig = (
            plot_progressive_passes_plotly(
                df,
                "Away",
                "#ffffff",
                is_away=True,
            )
        )

        # Two semantic categories remain present.
        home_names = [
            trace.name
            for trace in home_fig.data
            if trace.showlegend is not False
        ]

        self.assertIn(
            "Completed (1)",
            home_names,
        )
        self.assertIn(
            "Incomplete (1)",
            home_names,
        )

        completed_home = next(
            trace
            for trace in home_fig.data
            if trace.name == "Completed (1)"
        )
        completed_away = next(
            trace
            for trace in away_fig.data
            if trace.name == "Completed (1)"
        )

        self.assertEqual(
            completed_home.line.color,
            MATCH_HOME_CORAL,
        )
        self.assertEqual(
            completed_away.line.color,
            MATCH_AWAY_CYAN,
        )

        # No inner title: the Dash team card owns it.
        self.assertIsNone(
            home_fig.layout.title.text
        )

        # Presentation code must not mutate the metric DataFrame.
        pd.testing.assert_frame_equal(
            df,
            source,
        )


if __name__ == "__main__":
    unittest.main()
