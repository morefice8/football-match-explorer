from __future__ import annotations

from pathlib import Path

import inspect
import unittest

from dash import dcc
import numpy as np
import pandas as pd

import app
from src.components import pass_location_view
from src.visualization import pass_plotly


class PassLocationViewTests(unittest.TestCase):

    def setUp(self):
        self.home = pd.DataFrame(
            {
                "team_name": ["Home"] * 8,
                "x": [12, 18, 22, 28, 65, 70, 74, 82],
                "y": [18, 22, 25, 28, 42, 45, 48, 52],
            }
        )

        self.away = pd.DataFrame(
            {
                "team_name": ["Away"] * 10,
                "x": [10, 14, 18, 45, 48, 52, 70, 74, 78, 82],
                "y": [70, 72, 74, 48, 50, 52, 20, 22, 24, 26],
            }
        )

        self.all_passes = pd.concat(
            [
                self.home,
                self.away,
            ],
            ignore_index=True,
        )

    @staticmethod
    def _walk(component):
        yield component

        children = getattr(
            component,
            "children",
            None,
        )

        if isinstance(
            children,
            (list, tuple),
        ):
            for child in children:
                yield from (
                    PassLocationViewTests
                    ._walk(
                        child
                    )
                )
        elif children is not None:
            yield from (
                PassLocationViewTests
                ._walk(
                    children
                )
            )

    def test_controls_default_to_density(self):
        controls = (
            pass_location_view
            .controls()
        )

        toggle = next(
            component
            for component in self._walk(
                controls
            )
            if getattr(
                component,
                "id",
                None,
            )
            == "pass-location-view-toggle"
        )

        self.assertEqual(
            toggle.value,
            pass_location_view.VIEW_DENSITY,
        )

    def test_density_uses_percentage_normalisation(self):
        figure = (
            pass_plotly
            .plot_pass_density_plotly(
                self.home,
                "Home",
                shared_zmax=25.0,
            )
        )

        density = next(
            trace
            for trace in figure.data
            if trace.type
            == "histogram2dcontour"
        )

        self.assertEqual(
            density.histnorm,
            "percent",
        )

        self.assertEqual(
            float(density.zmax),
            25.0,
        )

    def test_grid_percentages_sum_to_100(self):
        figure = (
            pass_plotly
            .plot_pass_heatmap_plotly(
                self.home,
                "Home",
                shared_zmax=30.0,
            )
        )

        heatmap = next(
            trace
            for trace in figure.data
            if trace.type == "heatmap"
        )

        self.assertAlmostEqual(
            float(
                np.asarray(
                    heatmap.z,
                    dtype=float,
                ).sum()
            ),
            100.0,
            places=6,
        )

    def test_grid_hover_preserves_raw_counts(self):
        figure = (
            pass_plotly
            .plot_pass_heatmap_plotly(
                self.home,
                "Home",
            )
        )

        heatmap = next(
            trace
            for trace in figure.data
            if trace.type == "heatmap"
        )

        self.assertIn(
            "passes",
            heatmap.hovertemplate,
        )

        self.assertIsNotNone(
            heatmap.customdata,
        )

    def test_home_and_away_use_one_shared_percentage_scale(self):
        scales = (
            pass_plotly
            .pass_location_shared_scales(
                self.home,
                self.away,
            )
        )

        home_grid = (
            pass_plotly
            .plot_pass_heatmap_plotly(
                self.home,
                "Home",
                shared_zmax=scales["grid"],
            )
        )

        away_grid = (
            pass_plotly
            .plot_pass_heatmap_plotly(
                self.away,
                "Away",
                is_away=True,
                shared_zmax=scales["grid"],
            )
        )

        home_trace = next(
            trace
            for trace in home_grid.data
            if trace.type == "heatmap"
        )

        away_trace = next(
            trace
            for trace in away_grid.data
            if trace.type == "heatmap"
        )

        self.assertEqual(
            float(home_trace.zmax),
            float(away_trace.zmax),
        )

    def test_each_mode_renders_one_graph_per_team(self):
        for view_mode in (
            pass_location_view.VIEW_DENSITY,
            pass_location_view.VIEW_GRID,
        ):
            panels = (
                pass_location_view
                .panels(
                    self.all_passes,
                    "Home",
                    "Away",
                    view_mode=view_mode,
                )
            )

            graphs = [
                component
                for component in self._walk(
                    panels
                )
                if isinstance(
                    component,
                    dcc.Graph,
                )
            ]

            self.assertEqual(
                len(graphs),
                2,
            )

    def test_workspace_keeps_toggle_visible(self):
        shell = dcc.Loading(
            "dummy"
        )

        workspace = (
            pass_location_view
            .workspace(
                shell
            )
        )

        ids = {
            getattr(
                component,
                "id",
                None,
            )
            for component in self._walk(
                workspace
            )
        }

        self.assertIn(
            "pass-location-view-toggle",
            ids,
        )

    def test_app_uses_dynamic_density_grid_workspace(self):
        source = inspect.getsource(
            app.show_pass_location_plots_callback
        )

        self.assertIn(
            "pass_location_view",
            source,
        )

        self.assertIn(
            "pass-location-view-content",
            source,
        )

        self.assertNotIn(
            "fig_home_heatmap",
            source,
        )

        self.assertNotIn(
            "fig_home_density",
            source,
        )

    def test_toggle_callback_rebuilds_selected_view(self):
        source = inspect.getsource(
            app.update_pass_location_view
        )

        self.assertIn(
            "_build_pass_location_view",
            source,
        )

        self.assertIn(
            "view_mode",
            source,
        )


    def test_density_has_no_duplicate_internal_team_heading(self):
        figure = pass_plotly.plot_pass_density_plotly(
            self.home,
            "Home",
            shared_zmax=25.0,
        )

        text = " ".join(
            str(annotation.text)
            for annotation in (
                figure.layout.annotations
                or []
            )
        )

        self.assertNotIn(
            "Pass origin density",
            text,
        )

        self.assertLess(
            int(figure.layout.height),
            500,
        )

    def test_grid_has_no_duplicate_internal_team_heading(self):
        figure = pass_plotly.plot_pass_heatmap_plotly(
            self.home,
            "Home",
            shared_zmax=30.0,
        )

        text = " ".join(
            str(annotation.text)
            for annotation in (
                figure.layout.annotations
                or []
            )
        )

        self.assertNotIn(
            "Pass origin grid",
            text,
        )

        self.assertLess(
            int(figure.layout.height),
            500,
        )

    def test_pass_locations_use_shared_analyst_notes_ids(self):
        source = inspect.getsource(
            app.show_pass_location_plots_callback
        )

        for component_id in (
            "comment-pass-locations",
            "save-comment-pass-locations",
            "save-status-pass-locations",
        ):
            self.assertIn(
                component_id,
                source,
            )

    def test_pass_locations_tab_no_longer_renders_legacy_comment_editor(self):
        source = Path(
            app.__file__
        ).read_text(
            encoding="utf-8"
        )

        start = source.find(
            'label="Pass Locations"'
        )
        end = source.find(
            'label="Crosses"',
            start,
        )

        self.assertGreaterEqual(
            start,
            0,
        )
        self.assertGreater(
            end,
            start,
        )

        branch = source[
            start:end
        ]

        self.assertNotIn(
            "Comments for Pass Locations:",
            branch,
        )

        self.assertNotIn(
            "Enter your analysis on pass locations",
            branch,
        )

if __name__ == "__main__":
    unittest.main()
