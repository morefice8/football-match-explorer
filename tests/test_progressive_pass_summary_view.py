import inspect
import ast
import unittest
from pathlib import Path

import pandas as pd
from dash import html

import app
from src.components import progressive_pass_view
from src.metrics import pass_metrics
from src.visualization import pass_plotly


def sample_progressive_passes():
    rows = []

    def add(
        player,
        *,
        x,
        y,
        end_x,
        end_y,
        completed,
        gain,
        channel,
    ):
        rows.append({
            "id": len(rows) + 1,
            "team_name": "Team A",
            "playerName": player,
            "x": x,
            "y": y,
            "end_x": end_x,
            "end_y": end_y,
            "is_progressive_attempt": True,
            "is_progressive": completed,
            "progressive_distance_m": gain,
            "progressive_channel": channel,
        })

    add(
        "Player A",
        x=30, y=20, end_x=55, end_y=24,
        completed=True, gain=18, channel="Right",
    )
    add(
        "Player A",
        x=31, y=21, end_x=56, end_y=25,
        completed=True, gain=19, channel="Right",
    )
    add(
        "Player A",
        x=32, y=22, end_x=57, end_y=26,
        completed=False, gain=18, channel="Right",
    )
    add(
        "Player B",
        x=46, y=50, end_x=72, end_y=52,
        completed=True, gain=14, channel="Central",
    )
    add(
        "Player B",
        x=47, y=51, end_x=73, end_y=53,
        completed=True, gain=15, channel="Central",
    )
    add(
        "Player C",
        x=50, y=82, end_x=79, end_y=76,
        completed=True, gain=16, channel="Left",
    )

    return pd.DataFrame(rows)


class ProgressivePassSummaryViewTests(unittest.TestCase):
    def setUp(self):
        self.passes = sample_progressive_passes()

    def test_existing_summary_metrics_are_unchanged(self):
        summary = pass_metrics.progressive_pass_summary(
            self.passes
        )

        self.assertEqual(summary["attempted"], 6)
        self.assertEqual(summary["successful"], 5)
        self.assertEqual(
            summary["channel_counts"],
            {
                "Left": 1,
                "Central": 2,
                "Right": 3,
            },
        )

    def test_top_progressor_uses_existing_player_ranking(self):
        ranking = (
            pass_metrics
            .progressive_pass_player_summary(
                self.passes,
                limit=1,
            )
        )

        self.assertEqual(
            ranking.iloc[0]["Player"],
            "Player A",
        )
        self.assertEqual(
            int(ranking.iloc[0]["Successful"]),
            2,
        )

    def test_summary_map_aggregates_locations(self):
        aggregated = (
            pass_plotly
            ._progressive_pass_aggregate_locations(
                self.passes
            )
        )

        origins = aggregated[
            aggregated["kind"].eq("Origin")
        ]

        self.assertLess(
            len(origins),
            len(self.passes),
        )
        self.assertEqual(
            int(origins["count"].sum()),
            len(self.passes),
        )

    def test_summary_map_has_limited_aggregated_routes(self):
        routes = (
            pass_plotly
            ._progressive_pass_aggregate_routes(
                self.passes,
                top_n=8,
            )
        )

        self.assertLessEqual(
            len(routes),
            8,
        )

        self.assertGreater(
            len(routes),
            0,
        )

        figure = (
            pass_plotly
            .plot_progressive_pass_summary_plotly(
                self.passes,
                "Team A",
                "#e96a4a",
                False,
            )
        )

        route_annotations = [
            annotation
            for annotation
            in figure.layout.annotations
            if (
                bool(
                    annotation.showarrow
                )
                and annotation.text == ""
            )
        ]

        self.assertLessEqual(
            len(
                route_annotations
            ),
            8,
        )

        self.assertGreater(
            len(
                route_annotations
            ),
            0,
        )

        self.assertFalse(
            any(
                (
                    getattr(
                        trace,
                        "mode",
                        "",
                    )
                    and "lines"
                    in getattr(
                        trace,
                        "mode",
                        "",
                    )
                )
                for trace
                in figure.data
            )
        )

    def test_route_tooltip_exposes_length_and_progression(self):
        figure = (
            pass_plotly
            .plot_progressive_pass_summary_plotly(
                self.passes,
                "Team A",
                "#e96a4a",
                False,
            )
        )

        hover_templates = " ".join(
            str(
                getattr(
                    trace,
                    "hovertemplate",
                    "",
                )
            )
            for trace in figure.data
        )

        self.assertIn(
            "Average pass length",
            hover_templates,
        )

        self.assertIn(
            "Average progression",
            hover_templates,
        )

    def test_all_attempts_mode_builds_existing_renderer(self):
        panel = progressive_pass_view.team_panel(
            self.passes,
            "Team A",
            "#e96a4a",
            is_away=False,
            view_mode=(
                progressive_pass_view
                .VIEW_ALL_ATTEMPTS
            ),
        )

        self.assertIsNotNone(panel)

    def test_toggle_defaults_to_summary(self):
        control = progressive_pass_view.controls()
        radio = control.children[0].children[1]

        self.assertEqual(
            radio.value,
            progressive_pass_view.VIEW_SUMMARY,
        )
        self.assertEqual(
            {
                option["value"]
                for option in radio.options
            },
            {
                "summary",
                "all_attempts",
            },
        )

    def test_app_callback_uses_summary_as_initial_view(self):
        source = inspect.getsource(
            app.show_progressive_passes_content_callback
        )

        self.assertIn("VIEW_SUMMARY", source)
        self.assertIn(
            "progressive-pass-view-content",
            source,
        )

    def test_analyst_note_ids_are_preserved(self):
        source = inspect.getsource(
            app.show_progressive_passes_content_callback
        )

        self.assertIn(
            "comment-progressive-passes",
            source,
        )
        self.assertIn(
            "save-comment-progressive-passes",
            source,
        )
        self.assertIn(
            "save-status-progressive-passes",
            source,
        )


    def test_team_panel_uses_supported_match_graph_panel_api(self):
        source = inspect.getsource(
            progressive_pass_view.team_panel
        )

        self.assertIn(
            "match_graph_panel(",
            source,
        )
        self.assertNotIn(
            'class_name="progressive-summary-team-panel"',
            source,
        )



    def test_toggle_is_rendered_outside_graph_shell(self):
        callback_source = inspect.getsource(
            app.show_progressive_passes_content_callback
        )

        self.assertIn(
            "progressive_pass_view",
            callback_source,
        )
        self.assertIn(
            ".workspace(",
            callback_source,
        )

        shell = html.Div(
            "dummy shell"
        )

        workspace = (
            progressive_pass_view
            .workspace(
                shell
            )
        )

        ids = set()

        def walk(component):
            component_id = getattr(
                component,
                "id",
                None,
            )

            if component_id:
                ids.add(
                    component_id
                )

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
                    walk(
                        child
                    )
            elif children is not None:
                walk(
                    children
                )

        walk(
            workspace
        )

        self.assertIn(
            "progressive-view-toggle",
            ids,
        )


def test_route_tooltip_exposes_length_and_progression(self):
        figure = (
            pass_plotly
            .plot_progressive_pass_summary_plotly(
                self.passes,
                "Team A",
                "#e96a4a",
                False,
            )
        )

        hover_templates = " ".join(
            str(
                getattr(
                    trace,
                    "hovertemplate",
                    "",
                )
            )
            for trace
            in figure.data
        )

        self.assertIn(
            "Average pass length",
            hover_templates,
        )

        self.assertIn(
            "Average progression",
            hover_templates,
        )

def test_tactical_zone_contract_is_four_by_three(self):
        # y low = attacking right; y high = attacking left.
        self.assertEqual(
            pass_plotly
            ._progressive_pass_tactical_zone(
                10,
                10,
            ),
            (0, 2),
        )

        self.assertEqual(
            pass_plotly
            ._progressive_pass_tactical_zone(
                49,
                50,
            ),
            (1, 1),
        )

        self.assertEqual(
            pass_plotly
            ._progressive_pass_tactical_zone(
                76,
                90,
            ),
            (3, 0),
        )

def test_workspace_always_contains_view_toggle(self):
    shell = html.Div(
        "dummy shell"
    )

    workspace = (
        progressive_pass_view
        .workspace(
            shell
        )
    )

    ids = set()

    def walk(component):
        component_id = getattr(
            component,
            "id",
            None,
        )

        if component_id:
            ids.add(
                component_id
            )

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
                walk(
                    child
                )

        elif children is not None:
            walk(
                children
            )

    walk(
        workspace
    )

    self.assertIn(
        "progressive-view-toggle",
        ids,
    )


    def test_route_attempt_badges_are_offset_from_arrow_midpoints(self):
        routes = (
            pass_plotly
            ._progressive_pass_aggregate_routes(
                self.passes,
                top_n=8,
            )
        )

        figure = (
            pass_plotly
            .plot_progressive_pass_summary_plotly(
                self.passes,
                "Team A",
                "#e96a4a",
                False,
            )
        )

        badge_trace = next(
            trace
            for trace in figure.data
            if getattr(
                trace,
                "name",
                None,
            ) == "Top tactical routes"
        )

        self.assertEqual(
            len(badge_trace.x),
            len(routes),
        )

        for index, row in routes.iterrows():
            midpoint_x = (
                float(row["start_x"])
                + float(row["end_x"])
            ) / 2.0

            midpoint_y = (
                float(row["start_y"])
                + float(row["end_y"])
            ) / 2.0

            distance_from_arrow_midpoint = (
                (
                    float(
                        badge_trace.x[index]
                    )
                    - midpoint_x
                ) ** 2
                + (
                    float(
                        badge_trace.y[index]
                    )
                    - midpoint_y
                ) ** 2
            ) ** 0.5

            self.assertGreater(
                distance_from_arrow_midpoint,
                2.0,
            )



    def test_progressive_tab_output_is_bound_to_workspace_callback(self):
        source = Path(
            "app.py"
        ).read_text(
            encoding="utf-8"
        )

        tree = ast.parse(
            source
        )

        functions = {
            node.name: node
            for node in tree.body
            if isinstance(
                node,
                ast.FunctionDef,
            )
        }

        helper = functions[
            "_build_progressive_passes_view"
        ]

        renderer = functions[
            "show_progressive_passes_content_callback"
        ]

        def decorators(node):
            return [
                (
                    ast.get_source_segment(
                        source,
                        decorator,
                    )
                    or ""
                )
                for decorator
                in node.decorator_list
            ]

        helper_decorators = decorators(
            helper
        )

        renderer_decorators = decorators(
            renderer
        )

        self.assertFalse(
            any(
                "div-progressive-passes-content"
                in decorator
                for decorator
                in helper_decorators
            )
        )

        self.assertTrue(
            any(
                "div-progressive-passes-content"
                in decorator
                for decorator
                in renderer_decorators
            )
        )

        renderer_source = (
            ast.get_source_segment(
                source,
                renderer,
            )
            or ""
        )

        self.assertIn(
            ".workspace(",
            renderer_source,
        )



if __name__ == "__main__":
    unittest.main()
