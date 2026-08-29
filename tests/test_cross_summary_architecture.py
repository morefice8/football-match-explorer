from __future__ import annotations

import ast
from pathlib import Path
import unittest

from dash import html
import pandas as pd

from src.components import cross_flow_view
from src.components import cross_summary_view
from src.metrics import cross_metrics


ROOT = Path(__file__).resolve().parents[1]


def sample_crosses():
    return pd.DataFrame([
        {
            "playerName": "Alice",
            "Play Type": "Open Play",
            "Foot": "Right",
            "Swing": "In-swinger",
            "Origin Zone": "Left Advanced",
            "Destination Zone": "Center Deep",
            "Outcome": "Completed",
            "Retained": True,
            "Shot Generated": True,
        },
        {
            "playerName": "Alice",
            "Play Type": "From Corner",
            "Foot": "Right",
            "Swing": "Straight",
            "Origin Zone": "Left Advanced",
            "Destination Zone": "Center Deep",
            "Outcome": "Incomplete",
            "Retained": False,
            "Shot Generated": True,
        },
        {
            "playerName": "Bob",
            "Play Type": "Open Play",
            "Foot": "Left",
            "Swing": "Out-swinger",
            "Origin Zone": "Right Advanced",
            "Destination Zone": "Right Deep",
            "Outcome": "Incomplete",
            "Retained": False,
            "Shot Generated": False,
        },
    ])


def walk(component):
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, (list, tuple)):
        for child in children:
            if child is not None:
                yield from walk(child)
    elif not isinstance(children, (str, int, float, bool)):
        yield from walk(children)


class CrossSummaryArchitectureTests(unittest.TestCase):
    def test_cross_metrics_has_no_dash_imports(self):
        source = (
            ROOT / "src/metrics/cross_metrics.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)

        forbidden = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                if (
                    module == "dash"
                    or module.startswith("dash.")
                    or module == "dash_bootstrap_components"
                    or module.startswith("dash_bootstrap_components.")
                ):
                    forbidden.append(module)

        self.assertEqual(forbidden, [])
        self.assertFalse(
            hasattr(cross_metrics, "create_cross_summary_cards")
        )
        self.assertTrue(
            hasattr(cross_summary_view, "create_cross_summary_cards")
        )

    def test_flow_summary_top_routes_and_top_crosser(self):
        data = sample_crosses()
        summary, routes = cross_metrics.build_cross_flow_profile(data, limit=8)

        self.assertEqual(summary["total_crosses"], 3)
        self.assertEqual(summary["retained_crosses"], 1)
        self.assertAlmostEqual(summary["retention_pct"], 100.0 / 3.0)
        self.assertEqual(summary["shot_crosses"], 2)
        self.assertAlmostEqual(summary["shot_rate_pct"], 200.0 / 3.0)
        self.assertEqual(summary["top_crosser"], "Alice")
        self.assertEqual(summary["top_crosser_count"], 2)
        self.assertEqual(
            summary["top_route"],
            "Left Advanced → Center Deep",
        )
        self.assertEqual(int(routes.iloc[0]["Crosses"]), 2)

    def test_summary_rendering_data_empty_and_filter_zero(self):
        data = sample_crosses()
        component = cross_summary_view.create_cross_summary_cards(
            data,
            active_filter={"taker": "Alice"},
        )
        self.assertIsInstance(component, html.Div)

        ids = [
            node.id
            for node in walk(component)
            if isinstance(getattr(node, "id", None), dict)
        ]
        self.assertTrue(
            any(
                item.get("type") == "cross-filter"
                and item.get("filter_type") == "taker"
                and item.get("value") == "Alice"
                for item in ids
            )
        )

        empty = cross_summary_view.create_cross_summary_cards(pd.DataFrame())
        self.assertIsInstance(empty, html.Div)
        self.assertFalse(bool(empty.children))

        filtered = data[data["Origin Zone"] == "Missing"]
        filtered_component = cross_summary_view.create_cross_summary_cards(
            filtered,
            active_filter={"origin": "Missing"},
        )
        self.assertIsInstance(filtered_component, html.Div)
        self.assertFalse(bool(filtered_component.children))

    def test_cross_flow_route_ids_are_preserved(self):
        data = sample_crosses()
        summary, routes = cross_metrics.build_cross_flow_profile(data, limit=8)
        component = cross_flow_view.flow_panel(summary, routes, "#000000")

        route_ids = [
            node.id
            for node in walk(component)
            if isinstance(getattr(node, "id", None), dict)
            and node.id.get("type") == "cross-flow-route"
        ]

        self.assertGreaterEqual(len(route_ids), 1)
        self.assertEqual(
            route_ids[0]["origin"],
            str(routes.iloc[0]["Origin Zone"]),
        )
        self.assertEqual(
            route_ids[0]["destination"],
            str(routes.iloc[0]["Destination Zone"]),
        )

    def test_cross_flow_to_maps_wiring_is_preserved(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }

        self.assertIn("select_cross_flow_route", functions)
        self.assertIn("update_cross_plots_on_selection", functions)

        selection_source = ast.get_source_segment(
            source,
            functions["select_cross_flow_route"],
        ) or ""
        self.assertIn("heatmaps-tab", selection_source)

        plot_fn = functions["update_cross_plots_on_selection"]
        calls = []
        for node in ast.walk(plot_fn):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "plot_cross_heatmap"
            ):
                calls.append({
                    kw.arg for kw in node.keywords if kw.arg is not None
                })

        self.assertEqual(len(calls), 2)
        for keywords in calls:
            self.assertIn("selected_flow_route", keywords)

    def test_app_uses_cross_summary_view(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        self.assertIn(
            "from src.components import cross_summary_view",
            source,
        )
        self.assertIn(
            "cross_summary_view.create_cross_summary_cards(",
            source,
        )

        old_calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "create_cross_summary_cards"
                and isinstance(func.value, ast.Name)
                and func.value.id == "cross_metrics"
            ):
                old_calls.append(node.lineno)

        self.assertEqual(old_calls, [])


if __name__ == "__main__":
    unittest.main()
