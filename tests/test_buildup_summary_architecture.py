import ast
import unittest
from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd

from src.components import buildup_summary_view
from src.metrics import buildup_metrics


ROOT = Path(__file__).resolve().parents[1]


def walk_components(component):
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, (list, tuple)):
        for child in children:
            if child is not None:
                yield from walk_components(child)
    elif not isinstance(children, (str, int, float, bool)):
        yield from walk_components(children)


def component_text(component):
    values = []

    def visit(value):
        if value is None:
            return
        if isinstance(value, (str, int, float)):
            values.append(str(value))
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
            return
        visit(getattr(value, "children", None))

    visit(component)
    return " ".join(values)


class BuildupSummaryArchitectureTests(unittest.TestCase):
    def test_buildup_metrics_has_no_dash_or_dbc_imports(self):
        path = ROOT / "src" / "metrics" / "buildup_metrics.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_modules.append(node.module or "")

        forbidden = [
            module
            for module in imported_modules
            if module == "dash"
            or module.startswith("dash.")
            or module == "dash_bootstrap_components"
            or module.startswith("dash_bootstrap_components.")
        ]

        self.assertEqual(forbidden, [])
        self.assertFalse(
            hasattr(buildup_metrics, "create_buildup_summary_cards")
        )

    def test_app_uses_buildup_summary_view(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn(
            "from src.components import buildup_summary_view",
            source,
        )
        self.assertIn(
            "buildup_summary_view.create_buildup_summary_cards(",
            source,
        )
        self.assertNotIn(
            "buildup_metrics.create_buildup_summary_cards(",
            source,
        )

    def test_calculate_buildup_stats_contract_is_unchanged(self):
        short_short_goal = pd.DataFrame([
            {
                "sequence_outcome_type": "Goals",
                "terminal_outcome": "goal",
                "dominant_flank": "Left",
                "buildup_type": "Short-Short",
                "type_name": "Pass",
                "lb": 0,
            }
        ])
        long_ball_shot = pd.DataFrame([
            {
                "sequence_outcome_type": "Shots",
                "terminal_outcome": "shot",
                "dominant_flank": "Right",
                "buildup_type": "Long Ball",
                "type_name": "Pass",
                "lb": 1,
            }
        ])

        result = buildup_metrics.calculate_buildup_stats(
            [short_short_goal, long_ball_shot],
            attacking_team_is_home=True,
        )

        self.assertEqual(
            result,
            {
                "total": 2,
                "outcomes": {"Goals": 1, "Shots": 1},
                "terminal_outcomes": {"goal": 1, "shot": 1},
                "flanks": {"Left": 1, "Right": 1},
                "types": {
                    "Short-Short": 1,
                    "Short-Long": 0,
                    "Long Ball": 1,
                },
            },
        )
        self.assertEqual(
            buildup_metrics.calculate_buildup_stats([], True),
            {},
        )

    def test_summary_view_renders_normal_data(self):
        stats = {
            "total": 2,
            "outcomes": {"Shots": 1, "Goals": 1},
            "terminal_outcomes": {"shot": 1, "goal": 1},
            "flanks": {"Left": 1, "Right": 1},
            "types": {
                "Short-Short": 1,
                "Short-Long": 0,
                "Long Ball": 1,
            },
        }

        component = buildup_summary_view.create_buildup_summary_cards(
            stats,
            active_filter={"types": "Short-Long"},
        )

        self.assertIsInstance(component, dbc.Row)
        self.assertEqual(component.className, "g-3 match-summary-grid")
        self.assertEqual(len(component.children), 3)

        text = component_text(component)
        self.assertIn("Buildup outcomes", text)
        self.assertIn("Dominant flank", text)
        self.assertIn("Initial buildup type", text)
        self.assertIn("Goals", text)
        self.assertIn("Shots", text)
        self.assertIn("Short-Long", text)
        self.assertIn("0 (0%)", text)

        filter_items = {
            tuple(sorted(node.id.items())): node
            for node in walk_components(component)
            if isinstance(getattr(node, "id", None), dict)
            and node.id.get("type") == "buildup-filter"
        }

        short_long_key = tuple(sorted({
            "type": "buildup-filter",
            "filter_type": "types",
            "value": "Short-Long",
        }.items()))

        self.assertIn(short_long_key, filter_items)
        self.assertTrue(filter_items[short_long_key].active)
        self.assertEqual(
            filter_items[short_long_key].className,
            "match-summary-row is-active",
        )

    def test_summary_view_renders_empty_sample(self):
        for stats in ({}, {"total": 0}):
            with self.subTest(stats=stats):
                component = (
                    buildup_summary_view.create_buildup_summary_cards(stats)
                )
                self.assertIsInstance(component, dbc.Alert)
                self.assertEqual(
                    component.children,
                    "No summary data to display.",
                )
                self.assertEqual(component.color, "secondary")


if __name__ == "__main__":
    unittest.main()
