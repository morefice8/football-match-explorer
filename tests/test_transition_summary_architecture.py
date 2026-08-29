from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
import unittest

import dash_bootstrap_components as dbc
from dash import dash_table
import numpy as np
import pandas as pd

from src.components import transition_summary_view
from src.metrics import transition_metrics


ROOT = Path(__file__).resolve().parents[1]
BASELINE = {'defensive': {'flanks': {'Center': 1, 'Right': 1}, 'outcomes': {'Goals conceded': 1, 'Unknown': 1}, 'terminal_outcomes': {'goal': 1, 'unknown': 1}, 'total': 2, 'transition_profile_table': [{'Avg Duration (s)': 5.0, 'Avg Passes': 1.0, 'Counterattack Side': 'Right', 'Loss Zone': 'Defensive Third', 'Num_Sequences': 1}, {'Avg Duration (s)': 5.0, 'Avg Passes': 1.0, 'Counterattack Side': 'Center', 'Loss Zone': 'Middle Third', 'Num_Sequences': 1}], 'types': {'Dispossessed': 1, 'Pass': 1}}, 'empty_defensive': {}, 'empty_offensive': {}, 'offensive': {'flanks': {'Center': 1, 'Left': 1}, 'outcomes': {'Goals': 1, 'Unknown': 1}, 'terminal_outcomes': {'goal': 1, 'unknown': 1}, 'total': 2, 'transition_profile_table': [{'Attack Side': 'Left', 'Avg Duration (s)': 5.0, 'Avg Passes': 1.0, 'Num_Sequences': 1, 'Recovery Zone': 'Attacking Third'}, {'Attack Side': 'Center', 'Avg Duration (s)': 5.0, 'Avg Passes': 1.0, 'Num_Sequences': 1, 'Recovery Zone': 'Middle Third'}], 'types': {'Ball recovery': 1, 'Interception': 1}}}


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


def text(component):
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


def seq(outcome, terminal, loss_type, zone, y, minute):
    return pd.DataFrame([
        {
            "sequence_outcome_type": outcome,
            "terminal_outcome": terminal,
            "type_of_initial_loss": loss_type,
            "loss_zone": zone,
            "y": float(y),
            "timeMin": int(minute),
            "timeSec": 0,
            "type_name": "Ball recovery",
            "team_name": "Attacking Team",
        },
        {
            "sequence_outcome_type": outcome,
            "terminal_outcome": terminal,
            "type_of_initial_loss": loss_type,
            "loss_zone": zone,
            "y": float(y) + 3.0,
            "timeMin": int(minute),
            "timeSec": 5,
            "type_name": "Pass",
            "team_name": "Attacking Team",
        },
    ])


def norm(value):
    if isinstance(value, pd.DataFrame):
        return [norm(x) for x in value.to_dict("records")]
    if isinstance(value, dict):
        return {str(k): norm(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [norm(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def snapshot():
    defensive = [
        seq("Goals conceded", "goal", "Pass", "Defensive Third", 18, 10),
        seq("Unknown", "unknown", "Dispossessed", "Middle Third", 50, 20),
    ]
    offensive = [
        seq("Goals", "goal", "Ball recovery", "Attacking Third", 82, 30),
        seq("Unknown", "unknown", "Interception", "Middle Third", 48, 40),
    ]
    return {
        "defensive": norm(
            transition_metrics.calculate_def_transition_stats(
                defensive, is_away=False
            )
        ),
        "offensive": norm(
            transition_metrics.calculate_off_transition_stats(
                offensive, is_away=False
            )
        ),
        "empty_defensive": norm(
            transition_metrics.calculate_def_transition_stats(
                [], is_away=False
            )
        ),
        "empty_offensive": norm(
            transition_metrics.calculate_off_transition_stats(
                [], is_away=False
            )
        ),
    }


class TransitionSummaryArchitectureTests(unittest.TestCase):
    def test_metrics_has_no_dash_dependencies(self):
        source = (
            ROOT / "src/metrics/transition_metrics.py"
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

        for name in (
            "generate_transition_profile_table",
            "create_def_transition_summary_cards",
            "create_off_transition_summary_cards",
        ):
            self.assertFalse(hasattr(transition_metrics, name))
            self.assertTrue(hasattr(transition_summary_view, name))

    def test_headless_metrics_import_with_dash_blocked(self):
        code = (
            "import importlib.abc, sys\n"
            "class Block(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, fullname, path=None, target=None):\n"
            "        if fullname == 'dash' or fullname.startswith('dash.') "
            "or fullname == 'dash_bootstrap_components' "
            "or fullname.startswith('dash_bootstrap_components.'):\n"
            "            raise ImportError('Dash blocked')\n"
            "        return None\n"
            "sys.meta_path.insert(0, Block())\n"
            "import src.metrics.transition_metrics as tm\n"
            "assert tm.get_pitch_third(10.0) == 'Defensive Third'\n"
            "print('headless-ok')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("headless-ok", result.stdout)

    def test_defensive_offensive_summaries_equal_baseline(self):
        self.assertEqual(snapshot(), BASELINE)

    def test_summary_components_render_data_and_unknown(self):
        defensive = {
            "total": 2,
            "outcomes": {"Goals conceded": 1, "Unknown": 1},
            "terminal_outcomes": {"goal": 1, "unknown": 1},
            "flanks": {"Left": 1, "Central": 1},
            "types": {"Pass": 1, "Unknown": 1},
            "transition_profile_table": pd.DataFrame(),
        }
        offensive = {
            "total": 2,
            "outcomes": {"Goals": 1, "Unknown": 1},
            "terminal_outcomes": {"goal": 1, "unknown": 1},
            "flanks": {"Right": 1, "Central": 1},
            "types": {"Ball recovery": 1, "Unknown": 1},
            "transition_profile_table": pd.DataFrame(),
        }

        d = transition_summary_view.create_def_transition_summary_cards(defensive)
        o = transition_summary_view.create_off_transition_summary_cards(offensive)

        self.assertIsInstance(d, dbc.Row)
        self.assertIsInstance(o, dbc.Row)
        self.assertEqual(len(d.children), 3)
        self.assertEqual(len(o.children), 3)
        self.assertIn("Unknown", text(d))
        self.assertIn("Unknown", text(o))

        d_ids = [
            node.id for node in walk(d)
            if isinstance(getattr(node, "id", None), dict)
        ]
        o_ids = [
            node.id for node in walk(o)
            if isinstance(getattr(node, "id", None), dict)
        ]
        self.assertTrue(any(x.get("type") == "def-filter" for x in d_ids))
        self.assertTrue(any(x.get("type") == "off-filter" for x in o_ids))

    def test_summary_components_render_zero_results(self):
        factories = (
            transition_summary_view.create_def_transition_summary_cards,
            transition_summary_view.create_off_transition_summary_cards,
        )
        for factory in factories:
            for stats in ({}, {"total": 0}):
                with self.subTest(factory=factory.__name__, stats=stats):
                    component = factory(stats)
                    self.assertIsInstance(component, dbc.Alert)
                    self.assertEqual(
                        component.children,
                        "No summary data to display.",
                    )
                    self.assertEqual(component.color, "secondary")

    def test_profile_components_render_data_and_zero(self):
        generic = pd.DataFrame([{"Pattern": "A", "Sequences": 2}])

        legacy = transition_summary_view.generate_transition_profile_table(generic)
        self.assertIsInstance(legacy, dash_table.DataTable)

        renderer = getattr(
            transition_summary_view,
            "render_off_transition_profile_table",
        )
        populated = renderer(generic)
        self.assertFalse(isinstance(populated, dbc.Alert))
        empty = renderer(pd.DataFrame())
        self.assertIsInstance(empty, dbc.Alert)

        legacy_empty = transition_summary_view.generate_transition_profile_table(
            pd.DataFrame()
        )
        self.assertIsInstance(legacy_empty, dbc.Alert)
        self.assertEqual(
            legacy_empty.children,
            "No transition profile data available.",
        )

    def test_app_call_sites_use_transition_summary_view(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn(
            "from src.components import transition_summary_view",
            source,
        )
        for name in (
            "create_def_transition_summary_cards",
            "create_off_transition_summary_cards",
            "render_off_transition_profile_table",
        ):
            self.assertIn(
                f"transition_summary_view.{name}(",
                source,
            )
        tree = ast.parse(source)
        forbidden = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in (
                    "generate_transition_profile_table",
                    "create_def_transition_summary_cards",
                    "create_off_transition_summary_cards",
                )
                and isinstance(func.value, ast.Name)
                and func.value.id == "transition_metrics"
            ):
                forbidden.append(
                    (func.attr, node.lineno)
                )
        self.assertEqual(forbidden, [])


if __name__ == "__main__":
    unittest.main()
