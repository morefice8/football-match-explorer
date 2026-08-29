from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
import unittest

import dash_bootstrap_components as dbc
from dash import html
import pandas as pd

from src.components import restart_summary_view
from src.metrics import set_piece_metrics


ROOT = Path(__file__).resolve().parents[1]
BASELINE_METRIC = {'details': [{'Action Type': 'Corner', 'Delivery': 'Direct Cross', 'Destination': 'Center Box', 'Development Outcome': 'Shot', 'Foot': 'Right', 'Outcome': 'Possession Retained', 'Side': 'Right', 'Swing': 'In-swinger', 'playerName': 'Corner Taker', 'sequence_id': 'seq-1', 'terminal_outcome': 'possession_retained', 'termination_reason': 'controlled_end', 'viewpoint': 'attacking'}, {'Action Type': 'Free Kick', 'Delivery': 'Short Pass', 'Destination': 'N/A', 'Development Outcome': 'Possession Retained', 'Foot': 'Right', 'Outcome': 'Possession Retained', 'Side': 'Right', 'Swing': 'N/A', 'playerName': 'FK Taker', 'sequence_id': 'seq-2', 'terminal_outcome': 'possession_retained', 'termination_reason': 'controlled_end', 'viewpoint': 'attacking'}, {'Action Type': 'Throw-in', 'Delivery': 'Throw-in', 'Destination': 'N/A', 'Development Outcome': 'Possession Retained', 'Foot': 'Right', 'Outcome': 'Possession Retained', 'Side': 'Right', 'Swing': 'N/A', 'playerName': 'Thrower', 'sequence_id': 'seq-3', 'terminal_outcome': 'possession_retained', 'termination_reason': 'controlled_end', 'viewpoint': 'attacking'}, {'Action Type': 'Goal Kick', 'Delivery': 'Goal kick', 'Destination': 'N/A', 'Development Outcome': 'Possession Retained', 'Foot': 'Right', 'Outcome': 'Possession Retained', 'Side': 'Right', 'Swing': 'N/A', 'playerName': 'Keeper', 'sequence_id': 'seq-4', 'terminal_outcome': 'possession_retained', 'termination_reason': 'controlled_end', 'viewpoint': 'attacking'}, {'Action Type': 'Penalty', 'Delivery': 'Penalty kick', 'Destination': 'N/A', 'Development Outcome': 'Goal', 'Foot': 'Right', 'Outcome': 'Penalty Goal', 'Side': 'Center', 'Swing': 'N/A', 'playerName': 'Penalty Taker', 'sequence_id': 'penalty-99', 'terminal_outcome': 'goal', 'termination_reason': 'goal', 'viewpoint': 'attacking'}], 'empty_analyzed': [[], {}], 'empty_summary_table': [], 'stats': {'action_types': {'Corner': 1, 'Free Kick': 1, 'Goal Kick': 1, 'Penalty': 1, 'Throw-in': 1}, 'deliveries': {'Direct Cross': 1, 'Goal kick': 1, 'Penalty kick': 1, 'Short Pass': 1, 'Throw-in': 1}, 'destinations': {'Center Box': 1}, 'development_outcomes': {'Goal': 1, 'Possession Retained': 3, 'Shot': 1}, 'feet': {'Right': 5}, 'outcomes': {'Penalty Goal': 1, 'Possession Retained': 4}, 'sides': {'Center': 1, 'Right': 4}, 'swings': {'In-swinger': 1}, 'terminal_outcomes': {'goal': 1, 'possession_retained': 4}, 'total': 5}, 'summary_table': [{'Action Type': 'Corner', 'Delivery': 'Cross', 'Penalty Goal': 0, 'Possession Retained': 1, 'Swing': 'In-swinger', 'Total': 1}, {'Action Type': 'Free Kick', 'Delivery': 'Short Pass', 'Penalty Goal': 0, 'Possession Retained': 1, 'Swing': 'N/A', 'Total': 1}, {'Action Type': 'Goal Kick', 'Delivery': 'Short Pass', 'Penalty Goal': 0, 'Possession Retained': 1, 'Swing': 'N/A', 'Total': 1}, {'Action Type': 'Penalty', 'Delivery': 'Penalty kick', 'Penalty Goal': 1, 'Possession Retained': 0, 'Swing': 'N/A', 'Total': 1}, {'Action Type': 'Throw-in', 'Delivery': 'Short Pass', 'Penalty Goal': 0, 'Possession Retained': 1, 'Swing': 'N/A', 'Total': 1}]}
BASELINE_UI = {'summary': {'namespace': 'dash_html_components', 'props': {'children': [{'namespace': 'dash_bootstrap_components', 'props': {'children': [{'namespace': 'dash_bootstrap_components', 'props': {'children': {'namespace': 'dash_bootstrap_components', 'props': {'children': [{'namespace': 'dash_bootstrap_components', 'props': {'children': 'Action Type'}, 'type': 'CardHeader'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': [{'namespace': 'dash_bootstrap_components', 'props': {'action': True, 'active': True, 'children': [{'namespace': 'dash_html_components', 'props': {'children': 'Corner'}, 'type': 'Div'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': '2', 'className': 'ms-auto', 'color': 'light'}, 'type': 'Badge'}], 'className': 'd-flex justify-content-between align-items-center', 'id': {'filter_type': 'action', 'type': 'sp-filter', 'value': 'Corner'}, 'n_clicks': 0}, 'type': 'ListGroupItem'}, {'namespace': 'dash_bootstrap_components', 'props': {'action': True, 'active': False, 'children': [{'namespace': 'dash_html_components', 'props': {'children': 'Free Kick'}, 'type': 'Div'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': '1', 'className': 'ms-auto', 'color': 'light'}, 'type': 'Badge'}], 'className': 'd-flex justify-content-between align-items-center', 'id': {'filter_type': 'action', 'type': 'sp-filter', 'value': 'Free Kick'}, 'n_clicks': 0}, 'type': 'ListGroupItem'}, {'namespace': 'dash_bootstrap_components', 'props': {'action': True, 'active': False, 'children': [{'namespace': 'dash_html_components', 'props': {'children': 'Throw-in'}, 'type': 'Div'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': '1', 'className': 'ms-auto', 'color': 'light'}, 'type': 'Badge'}], 'className': 'd-flex justify-content-between align-items-center', 'id': {'filter_type': 'action', 'type': 'sp-filter', 'value': 'Throw-in'}, 'n_clicks': 0}, 'type': 'ListGroupItem'}, {'namespace': 'dash_bootstrap_components', 'props': {'action': True, 'active': False, 'children': [{'namespace': 'dash_html_components', 'props': {'children': 'Goal Kick'}, 'type': 'Div'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': '1', 'className': 'ms-auto', 'color': 'light'}, 'type': 'Badge'}], 'className': 'd-flex justify-content-between align-items-center', 'id': {'filter_type': 'action', 'type': 'sp-filter', 'value': 'Goal Kick'}, 'n_clicks': 0}, 'type': 'ListGroupItem'}], 'flush': True}, 'type': 'ListGroup'}]}, 'type': 'Card'}, 'md': 4}, 'type': 'Col'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': {'namespace': 'dash_bootstrap_components', 'props': {'children': [{'namespace': 'dash_bootstrap_components', 'props': {'children': 'Action Side'}, 'type': 'CardHeader'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': [{'namespace': 'dash_bootstrap_components', 'props': {'action': True, 'active': False, 'children': [{'namespace': 'dash_html_components', 'props': {'children': 'Left'}, 'type': 'Div'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': '2', 'className': 'ms-auto', 'color': 'light'}, 'type': 'Badge'}], 'className': 'd-flex justify-content-between align-items-center', 'id': {'filter_type': 'side', 'type': 'sp-filter', 'value': 'Left'}, 'n_clicks': 0}, 'type': 'ListGroupItem'}, {'namespace': 'dash_bootstrap_components', 'props': {'action': True, 'active': False, 'children': [{'namespace': 'dash_html_components', 'props': {'children': 'Right'}, 'type': 'Div'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': '2', 'className': 'ms-auto', 'color': 'light'}, 'type': 'Badge'}], 'className': 'd-flex justify-content-between align-items-center', 'id': {'filter_type': 'side', 'type': 'sp-filter', 'value': 'Right'}, 'n_clicks': 0}, 'type': 'ListGroupItem'}, {'namespace': 'dash_bootstrap_components', 'props': {'action': True, 'active': False, 'children': [{'namespace': 'dash_html_components', 'props': {'children': 'Center'}, 'type': 'Div'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': '1', 'className': 'ms-auto', 'color': 'light'}, 'type': 'Badge'}], 'className': 'd-flex justify-content-between align-items-center', 'id': {'filter_type': 'side', 'type': 'sp-filter', 'value': 'Center'}, 'n_clicks': 0}, 'type': 'ListGroupItem'}], 'flush': True}, 'type': 'ListGroup'}]}, 'type': 'Card'}, 'md': 4}, 'type': 'Col'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': {'namespace': 'dash_bootstrap_components', 'props': {'children': [{'namespace': 'dash_bootstrap_components', 'props': {'children': 'Delivery Type'}, 'type': 'CardHeader'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': [{'namespace': 'dash_bootstrap_components', 'props': {'action': True, 'active': False, 'children': [{'namespace': 'dash_html_components', 'props': {'children': 'Direct Cross'}, 'type': 'Div'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': '2', 'className': 'ms-auto', 'color': 'light'}, 'type': 'Badge'}], 'className': 'd-flex justify-content-between align-items-center', 'id': {'filter_type': 'delivery', 'type': 'sp-filter', 'value': 'Direct Cross'}, 'n_clicks': 0}, 'type': 'ListGroupItem'}, {'namespace': 'dash_bootstrap_components', 'props': {'action': True, 'active': False, 'children': [{'namespace': 'dash_html_components', 'props': {'children': 'Short Pass'}, 'type': 'Div'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': '1', 'className': 'ms-auto', 'color': 'light'}, 'type': 'Badge'}], 'className': 'd-flex justify-content-between align-items-center', 'id': {'filter_type': 'delivery', 'type': 'sp-filter', 'value': 'Short Pass'}, 'n_clicks': 0}, 'type': 'ListGroupItem'}, {'namespace': 'dash_bootstrap_components', 'props': {'action': True, 'active': False, 'children': [{'namespace': 'dash_html_components', 'props': {'children': 'Throw-in'}, 'type': 'Div'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': '1', 'className': 'ms-auto', 'color': 'light'}, 'type': 'Badge'}], 'className': 'd-flex justify-content-between align-items-center', 'id': {'filter_type': 'delivery', 'type': 'sp-filter', 'value': 'Throw-in'}, 'n_clicks': 0}, 'type': 'ListGroupItem'}, {'namespace': 'dash_bootstrap_components', 'props': {'action': True, 'active': False, 'children': [{'namespace': 'dash_html_components', 'props': {'children': 'Goal kick'}, 'type': 'Div'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': '1', 'className': 'ms-auto', 'color': 'light'}, 'type': 'Badge'}], 'className': 'd-flex justify-content-between align-items-center', 'id': {'filter_type': 'delivery', 'type': 'sp-filter', 'value': 'Goal kick'}, 'n_clicks': 0}, 'type': 'ListGroupItem'}], 'flush': True}, 'type': 'ListGroup'}]}, 'type': 'Card'}, 'md': 4}, 'type': 'Col'}], 'className': 'mb-3'}, 'type': 'Row'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': [{'namespace': 'dash_bootstrap_components', 'props': {'children': {'namespace': 'dash_bootstrap_components', 'props': {'children': [{'namespace': 'dash_bootstrap_components', 'props': {'children': 'Cross Swing'}, 'type': 'CardHeader'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': [{'namespace': 'dash_bootstrap_components', 'props': {'action': True, 'active': False, 'children': [{'namespace': 'dash_html_components', 'props': {'children': 'In-swinger'}, 'type': 'Div'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': '2', 'className': 'ms-auto', 'color': 'light'}, 'type': 'Badge'}], 'className': 'd-flex justify-content-between align-items-center', 'id': {'filter_type': 'swing', 'type': 'sp-filter', 'value': 'In-swinger'}, 'n_clicks': 0}, 'type': 'ListGroupItem'}], 'flush': True}, 'type': 'ListGroup'}]}, 'type': 'Card'}, 'md': 4}, 'type': 'Col'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': {'namespace': 'dash_bootstrap_components', 'props': {'children': [{'namespace': 'dash_bootstrap_components', 'props': {'children': 'Taker Foot'}, 'type': 'CardHeader'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': [{'namespace': 'dash_bootstrap_components', 'props': {'action': True, 'active': False, 'children': [{'namespace': 'dash_html_components', 'props': {'children': 'Right'}, 'type': 'Div'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': '4', 'className': 'ms-auto', 'color': 'light'}, 'type': 'Badge'}], 'className': 'd-flex justify-content-between align-items-center', 'id': {'filter_type': 'foot', 'type': 'sp-filter', 'value': 'Right'}, 'n_clicks': 0}, 'type': 'ListGroupItem'}, {'namespace': 'dash_bootstrap_components', 'props': {'action': True, 'active': False, 'children': [{'namespace': 'dash_html_components', 'props': {'children': 'Left'}, 'type': 'Div'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': '1', 'className': 'ms-auto', 'color': 'light'}, 'type': 'Badge'}], 'className': 'd-flex justify-content-between align-items-center', 'id': {'filter_type': 'foot', 'type': 'sp-filter', 'value': 'Left'}, 'n_clicks': 0}, 'type': 'ListGroupItem'}], 'flush': True}, 'type': 'ListGroup'}]}, 'type': 'Card'}, 'md': 4}, 'type': 'Col'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': {'namespace': 'dash_bootstrap_components', 'props': {'children': [{'namespace': 'dash_bootstrap_components', 'props': {'children': 'Corner Cross Destination'}, 'type': 'CardHeader'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': [{'namespace': 'dash_bootstrap_components', 'props': {'action': True, 'active': False, 'children': [{'namespace': 'dash_html_components', 'props': {'children': 'Near Post'}, 'type': 'Div'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': '1', 'className': 'ms-auto', 'color': 'light'}, 'type': 'Badge'}], 'className': 'd-flex justify-content-between align-items-center', 'id': {'filter_type': 'destination', 'type': 'sp-filter', 'value': 'Near Post'}, 'n_clicks': 0}, 'type': 'ListGroupItem'}, {'namespace': 'dash_bootstrap_components', 'props': {'action': True, 'active': False, 'children': [{'namespace': 'dash_html_components', 'props': {'children': 'Far Post'}, 'type': 'Div'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': '1', 'className': 'ms-auto', 'color': 'light'}, 'type': 'Badge'}], 'className': 'd-flex justify-content-between align-items-center', 'id': {'filter_type': 'destination', 'type': 'sp-filter', 'value': 'Far Post'}, 'n_clicks': 0}, 'type': 'ListGroupItem'}], 'flush': True}, 'type': 'ListGroup'}]}, 'type': 'Card'}, 'md': 4}, 'type': 'Col'}], 'className': 'mb-3'}, 'type': 'Row'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': [{'namespace': 'dash_bootstrap_components', 'props': {'children': {'namespace': 'dash_bootstrap_components', 'props': {'children': [{'namespace': 'dash_bootstrap_components', 'props': {'children': 'Execution Outcome'}, 'type': 'CardHeader'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': [{'namespace': 'dash_bootstrap_components', 'props': {'action': True, 'active': False, 'children': [{'namespace': 'dash_html_components', 'props': {'children': 'Goals'}, 'type': 'Div'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': '1', 'className': 'ms-auto', 'color': 'light'}, 'type': 'Badge'}], 'className': 'd-flex justify-content-between align-items-center', 'id': {'filter_type': 'outcome', 'type': 'sp-filter', 'value': 'Goals'}, 'n_clicks': 0}, 'type': 'ListGroupItem'}, {'namespace': 'dash_bootstrap_components', 'props': {'action': True, 'active': False, 'children': [{'namespace': 'dash_html_components', 'props': {'children': 'Shots'}, 'type': 'Div'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': '2', 'className': 'ms-auto', 'color': 'light'}, 'type': 'Badge'}], 'className': 'd-flex justify-content-between align-items-center', 'id': {'filter_type': 'outcome', 'type': 'sp-filter', 'value': 'Shots'}, 'n_clicks': 0}, 'type': 'ListGroupItem'}, {'namespace': 'dash_bootstrap_components', 'props': {'action': True, 'active': False, 'children': [{'namespace': 'dash_html_components', 'props': {'children': 'Lost Possessions'}, 'type': 'Div'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': '2', 'className': 'ms-auto', 'color': 'light'}, 'type': 'Badge'}], 'className': 'd-flex justify-content-between align-items-center', 'id': {'filter_type': 'outcome', 'type': 'sp-filter', 'value': 'Lost Possessions'}, 'n_clicks': 0}, 'type': 'ListGroupItem'}], 'flush': True}, 'type': 'ListGroup'}]}, 'type': 'Card'}, 'md': 4}, 'type': 'Col'}], 'className': 'mb-3'}, 'type': 'Row'}]}, 'type': 'Div'}, 'summary_empty': {'namespace': 'dash_bootstrap_components', 'props': {'children': 'No restart data to display.', 'color': 'secondary'}, 'type': 'Alert'}, 'takers': {'namespace': 'dash_bootstrap_components', 'props': {'children': {'namespace': 'dash_bootstrap_components', 'props': {'children': [{'namespace': 'dash_bootstrap_components', 'props': {'children': 'Set-piece Takers'}, 'type': 'CardHeader'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': [{'namespace': 'dash_bootstrap_components', 'props': {'action': True, 'active': True, 'children': [{'namespace': 'dash_html_components', 'props': {'children': [{'namespace': 'dash_html_components', 'props': {'children': '#8', 'className': 'fw-bold me-2', 'style': {'display': 'inline-block', 'minWidth': '35px'}}, 'type': 'Span'}, {'namespace': 'dash_html_components', 'props': {'children': 'Alice'}, 'type': 'Span'}]}, 'type': 'Div'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': 'Right | ✖ 2', 'className': 'ms-auto', 'color': 'light', 'pill': True}, 'type': 'Badge'}], 'className': 'd-flex justify-content-between align-items-center', 'id': {'filter_type': 'taker', 'type': 'sp-filter', 'value': 'Alice'}, 'n_clicks': 0}, 'type': 'ListGroupItem'}, {'namespace': 'dash_bootstrap_components', 'props': {'action': True, 'active': False, 'children': [{'namespace': 'dash_html_components', 'props': {'children': [{'namespace': 'dash_html_components', 'props': {'children': '#10', 'className': 'fw-bold me-2', 'style': {'display': 'inline-block', 'minWidth': '35px'}}, 'type': 'Span'}, {'namespace': 'dash_html_components', 'props': {'children': 'Bob'}, 'type': 'Span'}]}, 'type': 'Div'}, {'namespace': 'dash_bootstrap_components', 'props': {'children': 'Left | ✖ 1', 'className': 'ms-auto', 'color': 'light', 'pill': True}, 'type': 'Badge'}], 'className': 'd-flex justify-content-between align-items-center', 'id': {'filter_type': 'taker', 'type': 'sp-filter', 'value': 'Bob'}, 'n_clicks': 0}, 'type': 'ListGroupItem'}], 'flush': True}, 'type': 'ListGroup'}]}, 'type': 'Card'}, 'md': 4}, 'type': 'Col'}, 'takers_empty': None, 'takers_none': None}


def encode(value):
    if hasattr(value, "to_plotly_json"):
        return encode(value.to_plotly_json())
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return value


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


def regular(trigger, execution, development, player, event_id, cross=0):
    return pd.DataFrame([{
        "trigger_sequence_id": f"seq-{event_id}",
        "type_of_initial_trigger": trigger,
        "type_name": "Pass",
        "eventId": event_id,
        "playerName": player,
        "x": 60.0,
        "y": 20.0,
        "end_x": 90.0,
        "end_y": 40.0,
        "cross": cross,
        "Right footed": 1,
        "Left footed": 0,
        "In-swinger": 1 if cross else 0,
        "Out-swinger": 0,
        "Straight": 0,
        "sequence_outcome_type": execution,
        "terminal_outcome": "possession_retained",
        "termination_reason": "controlled_end",
        "viewpoint": "attacking",
        "restart_execution_outcome": execution,
        "restart_development_outcome": development,
    }])


class RestartSummaryArchitectureTests(unittest.TestCase):
    def test_set_piece_metrics_has_no_dash_imports(self):
        source = (
            ROOT / "src/metrics/set_piece_metrics.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)

        forbidden = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
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
            hasattr(set_piece_metrics, "create_set_piece_summary_cards")
        )
        self.assertFalse(
            hasattr(set_piece_metrics, "create_takers_card")
        )
        self.assertTrue(
            hasattr(restart_summary_view, "create_set_piece_summary_cards")
        )
        self.assertTrue(
            hasattr(restart_summary_view, "create_takers_card")
        )

    def test_neutral_tables_and_summaries_equal_baseline(self):
        self.assertEqual(
            json.loads(subprocess.run(
            [sys.executable, "-c", '\nimport json\nimport numpy as np\nimport pandas as pd\nfrom src.metrics import set_piece_metrics as m\n\n\ndef regular(trigger, execution, development, player, event_id, cross=0):\n    return pd.DataFrame([{\n        "trigger_sequence_id": f"seq-{event_id}",\n        "type_of_initial_trigger": trigger,\n        "type_name": "Pass",\n        "eventId": event_id,\n        "playerName": player,\n        "x": 60.0,\n        "y": 20.0,\n        "end_x": 90.0,\n        "end_y": 40.0,\n        "cross": cross,\n        "Right footed": 1,\n        "Left footed": 0,\n        "In-swinger": 1 if cross else 0,\n        "Out-swinger": 0,\n        "Straight": 0,\n        "sequence_outcome_type": execution,\n        "terminal_outcome": "possession_retained",\n        "termination_reason": "controlled_end",\n        "viewpoint": "attacking",\n        "restart_execution_outcome": execution,\n        "restart_development_outcome": development,\n    }])\n\n\ndef penalty(event_id=99):\n    return pd.DataFrame([{\n        "trigger_sequence_id": f"penalty-{event_id}",\n        "type_of_initial_trigger": "Penalty",\n        "type_name": "Goal",\n        "Penalty": 1,\n        "eventId": event_id,\n        "id": event_id,\n        "playerName": "Penalty Taker",\n        "x": 88.0,\n        "y": 50.0,\n        "end_x": 100.0,\n        "end_y": 50.0,\n        "cross": 0,\n        "Right footed": 1,\n        "Left footed": 0,\n        "sequence_outcome_type": "Penalty Goal",\n        "terminal_outcome": "goal",\n        "termination_reason": "goal",\n        "viewpoint": "attacking",\n        "restart_execution_outcome": "Penalty Goal",\n        "restart_development_outcome": "Goal",\n    }])\n\n\ndef norm(value):\n    if isinstance(value, pd.DataFrame):\n        return [norm(x) for x in value.to_dict("records")]\n    if isinstance(value, dict):\n        return {str(k): norm(v) for k, v in value.items()}\n    if isinstance(value, (list, tuple)):\n        return [norm(v) for v in value]\n    if isinstance(value, np.integer):\n        return int(value)\n    if isinstance(value, np.floating):\n        if np.isnan(value):\n            return None\n        return float(value)\n    if value is pd.NA:\n        return None\n    return value\n\n\nsequences = [\n    regular("Corner", "Possession Retained", "Shot", "Corner Taker", 1, cross=1),\n    regular("Free Kick", "Possession Retained", "Possession Retained", "FK Taker", 2),\n    regular("Throw-in", "Possession Retained", "Possession Retained", "Thrower", 3),\n    regular("Goal Kick", "Possession Retained", "Possession Retained", "Keeper", 4),\n    penalty(),\n]\n\nsummary_table = m.calculate_set_piece_stats(sequences)\ndetails, stats = m.analyze_and_summarize_set_pieces(sequences)\n\npayload = {\n    "summary_table": norm(summary_table),\n    "details": norm(details),\n    "stats": norm(stats),\n    "empty_summary_table": norm(m.calculate_set_piece_stats([])),\n    "empty_analyzed": norm(m.analyze_and_summarize_set_pieces([])),\n}\n\nprint("ARCH04C_METRIC=" + json.dumps(\n    payload,\n    sort_keys=True,\n    ensure_ascii=True,\n))\n'],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        ).stdout.split("ARCH04C_METRIC=", 1)[1].strip()),
            BASELINE_METRIC,
        )

    def test_rendering_equal_baseline(self):
        stats = {
            "total": 5,
            "action_types": {
                "Corner": 2,
                "Free Kick": 1,
                "Throw-in": 1,
                "Goal Kick": 1,
            },
            "sides": {"Left": 2, "Right": 2, "Center": 1},
            "deliveries": {
                "Direct Cross": 2,
                "Short Pass": 1,
                "Throw-in": 1,
                "Goal kick": 1,
            },
            "swings": {"In-swinger": 2},
            "feet": {"Right": 4, "Left": 1},
            "destinations": {"Near Post": 1, "Far Post": 1},
            "outcomes": {
                "Goals": 1,
                "Shots": 2,
                "Lost Possessions": 2,
            },
        }
        takers = pd.DataFrame([
            {"Action Type": "Corner", "playerName": "Alice", "Foot": "Right"},
            {"Action Type": "Corner", "playerName": "Alice", "Foot": "Right"},
            {"Action Type": "Free Kick", "playerName": "Bob", "Foot": "Left"},
            {"Action Type": "Throw-in", "playerName": "Thrower", "Foot": "Right"},
            {"Action Type": "Goal Kick", "playerName": "Keeper", "Foot": "Right"},
        ])
        no_takers = pd.DataFrame([
            {"Action Type": "Throw-in", "playerName": "Thrower", "Foot": "Right"},
            {"Action Type": "Goal Kick", "playerName": "Keeper", "Foot": "Right"},
        ])
        jerseys = {"Alice": 8, "Bob": 10, "Thrower": 2, "Keeper": 1}

        current = {
            "summary": encode(
                restart_summary_view.create_set_piece_summary_cards(
                    stats,
                    active_filter={"action": "Corner"},
                )
            ),
            "summary_empty": encode(
                restart_summary_view.create_set_piece_summary_cards({})
            ),
            "takers": encode(
                restart_summary_view.create_takers_card(
                    takers,
                    jerseys,
                    active_filter={"taker": "Alice"},
                )
            ),
            "takers_empty": encode(
                restart_summary_view.create_takers_card(
                    pd.DataFrame(),
                    jerseys,
                )
            ),
            "takers_none": encode(
                restart_summary_view.create_takers_card(
                    no_takers,
                    jerseys,
                )
            ),
        }
        self.assertEqual(current, BASELINE_UI)

    def test_rendering_normal_empty_and_without_taker(self):
        stats = {
            "total": 1,
            "action_types": {"Corner": 1},
            "sides": {"Left": 1},
            "deliveries": {"Direct Cross": 1},
            "swings": {"In-swinger": 1},
            "feet": {"Right": 1},
            "destinations": {"Near Post": 1},
            "outcomes": {"Shots": 1},
        }
        component = restart_summary_view.create_set_piece_summary_cards(stats)
        self.assertIsInstance(component, html.Div)

        empty = restart_summary_view.create_set_piece_summary_cards({})
        self.assertIsInstance(empty, dbc.Alert)
        self.assertEqual(empty.children, "No restart data to display.")

        no_taker_df = pd.DataFrame([
            {
                "Action Type": "Throw-in",
                "playerName": "Thrower",
                "Foot": "Right",
            },
            {
                "Action Type": "Goal Kick",
                "playerName": "Keeper",
                "Foot": "Right",
            },
        ])
        self.assertIsNone(
            restart_summary_view.create_takers_card(
                no_taker_df,
                {"Thrower": 2, "Keeper": 1},
            )
        )
        self.assertIsNone(
            restart_summary_view.create_takers_card(
                pd.DataFrame(),
                {},
            )
        )

    def test_taker_ranking_and_ids_are_preserved(self):
        df = pd.DataFrame([
            {"Action Type": "Corner", "playerName": "Alice", "Foot": "Right"},
            {"Action Type": "Corner", "playerName": "Alice", "Foot": "Right"},
            {"Action Type": "Free Kick", "playerName": "Bob", "Foot": "Left"},
            {"Action Type": "Throw-in", "playerName": "Ignored", "Foot": "Right"},
        ])
        card = restart_summary_view.create_takers_card(
            df,
            {"Alice": 8, "Bob": 10, "Ignored": 2},
        )
        self.assertIsInstance(card, dbc.Col)

        ids = [
            node.id
            for node in walk(card)
            if isinstance(getattr(node, "id", None), dict)
        ]
        taker_ids = [
            value
            for value in ids
            if value.get("type") == "sp-filter"
            and value.get("filter_type") == "taker"
        ]
        self.assertEqual(
            [item["value"] for item in taker_ids],
            ["Alice", "Bob"],
        )

    def test_taxonomy_execution_and_development_outcomes(self):
        sequences = [
            regular(
                "Corner",
                "Possession Retained",
                "Shot",
                "Corner Taker",
                1,
                cross=1,
            ),
            regular(
                "Free Kick",
                "Possession Retained",
                "Possession Retained",
                "FK Taker",
                2,
            ),
            regular(
                "Throw-in",
                "Possession Retained",
                "Possession Retained",
                "Thrower",
                3,
            ),
            regular(
                "Goal Kick",
                "Possession Retained",
                "Possession Retained",
                "Keeper",
                4,
            ),
        ]

        df, stats = set_piece_metrics.analyze_and_summarize_set_pieces(
            sequences
        )
        self.assertEqual(
            set(df["Action Type"]),
            {"Corner", "Free Kick", "Throw-in", "Goal Kick"},
        )

        corner = df.loc[df["Action Type"] == "Corner"].iloc[0]
        self.assertEqual(corner["Outcome"], "Possession Retained")
        self.assertEqual(corner["Development Outcome"], "Shot")

    def test_penalty_outcomes_are_preserved(self):
        source = pd.DataFrame([
            {
                "team_name": "Napoli",
                "type_name": "Goal",
                "Penalty": 1,
                "id": 101,
                "eventId": 101,
                "timeMin": 10,
                "timeSec": 0,
                "playerName": "A",
                "x": 88.0,
                "y": 50.0,
                "end_x": 100.0,
                "end_y": 50.0,
                "cross": 0,
            },
            {
                "team_name": "Napoli",
                "type_name": "Attempt Saved",
                "Penalty": 1,
                "id": 102,
                "eventId": 102,
                "timeMin": 20,
                "timeSec": 0,
                "playerName": "B",
                "x": 88.0,
                "y": 50.0,
                "end_x": 100.0,
                "end_y": 50.0,
                "cross": 0,
            },
            {
                "team_name": "Napoli",
                "type_name": "Miss",
                "Penalty": 1,
                "id": 103,
                "eventId": 103,
                "timeMin": 30,
                "timeSec": 0,
                "playerName": "C",
                "x": 88.0,
                "y": 50.0,
                "end_x": 100.0,
                "end_y": 50.0,
                "cross": 0,
            },
        ])

        sequences = set_piece_metrics.extract_penalty_set_piece_sequences(
            source,
            "Napoli",
        )
        self.assertEqual(len(sequences), 3)
        self.assertEqual(
            [
                seq.iloc[0]["sequence_outcome_type"]
                for seq in sequences
            ],
            ["Penalty Goal", "Penalty Saved", "Penalty Missed"],
        )

        analyzed, stats = (
            set_piece_metrics.analyze_and_summarize_set_pieces(sequences)
        )
        self.assertEqual(set(analyzed["Action Type"]), {"Penalty"})
        self.assertEqual(
            stats["outcomes"],
            {
                "Penalty Goal": 1,
                "Penalty Saved": 1,
                "Penalty Missed": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
