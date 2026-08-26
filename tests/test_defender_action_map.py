import ast
import unittest
from pathlib import Path

import pandas as pd

from src.components import defender_action_map_view
from src.metrics import defensive_contribution_metrics
from src.visualization import defender_action_map
from src.visualization.plotly_branding import MATCH_COMPARE_HEIGHT, MATCH_PITCH_BG


def defensive_events_frame():
    return pd.DataFrame([
        {
            "id": 1,
            "typeId": 7,
            "type_name": "Tackle",
            "outcome": "Successful",
            "playerName": "Defender",
            "team_name": "Home",
            "Mapped Jersey Number": 4,
            "x": 28.0,
            "y": 44.0,
            "timeMin": 8,
            "timeSec": 2,
        },
        {
            "id": 2,
            "typeId": 7,
            "type_name": "Tackle",
            "outcome": "Unsuccessful",
            "playerName": "Defender",
            "team_name": "Home",
            "Mapped Jersey Number": 4,
            "x": 38.0,
            "y": 52.0,
            "timeMin": 19,
            "timeSec": 9,
        },
        {
            "id": 3,
            "typeId": 8,
            "type_name": "Interception",
            "outcome": "Successful",
            "playerName": "Defender",
            "team_name": "Home",
            "Mapped Jersey Number": 4,
            "x": 47.0,
            "y": 39.0,
            "timeMin": 25,
            "timeSec": 0,
        },
        {
            "id": 4,
            "typeId": 49,
            "type_name": "Ball recovery",
            "outcome": "Successful",
            "playerName": "Defender",
            "team_name": "Home",
            "Mapped Jersey Number": 4,
            "x": 57.0,
            "y": 63.0,
            "timeMin": 31,
            "timeSec": 5,
        },
        {
            "id": 5,
            "typeId": 12,
            "type_name": "Clearance",
            "outcome": "Successful",
            "playerName": "Defender",
            "team_name": "Home",
            "Mapped Jersey Number": 4,
            "x": 18.0,
            "y": 66.0,
            "timeMin": 43,
            "timeSec": 13,
        },
        {
            "id": 6,
            "typeId": 74,
            "type_name": "Blocked pass",
            "outcome": "Successful",
            "playerName": "Defender",
            "team_name": "Home",
            "Mapped Jersey Number": 4,
            "x": 34.0,
            "y": 72.0,
            "timeMin": 49,
            "timeSec": 44,
        },
        {
            "id": 7,
            "typeId": 10,
            "type_name": "Save",
            "outcome": "Successful",
            "Def block": 1,
            "playerName": "Defender",
            "team_name": "Home",
            "Mapped Jersey Number": 4,
            "x": 13.0,
            "y": 48.0,
            "timeMin": 62,
            "timeSec": 30,
        },
        {
            "id": 8,
            "typeId": 4,
            "type_name": "Foul",
            "outcome": "Unsuccessful",
            "playerName": "Defender",
            "team_name": "Home",
            "Mapped Jersey Number": 4,
            "x": 51.0,
            "y": 28.0,
            "timeMin": 70,
            "timeSec": 4,
        },
        {
            "id": 9,
            "typeId": 4,
            "type_name": "Foul",
            "outcome": "Successful",
            "playerName": "Defender",
            "team_name": "Home",
            "Mapped Jersey Number": 4,
            "x": 60.0,
            "y": 20.0,
            "timeMin": 72,
            "timeSec": 1,
        },
    ])


class DefenderActionMetricTests(unittest.TestCase):
    def test_map_classifier_uses_same_six_action_contract(self):
        classified = defensive_contribution_metrics.classify_defensive_events(
            defensive_events_frame()
        )

        self.assertEqual(len(classified), 8)
        self.assertEqual(
            classified["defensive_category"].tolist(),
            [
                "Tackle",
                "Tackle",
                "Interception",
                "Recovery",
                "Clearance",
                "Block",
                "Block",
                "Foul",
            ],
        )
        self.assertEqual(
            classified.loc[
                classified["defensive_category"].eq("Tackle"),
                "tackle_won",
            ].tolist(),
            [True, False],
        )
        self.assertNotIn(9, classified["id"].tolist())

    def test_player_options_are_ranked_by_canonical_unique_actions(self):
        frame = pd.concat([
            defensive_events_frame(),
            pd.DataFrame([
                {
                    "id": 20,
                    "type_name": "Clearance",
                    "outcome": "Successful",
                    "playerName": "Other",
                    "team_name": "Home",
                    "Mapped Jersey Number": 5,
                    "x": 20,
                    "y": 20,
                }
            ]),
        ], ignore_index=True)

        options = defensive_contribution_metrics.team_defensive_player_options(
            frame,
            "Home",
        )

        self.assertEqual(options[0]["player_name"], "Defender")
        self.assertEqual(options[0]["jersey"], "4")


class DefenderActionPlotTests(unittest.TestCase):
    def test_plot_uses_distinct_action_shapes_and_tackle_state(self):
        classified = defensive_contribution_metrics.classify_defensive_events(
            defensive_events_frame()
        )
        fig = defender_action_map.plot_defensive_action_profile(
            classified,
            selected_player="Defender",
            jersey="4",
            team_color="#e96a4a",
        )

        traces = {
            trace.name: trace
            for trace in fig.data
            if trace.showlegend is not False
        }

        self.assertEqual(
            list(traces),
            [
                "Tackle",
                "Interception",
                "Recovery",
                "Clearance",
                "Block",
                "Foul",
            ],
        )
        tackle_traces = [
            trace
            for trace in fig.data
            if trace.name == "Tackle"
        ]
        self.assertEqual(
            [trace.marker.symbol for trace in tackle_traces],
            ["circle", "circle-open"],
        )
        self.assertEqual(traces["Interception"].marker.symbol, "diamond")
        self.assertEqual(traces["Recovery"].marker.symbol, "hexagon")
        self.assertEqual(traces["Clearance"].marker.symbol, "triangle-up")
        self.assertEqual(traces["Block"].marker.symbol, "square")
        self.assertEqual(traces["Foul"].marker.symbol, "x")

    def test_plot_matches_player_analysis_pitch_contract(self):
        classified = defensive_contribution_metrics.classify_defensive_events(
            defensive_events_frame()
        )
        fig = defender_action_map.plot_defensive_action_profile(
            classified,
            selected_player="Defender",
            jersey="4",
            team_color="#1597c2",
            is_away=True,
        )

        self.assertEqual(fig.layout.plot_bgcolor, MATCH_PITCH_BG)
        self.assertEqual(fig.layout.paper_bgcolor, MATCH_PITCH_BG)
        self.assertEqual(fig.layout.height, MATCH_COMPARE_HEIGHT)
        self.assertIsNone(fig.layout.title.text)
        self.assertEqual(list(fig.layout.xaxis.range), [0, 100])
        self.assertEqual(list(fig.layout.yaxis.range), [0, 100])
        self.assertTrue(
            any(
                "ATTACK" in str(annotation.text).upper()
                for annotation in fig.layout.annotations
            )
        )

        interception = next(
            trace for trace in fig.data
            if trace.name == "Interception"
        )
        self.assertEqual(list(interception.x), [47.0])
        self.assertEqual(list(interception.y), [39.0])

    def test_plot_adds_typical_defensive_zone_behind_actions(self):
        classified = defensive_contribution_metrics.classify_defensive_events(
            defensive_events_frame()
        )
        fig = defender_action_map.plot_defensive_action_profile(
            classified,
            selected_player="Defender",
            jersey="4",
            team_color="#1597c2",
        )

        zone = next(
            trace for trace in fig.data
            if trace.name == "Typical defensive zone"
        )

        self.assertFalse(zone.showlegend)
        self.assertEqual(zone.fill, "toself")
        self.assertEqual(fig.data[0].name, "Typical defensive zone")
        self.assertLess(max(zone.x) - min(zone.x), 30)
        self.assertLess(max(zone.y) - min(zone.y), 30)

    def test_numbered_marker_keeps_median_position_feature(self):
        classified = defensive_contribution_metrics.classify_defensive_events(
            defensive_events_frame()
        )
        fig = defender_action_map.plot_defensive_action_profile(
            classified,
            selected_player="Defender",
            jersey="4",
            team_color="#e96a4a",
        )

        median = next(
            trace for trace in fig.data
            if trace.name == "Median defensive-action position"
        )
        self.assertEqual(list(median.text), ["4"])
        self.assertEqual(list(median.x), [34.0])
        self.assertEqual(list(median.y), [52.0])
        self.assertFalse(median.showlegend)


class DefenderActionViewTests(unittest.TestCase):
    def test_player_panel_uses_rebranded_copy_and_kpis(self):
        classified = defensive_contribution_metrics.classify_defensive_events(
            defensive_events_frame()
        )
        profile = defensive_contribution_metrics.player_defensive_profile(
            defensive_events_frame()
        )
        fig = defender_action_map.plot_defensive_action_profile(
            classified,
            selected_player="Defender",
            jersey="4",
            team_color="#e96a4a",
        )

        component = defender_action_map_view.player_panel(
            selected_player="Defender",
            jersey="4",
            team_name="Home",
            team_color="#e96a4a",
            profile=profile,
            figure=fig,
        )
        text = str(component)

        self.assertIn("DEFENSIVE ACTION PROFILE", text)
        self.assertIn("Successful / attempted", text)
        self.assertIn("Fouls committed", text)
        self.assertIn("shaded ellipse", text)
        self.assertNotIn("Action Summary", text)
        self.assertNotIn("Analyst Comments", text)


class DefenderActionAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("app.py").read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def _function_source(self, name):
        lines = self.source.splitlines(keepends=True)
        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                start = min(
                    [node.lineno]
                    + [decorator.lineno for decorator in node.decorator_list]
                )
                return "".join(lines[start - 1:node.end_lineno])
        self.fail(f"Missing app function {name}")

    def test_defender_router_no_longer_calls_legacy_generator(self):
        block = self._function_source("render_defending_analysis_content")
        self.assertIn("create_defender_action_layout", block)
        self.assertNotIn("generate_defender_layout_and_data", block)

    def test_defender_callbacks_use_new_panel(self):
        home = self._function_source("update_home_defender_view")
        away = self._function_source("update_away_defender_view")
        self.assertIn("_build_defender_action_panel", home)
        self.assertIn("_build_defender_action_panel", away)
        self.assertNotIn("generate_defender_layout_and_data", home)
        self.assertNotIn("generate_defender_layout_and_data", away)


if __name__ == "__main__":
    unittest.main()
