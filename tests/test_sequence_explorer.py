from __future__ import annotations

import ast
from pathlib import Path
import unittest

import pandas as pd

from src.utils.sequence_normalization import normalize_sequence
from src.visualization.sequence_explorer import plot_sequence_explorer


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"


def _event(
    event_id,
    *,
    event_type="Pass",
    outcome="Successful",
    x=10.0,
    y=50.0,
    end_x=20.0,
    end_y=50.0,
    second=1.0,
    team="Home",
):
    minute = int(second // 60)
    sec = float(second % 60)

    return {
        "id": event_id,
        "eventId": event_id,
        "periodId": 1,
        "team_name": team,
        "type_name": event_type,
        "outcome": outcome,
        "x": x,
        "y": y,
        "end_x": end_x,
        "end_y": end_y,
        "timeMin": minute,
        "timeSec": sec,
        "playerName": f"Player {event_id}",
        "Mapped Jersey Number": event_id,
        "sequence_outcome_type":
            "Possession Retained",
        "terminal_outcome":
            "retained",
        "termination_reason":
            "controlled_exit",
        "viewpoint":
            "attacking",
    }


class SequenceExplorerTests(
    unittest.TestCase
):
    def test_normalizer_exposes_shared_contract(self):
        df = pd.DataFrame([
            _event(1),
        ])

        df["trigger_sequence_id"] = 7
        df["type_of_initial_trigger"] = (
            "Goal Kick"
        )
        df["timeMin_at_active_start"] = 0
        df["timeSec_at_active_start"] = 1
        df["buildup_active_duration_seconds"] = (
            4.25
        )

        sequence = normalize_sequence(
            df,
            sequence_type="buildup",
        )

        self.assertEqual(
            sequence["sequence_id"],
            7,
        )
        self.assertEqual(
            sequence["sequence_type"],
            "buildup",
        )
        self.assertEqual(
            sequence["trigger"],
            "Goal Kick",
        )
        self.assertAlmostEqual(
            sequence["duration_seconds"],
            4.25,
        )
        self.assertEqual(
            sequence["terminal_outcome"],
            "retained",
        )
        self.assertIn(
            "events",
            sequence,
        )

    def test_controlled_gap_becomes_inferred_carry(self):
        first = _event(
            1,
            end_x=25,
            end_y=40,
            second=1,
        )
        second = _event(
            2,
            x=35,
            y=45,
            end_x=50,
            end_y=46,
            second=4,
        )

        sequence = normalize_sequence(
            pd.DataFrame([
                first,
                second,
            ]),
            sequence_type=
                "offensive_transition",
        )

        kinds = [
            event["event_type"]
            for event
            in sequence["events"]
        ]

        self.assertEqual(
            kinds,
            [
                "turnover",
                "pass",
                "carry",
                "pass",
            ],
        )

        carry = next(
            event
            for event in sequence["events"]
            if event["event_type"]
            == "carry"
        )

        self.assertTrue(
            carry["metadata"][
                "inferred"
            ]
        )

    def test_failed_pass_does_not_seed_carry(self):
        first = _event(
            1,
            outcome="Unsuccessful",
            end_x=25,
            end_y=40,
            second=1,
        )
        second = _event(
            2,
            x=40,
            y=50,
            second=4,
        )

        sequence = normalize_sequence(
            pd.DataFrame([
                first,
                second,
            ]),
            sequence_type=
                "offensive_transition",
        )

        self.assertNotIn(
            "carry",
            [
                event[
                    "event_type"
                ]
                for event
                in sequence["events"]
            ],
        )

    def test_transition_recovery_is_turnover_event(self):
        recovery = _event(
            1,
            event_type="Ball recovery",
            x=30,
            y=55,
            end_x=None,
            end_y=None,
        )

        recovery["loss_sequence_id"] = 4
        recovery["type_of_initial_loss"] = (
            "Failed Pass"
        )
        recovery["timeMin_at_loss"] = 0
        recovery["timeSec_at_loss"] = 0

        sequence = normalize_sequence(
            pd.DataFrame([
                recovery,
            ]),
            sequence_type=
                "offensive_transition",
        )

        self.assertEqual(
            sequence["events"][0][
                "event_type"
            ],
            "turnover",
        )

    def test_set_piece_first_event_is_restart(self):
        restart = _event(
            1,
            x=5,
            y=10,
            end_x=30,
            end_y=25,
        )

        restart.update({
            "trigger_sequence_id": 9,
            "type_of_initial_trigger":
                "Corner",
            "restart_type": "Corner",
            "restart_delivery_type":
                "Direct Cross",
            "restart_execution_outcome":
                "Successful Delivery",
            "timeMin_at_trigger": 5,
            "timeSec_at_trigger": 20,
        })

        sequence = normalize_sequence(
            pd.DataFrame([
                restart,
            ]),
            sequence_type="set_piece",
        )

        self.assertEqual(
            sequence["events"][0][
                "event_type"
            ],
            "restart",
        )
        self.assertEqual(
            sequence["outcome"],
            "Successful Delivery",
        )
        self.assertIsNone(
            sequence[
                "duration_seconds"
            ]
        )

    def test_renderer_uses_shared_legend_and_summary(self):
        df = pd.DataFrame([
            _event(
                1,
                x=10,
                y=50,
                end_x=30,
                end_y=50,
            ),
            _event(
                2,
                event_type="Miss",
                x=35,
                y=50,
                end_x=100,
                end_y=55,
                second=5,
            ),
        ])

        df["loss_sequence_id"] = 2
        df["type_of_initial_loss"] = (
            "Interception"
        )
        df["timeMin_at_loss"] = 0
        df["timeSec_at_loss"] = 0

        sequence = normalize_sequence(
            df,
            sequence_type=
                "offensive_transition",
        )

        fig = plot_sequence_explorer(
            sequence,
            team_color="#e96a4a",
        )

        names = {
            trace.name
            for trace in fig.data
            if trace.showlegend
        }

        self.assertIn(
            "Pass",
            names,
        )
        self.assertIn(
            "Carry",
            names,
        )
        self.assertIn(
            "Shot",
            names,
        )

        annotations = " ".join(
            str(item.text)
            for item
            in fig.layout.annotations
        )

        self.assertIn(
            "Duration",
            annotations,
        )
        self.assertIn(
            "Outcome:",
            annotations,
        )
        self.assertIn(
            "Attacking",
            annotations,
        )

    def test_app_no_longer_calls_legacy_renderer(self):
        source = APP.read_text(
            encoding="utf-8"
        )
        tree = ast.parse(
            source
        )
        calls = []

        for node in ast.walk(
            tree
        ):
            if not isinstance(
                node,
                ast.Call,
            ):
                continue

            func = node.func

            if isinstance(
                func,
                ast.Attribute,
            ):
                name = func.attr
            elif isinstance(
                func,
                ast.Name,
            ):
                name = func.id
            else:
                continue

            if (
                name
                == "plot_opponent_buildup_after_loss_plotly"
            ):
                calls.append(
                    node.lineno
                )

        self.assertEqual(
            calls,
            [],
        )

    def test_four_callbacks_use_shared_explorer(self):
        source = APP.read_text(
            encoding="utf-8"
        )
        tree = ast.parse(
            source
        )

        targets = {
            "update_buildup_plot_and_indicator",
            "update_def_transition_plot",
            "update_off_transition_plot",
            "select_restart_from_map",
        }

        found = {}

        for node in tree.body:
            if (
                isinstance(
                    node,
                    ast.FunctionDef,
                )
                and node.name
                in targets
            ):
                text = ast.get_source_segment(
                    source,
                    node,
                )
                found[node.name] = text

        self.assertEqual(
            set(found),
            targets,
        )

        for name, text in found.items():
            with self.subTest(
                callback=name
            ):
                self.assertIn(
                    "normalize_sequence(",
                    text,
                )
                self.assertIn(
                    "plot_sequence_explorer(",
                    text,
                )

    def test_transition_has_exactly_one_turnover_trigger(self):
        rows = [
            _event(
                1,
                event_type="Ball recovery",
                x=30,
                y=50,
                end_x=None,
                end_y=None,
                second=1,
            ),
            _event(
                2,
                event_type="Tackle",
                x=35,
                y=48,
                end_x=None,
                end_y=None,
                second=3,
            ),
            _event(
                3,
                event_type="Interception",
                x=40,
                y=45,
                end_x=None,
                end_y=None,
                second=5,
            ),
        ]

        df = pd.DataFrame(rows)
        df["loss_sequence_id"] = 8
        df["type_of_initial_loss"] = "Dispossessed"
        df["timeMin_at_loss"] = 0
        df["timeSec_at_loss"] = 0

        sequence = normalize_sequence(
            df,
            sequence_type="offensive_transition",
        )

        turnovers = [
            event
            for event in sequence["events"]
            if event["event_type"] == "turnover"
        ]

        self.assertEqual(len(turnovers), 1)

    def test_transition_without_raw_turnover_gets_one_trigger(self):
        rows = [
            _event(
                1,
                event_type="Pass",
                x=30,
                y=50,
                end_x=45,
                end_y=52,
                second=2,
            ),
            _event(
                2,
                event_type="Miss",
                x=55,
                y=50,
                end_x=100,
                end_y=50,
                second=7,
            ),
        ]

        df = pd.DataFrame(rows)
        df["loss_sequence_id"] = 9
        df["type_of_initial_loss"] = "Unsuccessful Pass"
        df["timeMin_at_loss"] = 0
        df["timeSec_at_loss"] = 0

        sequence = normalize_sequence(
            df,
            sequence_type="defensive_transition",
        )

        turnovers = [
            event
            for event in sequence["events"]
            if event["event_type"] == "turnover"
        ]

        self.assertEqual(len(turnovers), 1)
        self.assertTrue(
            turnovers[0]["metadata"].get(
                "synthetic_trigger"
            )
        )

    def test_restart_summary_uses_delivery_not_missing_duration(self):
        restart = _event(
            1,
            x=5,
            y=10,
            end_x=35,
            end_y=30,
        )

        restart.update({
            "trigger_sequence_id": 12,
            "type_of_initial_trigger": "Corner",
            "restart_type": "Corner",
            "restart_delivery_type": "Direct Cross",
            "restart_execution_outcome": "Unsuccessful Delivery",
            "restart_length_m": 27.4,
            "timeMin_at_trigger": 12,
            "timeSec_at_trigger": 4,
        })

        sequence = normalize_sequence(
            pd.DataFrame([restart]),
            sequence_type="set_piece",
        )

        fig = plot_sequence_explorer(
            sequence,
            team_color="#e96a4a",
        )

        summary = str(
            fig.layout.annotations[0].text
        )

        self.assertNotIn("Duration", summary)
        self.assertIn("Direct Cross", summary)
        self.assertIn("27.4 m", summary)
        self.assertIn(
            "Unsuccessful Delivery",
            summary,
        )

    def test_sequence_summary_font_is_readable(self):
        sequence = normalize_sequence(
            pd.DataFrame([
                _event(1),
            ]),
            sequence_type="buildup",
        )

        fig = plot_sequence_explorer(
            sequence,
            team_color="#e96a4a",
        )

        self.assertGreaterEqual(
            fig.layout.annotations[0].font.size,
            13,
        )
    def test_turnover_step_marker_is_numbered_diamond(self):
        sequence = {
            "sequence_type":
                "offensive_transition",
            "trigger":
                "Pass Interception",
            "outcome":
                "Goals",
            "duration_seconds":
                8.0,
            "events": [
                {
                    "event_id": "trigger",
                    "event_type":
                        "turnover",
                    "raw_event_type":
                        "Pass Interception",
                    "team_name": "Napoli",
                    "player_name": None,
                    "second": 0.0,
                    "x": 25.0,
                    "y": 60.0,
                    "end_x": None,
                    "end_y": None,
                    "outcome":
                        "Possession change",
                    "successful": None,
                    "metadata": {},
                },
                {
                    "event_id": "pass-1",
                    "event_type": "pass",
                    "raw_event_type": "Pass",
                    "team_name": "Napoli",
                    "player_name": "Player",
                    "second": 2.0,
                    "x": 30.0,
                    "y": 55.0,
                    "end_x": 50.0,
                    "end_y": 50.0,
                    "outcome": "Successful",
                    "successful": True,
                    "metadata": {},
                },
            ],
        }

        fig = plot_sequence_explorer(
            sequence,
            team_color="#1597c2",
        )

        numbered_turnover = [
            trace
            for trace in fig.data
            if (
                getattr(
                    trace,
                    "mode",
                    None,
                )
                == "markers+text"
                and list(
                    getattr(
                        trace,
                        "text",
                        [],
                    )
                    or []
                )
                == ["1"]
            )
        ]

        self.assertEqual(
            len(numbered_turnover),
            1,
        )
        self.assertEqual(
            numbered_turnover[0]
            .marker.symbol,
            "diamond",
        )
        self.assertEqual(
            numbered_turnover[0]
            .marker.color,
            "#f0b44c",
        )

    def test_restart_summary_omits_zero_action_counters(self):
        restart = _event(
            1,
            x=5,
            y=10,
            end_x=35,
            end_y=30,
        )

        restart.update({
            "trigger_sequence_id": 13,
            "type_of_initial_trigger":
                "Goal Kick",
            "restart_type":
                "Goal Kick",
            "restart_delivery_type":
                "Long Goal Kick",
            "restart_execution_outcome":
                "Unsuccessful Delivery",
            "restart_length_m": 57.4,
            "timeMin_at_trigger": 12,
            "timeSec_at_trigger": 4,
        })

        sequence = normalize_sequence(
            pd.DataFrame([
                restart,
            ]),
            sequence_type="set_piece",
        )

        fig = plot_sequence_explorer(
            sequence,
            team_color="#e96a4a",
        )

        summary = str(
            fig.layout.annotations[0].text
        )

        self.assertIn(
            "Long Goal Kick",
            summary,
        )
        self.assertIn(
            "57.4 m",
            summary,
        )
        self.assertNotIn(
            "Passes 0",
            summary,
        )
        self.assertNotIn(
            "Carries 0",
            summary,
        )
        self.assertNotIn(
            "Shots 0",
            summary,
        )
        self.assertNotIn(
            "Turnovers 0",
            summary,
        )
    def test_marker_prefers_jersey_number_and_keeps_step_in_hover(self):
        sequence = {
            "sequence_type":
                "offensive_transition",
            "trigger":
                "Pass Interception",
            "outcome":
                "Possession Consolidated",
            "duration_seconds":
                5.0,
            "events": [
                {
                    "event_id": "trigger",
                    "event_type":
                        "turnover",
                    "raw_event_type":
                        "Pass Interception",
                    "team_name": "Napoli",
                    "player_name":
                        "S. Lobotka",
                    "jersey_number": 68,
                    "second": 0.0,
                    "x": 25.0,
                    "y": 60.0,
                    "end_x": None,
                    "end_y": None,
                    "outcome":
                        "Possession change",
                    "successful": None,
                    "metadata": {},
                },
            ],
        }

        fig = plot_sequence_explorer(
            sequence,
            team_color="#1597c2",
        )

        marker = next(
            trace
            for trace in fig.data
            if getattr(
                trace,
                "mode",
                None,
            ) == "markers+text"
        )

        self.assertEqual(
            list(marker.text),
            ["68"],
        )
        self.assertEqual(
            marker.marker.symbol,
            "diamond",
        )
        self.assertIn(
            "Step: 1",
            str(marker.hovertext[0]),
        )
        self.assertNotIn(
            "#68",
            str(marker.text[0]),
        )

    def test_defensive_transition_turnover_rotates_loss_into_transition_frame(self):
        rows = [
            _event(
                1,
                event_type="Pass",
                x=80,
                y=70,
                end_x=88,
                end_y=72,
                second=2,
            ),
            _event(
                2,
                event_type="Pass",
                x=88,
                y=72,
                end_x=92,
                end_y=70,
                second=5,
            ),
        ]

        df = pd.DataFrame(rows)
        df["loss_sequence_id"] = 77
        df["type_of_initial_loss"] = "Unsuccessful Pass"
        df["loss_x"] = 20.0
        df["loss_y"] = 30.0
        df["timeMin_at_loss"] = 0
        df["timeSec_at_loss"] = 0

        sequence = normalize_sequence(
            df,
            sequence_type="defensive_transition",
        )

        turnover = next(
            event
            for event in sequence["events"]
            if event["event_type"] == "turnover"
        )

        self.assertAlmostEqual(turnover["x"], 80.0)
        self.assertAlmostEqual(turnover["y"], 70.0)

        first_action = next(
            event
            for event in sequence["events"]
            if event["event_type"] == "pass"
        )

        self.assertAlmostEqual(
            turnover["x"],
            first_action["x"],
        )
        self.assertAlmostEqual(
            turnover["y"],
            first_action["y"],
        )

    def test_offensive_transition_trigger_is_not_double_rotated(self):
        rows = [
            _event(
                1,
                event_type="Pass",
                x=80,
                y=70,
                end_x=88,
                end_y=72,
                second=2,
            ),
        ]

        df = pd.DataFrame(rows)
        df["loss_sequence_id"] = 78
        df["type_of_initial_loss"] = "Pass Interception"
        df["loss_x"] = 80.0
        df["loss_y"] = 70.0
        df["timeMin_at_loss"] = 0
        df["timeSec_at_loss"] = 0

        sequence = normalize_sequence(
            df,
            sequence_type="offensive_transition",
        )

        turnover = next(
            event
            for event in sequence["events"]
            if event["event_type"] == "turnover"
        )

        self.assertAlmostEqual(turnover["x"], 80.0)
        self.assertAlmostEqual(turnover["y"], 70.0)

    def test_sequence_summary_exposes_match_time_for_video_review(self):
        row = _event(
            1,
            event_type="Pass",
            x=40,
            y=50,
            end_x=55,
            end_y=50,
            second=2707,
        )

        df = pd.DataFrame(
            [row]
        )
        df[
            "loss_sequence_id"
        ] = 99
        df[
            "type_of_initial_loss"
        ] = "Unsuccessful Pass"
        df[
            "timeMin_at_loss"
        ] = 45
        df[
            "timeSec_at_loss"
        ] = 7

        sequence = normalize_sequence(
            df,
            sequence_type=
                "defensive_transition",
        )

        fig = plot_sequence_explorer(
            sequence,
            team_color="#e96a4a",
        )

        summary = str(
            fig.layout.annotations[
                0
            ].text
        )

        self.assertIn(
            "Match time",
            summary,
        )
        self.assertIn(
            "45'07",
            summary,
        )

if __name__ == "__main__":
    unittest.main()
