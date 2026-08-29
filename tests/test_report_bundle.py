
from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from src.reporting.bundle import (
    MatchReportBundleConfig,
    ReportSectionStatus,
    _SECTION_PRODUCERS,
    _produced,
    build_match_report_data_bundle,
)


ROOT = Path(__file__).resolve().parents[1]


def complete_fixture():
    return pd.DataFrame([
        {
            "id": 1,
            "eventId": 1,
            "team_name": "Home FC",
            "playerName": "Home One",
            "type_name": "Pass",
            "typeId": 1,
            "outcome": "Successful",
            "x": 40.0,
            "y": 50.0,
            "end_x": 70.0,
            "end_y": 50.0,
            "timeMin": 1,
            "timeSec": 0,
            "periodId": 1,
        },
        {
            "id": 2,
            "eventId": 2,
            "team_name": "Away FC",
            "playerName": "Away One",
            "type_name": "Pass",
            "typeId": 1,
            "outcome": "Successful",
            "x": 40.0,
            "y": 50.0,
            "end_x": 70.0,
            "end_y": 50.0,
            "timeMin": 2,
            "timeSec": 0,
            "periodId": 1,
        },
    ])


def match_info():
    return {
        "hteamName": "Home FC",
        "ateamName": "Away FC",
        "hteamScore": 1,
        "ateamScore": 0,
    }


class ReportBundleTests(unittest.TestCase):
    def test_default_registry_covers_entire_manifest(self):
        from src.reporting import REPORT_MANIFEST

        self.assertEqual(
            set(_SECTION_PRODUCERS),
            {section.id for section in REPORT_MANIFEST.sections},
        )

    def test_complete_match_builds_every_section(self):
        def producer(ctx):
            return _produced(
                {
                    "teams": ctx.teams,
                    "rows": len(ctx.df),
                }
            )

        fake_registry = {
            section_id: producer
            for section_id in _SECTION_PRODUCERS
        }

        with patch.dict(
            _SECTION_PRODUCERS,
            fake_registry,
            clear=True,
        ):
            bundle = build_match_report_data_bundle(
                complete_fixture(),
                match_info(),
            )

        self.assertEqual(bundle.teams, ("Home FC", "Away FC"))
        self.assertEqual(
            {section.status for section in bundle.sections},
            {ReportSectionStatus.GENERATED},
        )
        self.assertEqual(len(bundle.sections), 16)

        encoded = bundle.to_json()
        decoded = json.loads(encoded)
        self.assertEqual(decoded["scope"], "full-match")
        self.assertEqual(decoded["teams"], ["Home FC", "Away FC"])

    def test_partial_data_still_returns_all_section_statuses(self):
        partial = pd.DataFrame([
            {
                "team_name": "Home FC",
                "type_name": "Pass",
            }
        ])

        def partial_producer(ctx):
            return _produced(
                {"available_columns": list(ctx.df.columns)},
                empty=False,
            )

        with patch.dict(
            _SECTION_PRODUCERS,
            {
                section_id: partial_producer
                for section_id in _SECTION_PRODUCERS
            },
            clear=True,
        ):
            bundle = build_match_report_data_bundle(
                partial,
                {"hteamName": "Home FC", "ateamName": "Away FC"},
            )

        self.assertEqual(len(bundle.sections), 16)
        self.assertTrue(
            all(
                section.status
                in {
                    ReportSectionStatus.GENERATED,
                    ReportSectionStatus.EMPTY,
                    ReportSectionStatus.SKIPPED,
                    ReportSectionStatus.ERROR,
                }
                for section in bundle.sections
            )
        )

    def test_empty_section_is_explicit(self):
        def normal(ctx):
            return _produced({"ok": True})

        def empty(ctx):
            return _produced(
                {"events": pd.DataFrame()},
                empty=True,
            )

        registry = {
            section_id: normal
            for section_id in _SECTION_PRODUCERS
        }
        registry["cross-flow"] = empty

        with patch.dict(
            _SECTION_PRODUCERS,
            registry,
            clear=True,
        ):
            bundle = build_match_report_data_bundle(
                complete_fixture(),
                match_info(),
            )

        self.assertEqual(
            bundle.section("cross-flow").status,
            ReportSectionStatus.EMPTY,
        )
        self.assertEqual(
            bundle.section("pass-network").status,
            ReportSectionStatus.GENERATED,
        )

    def test_isolated_error_does_not_abort_later_sections(self):
        def normal(ctx):
            return _produced({"ok": True})

        def broken(ctx):
            raise RuntimeError("synthetic isolated failure")

        registry = {
            section_id: normal
            for section_id in _SECTION_PRODUCERS
        }
        registry["defensive-shape"] = broken

        with patch.dict(
            _SECTION_PRODUCERS,
            registry,
            clear=True,
        ):
            bundle = build_match_report_data_bundle(
                complete_fixture(),
                match_info(),
            )

        failed = bundle.section("defensive-shape")
        self.assertEqual(failed.status, ReportSectionStatus.ERROR)
        self.assertEqual(failed.error_type, "RuntimeError")
        self.assertIn("synthetic isolated failure", failed.error_message)

        self.assertEqual(
            bundle.section("ppda").status,
            ReportSectionStatus.GENERATED,
        )
        self.assertEqual(
            bundle.section("methodology-appendix").status,
            ReportSectionStatus.GENERATED,
        )

    def test_disabled_section_is_skipped(self):
        enabled = ("overview", "methodology-appendix")

        def normal(ctx):
            return _produced({"ok": True})

        with patch.dict(
            _SECTION_PRODUCERS,
            {
                section_id: normal
                for section_id in _SECTION_PRODUCERS
            },
            clear=True,
        ):
            bundle = build_match_report_data_bundle(
                complete_fixture(),
                match_info(),
                MatchReportBundleConfig(
                    enabled_sections=enabled
                ),
            )

        self.assertEqual(
            bundle.section("overview").status,
            ReportSectionStatus.GENERATED,
        )
        self.assertEqual(
            bundle.section("pass-network").status,
            ReportSectionStatus.SKIPPED,
        )

    def test_shared_context_computation_runs_once(self):
        calls = {"count": 0}

        def first(ctx):
            value = ctx.once(
                "synthetic-shared",
                lambda: _increment(calls),
            )
            return _produced({"value": value})

        def second(ctx):
            value = ctx.once(
                "synthetic-shared",
                lambda: _increment(calls),
            )
            return _produced({"value": value})

        def normal(ctx):
            return _produced({"ok": True})

        registry = {
            section_id: normal
            for section_id in _SECTION_PRODUCERS
        }
        registry["overview"] = first
        registry["pass-network"] = second

        with patch.dict(
            _SECTION_PRODUCERS,
            registry,
            clear=True,
        ):
            build_match_report_data_bundle(
                complete_fixture(),
                match_info(),
            )

        self.assertEqual(calls["count"], 1)

    def test_bundle_json_normalizes_dataframes_numpy_and_infinity(self):
        import numpy as np

        def producer(ctx):
            return _produced({
                "frame": pd.DataFrame(
                    [{"value": np.int64(3), "missing": float("nan")}]
                ),
                "ppda": float("inf"),
            })

        with patch.dict(
            _SECTION_PRODUCERS,
            {
                section_id: producer
                for section_id in _SECTION_PRODUCERS
            },
            clear=True,
        ):
            bundle = build_match_report_data_bundle(
                complete_fixture(),
                match_info(),
            )

        payload = json.loads(bundle.to_json())
        first = payload["sections"][0]["data"]
        self.assertEqual(first["frame"][0]["value"], 3)
        self.assertIsNone(first["frame"][0]["missing"])
        self.assertEqual(first["ppda"], "Infinity")

    def test_reporting_bundle_has_no_app_dash_or_component_imports(self):
        path = ROOT / "src" / "reporting" / "bundle.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))

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
                    module == "app"
                    or module.startswith("app.")
                    or module == "dash"
                    or module.startswith("dash.")
                    or module == "dash_bootstrap_components"
                    or module.startswith("dash_bootstrap_components.")
                    or module == "src.components"
                    or module.startswith("src.components.")
                ):
                    forbidden.append(module)

        self.assertEqual(forbidden, [])

    def test_scope_is_full_match_and_ui_state_is_not_part_of_contract(self):
        config = MatchReportBundleConfig()
        self.assertEqual(config.scope.value, "full-match")

        source = (
            ROOT / "src" / "reporting" / "bundle.py"
        ).read_text(encoding="utf-8").lower()

        for token in (
            "active_tab",
            "callback_context",
            "store-df-match",
            "cross-filter-store",
        ):
            self.assertNotIn(token, source)


    def test_scorers_have_one_canonical_schema(self):
        from types import SimpleNamespace
        from src.reporting.bundle import _goal_list

        ctx = SimpleNamespace(
            home_team="Home FC",
            away_team="Away FC",
            match_info={
                "goals": [
                    {
                        "team_position": "home",
                        "scorer": "Home Nine",
                        "timeMin": 12,
                        "timeMinSec": "12:34",
                        "periodId": 1,
                        "goal_type": "G",
                    }
                ]
            },
        )

        supplied = _goal_list(
            ctx,
            [
                {
                    "team_name": "Away FC",
                    "scorer": "Unused Origin",
                    "minute": 50,
                    "second": 1,
                    "goal_event_id": 999,
                }
            ],
        )

        expected_keys = {
            "team_name",
            "team_position",
            "scorer",
            "minute",
            "second",
            "period_id",
            "goal_type",
            "goal_event_id",
        }
        self.assertEqual(set(supplied[0]), expected_keys)
        self.assertEqual(supplied[0]["team_name"], "Home FC")
        self.assertEqual(supplied[0]["team_position"], "home")
        self.assertEqual(supplied[0]["minute"], 12)
        self.assertEqual(str(supplied[0]["second"]), "34")

        fallback_ctx = SimpleNamespace(
            home_team="Home FC",
            away_team="Away FC",
            match_info={"goals": []},
        )
        fallback = _goal_list(
            fallback_ctx,
            [
                {
                    "team_name": "Away FC",
                    "scorer": "Away Ten",
                    "period_id": 2,
                    "minute": 55,
                    "second": 9,
                    "goal_event_id": 5001,
                }
            ],
        )

        self.assertEqual(set(fallback[0]), expected_keys)
        self.assertEqual(fallback[0]["team_position"], "away")
        self.assertEqual(fallback[0]["team_name"], "Away FC")
        self.assertEqual(fallback[0]["goal_event_id"], 5001)

    def test_real_registry_smoke_uses_canonical_metric_boundaries(self):
        from contextlib import ExitStack

        from src.data_processing import pass_processing
        from src.metrics import (
            buildup_metrics,
            cross_metrics,
            data_quality,
            defensive_contribution_metrics,
            defensive_metrics,
            goal_origin_metrics,
            pass_metrics,
            pass_network_metrics,
            player_metrics,
            player_pass_map_metrics,
            restart_metrics,
            restart_panel_metrics,
            sequence_outcome_metrics,
            set_piece_metrics,
            shot_metrics,
            shot_sequence_involvement_metrics,
            shot_sequence_metrics,
            threat_reception_metrics,
            transition_metrics,
        )
        from src.visualization import formation_plotly

        fixture = pd.DataFrame([
            {
                "id": 987654321,
                "eventId": 987654321,
                "team_name": "Registry Home",
                "playerName": "Registry Home One",
                "type_name": "Pass",
                "typeId": 1,
                "outcome": "Successful",
                "x": 40.0,
                "y": 50.0,
                "end_x": 70.0,
                "end_y": 50.0,
                "timeMin": 1,
                "timeSec": 0,
                "periodId": 1,
                "Penalty": 0,
            },
            {
                "id": 987654322,
                "eventId": 987654322,
                "team_name": "Registry Away",
                "playerName": "Registry Away One",
                "type_name": "Pass",
                "typeId": 1,
                "outcome": "Successful",
                "x": 40.0,
                "y": 50.0,
                "end_x": 70.0,
                "end_y": 50.0,
                "timeMin": 2,
                "timeSec": 0,
                "periodId": 1,
                "Penalty": 0,
            },
        ])

        passes = fixture.copy()
        passes["receiver"] = [
            "Registry Home Two",
            "Registry Away Two",
        ]
        passes["receiver_is_reliable"] = True
        passes["is_progressive_attempt"] = True
        passes["is_progressive"] = True
        passes["progressive_distance_m"] = 12.0
        passes["progressive_channel"] = "Central"

        carries = pd.DataFrame([
            {
                "team_name": "Registry Home",
                "x": 60.0,
                "end_x": 70.0,
            },
            {
                "team_name": "Registry Away",
                "x": 60.0,
                "end_x": 70.0,
            },
        ])

        entry_df = pd.DataFrame([
            {
                "entry_type": "Pass",
                "team_name": "Registry Home",
            }
        ])
        entry_stats = {
            "total_final_third": 1,
            "carry_entry_candidates": 1,
            "carry_entries": 1,
            "carry_entries_excluded_total": 0,
        }

        cross_df = pd.DataFrame([
            {
                "playerName": "Registry Crosser",
                "Origin Zone": "Left Advanced",
                "Destination Zone": "Center Deep",
                "Retained": True,
                "Shot Generated": True,
            }
        ])
        cross_summary = {
            "total_crosses": 1,
            "retention_pct": 100.0,
            "shot_rate_pct": 100.0,
            "top_crosser": "Registry Crosser",
        }
        cross_routes = pd.DataFrame([
            {
                "Origin Zone": "Left Advanced",
                "Destination Zone": "Center Deep",
                "Crosses": 1,
            }
        ])

        transition_df = pd.DataFrame([
            {
                "loss_sequence_id": "registry-seq",
                "y": 50.0,
                "sequence_outcome_type": "Possession Retained",
                "terminal_outcome": "possession_retained",
                "type_of_initial_loss": "Pass",
            }
        ])

        restart_sequence = pd.DataFrame([
            {
                "type_name": "Pass",
                "type_of_initial_trigger": "Corner",
            }
        ])
        restart_analysis = pd.DataFrame([
            {
                "Action Type": "Corner",
                "playerName": "Registry Home One",
                "sequence_id": "restart-1",
            }
        ])

        player_stats = pd.DataFrame(
            [{"Total Passes": 1}],
            index=pd.Index(
                ["Registry Home One"],
                name="playerName",
            ),
        )
        shot_sequence_stats = pd.DataFrame([
            {
                "playerName": "Registry Home One",
                "Shot Sequence Involvements": 1,
            }
        ])
        ranking = pd.DataFrame([
            {
                "playerName": "Registry Home One",
                "value": 1,
            }
        ])

        goal_origins = [
            {
                "goal_event_id": 987654399,
                "team_name": "Registry Home",
                "scorer": "Registry Home Nine",
                "period_id": 1,
                "minute": 10,
                "second": 5,
            }
        ]

        mean_players = pd.DataFrame([
            {
                "playerName": "Registry Home One",
                "median_x": 50.0,
                "median_y": 50.0,
            }
        ])

        network_edges = pd.DataFrame([
            {
                "player1": "Registry Home One",
                "player2": "Registry Home Two",
                "pass_count": 1,
            }
        ])
        network_nodes = pd.DataFrame([
            {
                "playerName": "Registry Home One",
                "pass_involvement": 1,
            }
        ])

        buildup_df = pd.DataFrame([
            {
                "sequence_id": "buildup-1",
                "team_name": "Registry Home",
            }
        ])
        buildup_summary = pd.DataFrame([
            {
                "sequence_id": "buildup-1",
                "final_outcome": "Possession Consolidated",
            }
        ])

        ppda_profile = {
            "overall": {
                "opponent_passes": 1,
                "defensive_actions": 1,
            },
            "timeline": pd.DataFrame(),
        }

        def add_mock(stack, module, name, return_value):
            return stack.enter_context(
                patch.object(
                    module,
                    name,
                    autospec=True,
                    return_value=return_value,
                )
            )

        with ExitStack() as stack:
            add_mock(stack, pass_processing, "get_passes_df", passes)
            add_mock(stack, pass_processing, "infer_carries", carries)
            add_mock(
                stack,
                pass_metrics,
                "classify_progressive_passes",
                passes,
            )
            add_mock(
                stack,
                player_metrics,
                "calculate_player_stats",
                player_stats,
            )
            add_mock(
                stack,
                goal_origin_metrics,
                "classify_goal_origins",
                goal_origins,
            )
            add_mock(
                stack,
                formation_plotly,
                "build_formation_timeline_model",
                {"moments": [{"minute": 0}]},
            )
            add_mock(
                stack,
                shot_sequence_metrics,
                "calculate_shot_sequence_player_stats",
                shot_sequence_stats,
            )
            add_mock(
                stack,
                defensive_metrics,
                "get_defensive_actions",
                pd.DataFrame([
                    {
                        "team_name": "Registry Home",
                        "type_name": "Ball recovery",
                    },
                    {
                        "team_name": "Registry Away",
                        "type_name": "Ball recovery",
                    },
                ]),
            )
            add_mock(
                stack,
                shot_metrics,
                "calculate_shot_stats",
                (
                    pd.DataFrame([{"type_name": "Goal"}]),
                    {"total_shots": 1, "shots_on_target": 1},
                    {"total_shots": 1, "shots_on_target": 1},
                ),
            )
            add_mock(
                stack,
                pass_metrics,
                "progressive_pass_summary",
                {"attempted": 1, "successful": 1},
            )
            add_mock(
                stack,
                pass_metrics,
                "progressive_pass_player_summary",
                pd.DataFrame([
                    {
                        "Player": "Registry Home One",
                        "Successful": 1,
                    }
                ]),
            )
            add_mock(
                stack,
                pass_metrics,
                "analyze_final_third_entries",
                (entry_df, entry_stats),
            )
            add_mock(
                stack,
                data_quality,
                "receiver_coverage",
                {"coverage_pct": 100.0},
            )
            add_mock(
                stack,
                data_quality,
                "coordinate_coverage",
                {"coverage_pct": 100.0},
            )
            add_mock(
                stack,
                data_quality,
                "outcome_coverage",
                {"known_pct": 100.0},
            )
            add_mock(
                stack,
                data_quality,
                "carry_candidate_coverage_from_stats",
                {"inclusion_pct": 100.0},
            )
            add_mock(
                stack,
                player_metrics,
                "get_mean_positions_profile",
                (
                    mean_players,
                    {"eligible_players": 1},
                ),
            )
            add_mock(
                stack,
                pass_network_metrics,
                "build_pass_network_profile",
                (
                    network_edges,
                    network_nodes,
                    {"shown_connections": 1},
                ),
            )
            add_mock(
                stack,
                cross_metrics,
                "analyze_crosses",
                cross_df,
            )
            add_mock(
                stack,
                cross_metrics,
                "build_cross_flow_profile",
                (cross_summary, cross_routes),
            )
            add_mock(
                stack,
                buildup_metrics,
                "find_buildup_sequences",
                buildup_df,
            )
            add_mock(
                stack,
                sequence_outcome_metrics,
                "summarize_sequences",
                buildup_summary,
            )
            add_mock(
                stack,
                sequence_outcome_metrics,
                "build_sequence_comparison",
                {"home": {}, "away": {}},
            )
            add_mock(
                stack,
                defensive_metrics,
                "build_defensive_shape_profile",
                {
                    "action_count": 1,
                    "actions": pd.DataFrame([{"x": 50.0}]),
                },
            )
            add_mock(
                stack,
                defensive_metrics,
                "calculate_ppda_profile",
                ppda_profile,
            )
            add_mock(
                stack,
                defensive_metrics,
                "extract_ppda_key_events",
                pd.DataFrame(),
            )
            add_mock(
                stack,
                transition_metrics,
                "find_buildup_after_possession_loss",
                transition_df,
            )
            add_mock(
                stack,
                transition_metrics,
                "calculate_def_transition_stats",
                {"total": 1},
            )
            add_mock(
                stack,
                transition_metrics,
                "calculate_off_transition_stats",
                {"total": 1},
            )
            add_mock(
                stack,
                set_piece_metrics,
                "extract_penalty_set_piece_sequences",
                [],
            )
            add_mock(
                stack,
                restart_metrics,
                "extract_restart_sequences",
                [restart_sequence],
            )
            add_mock(
                stack,
                set_piece_metrics,
                "analyze_and_summarize_set_pieces",
                (
                    restart_analysis,
                    {"total": 1},
                ),
            )
            add_mock(
                stack,
                restart_panel_metrics,
                "build_restart_records",
                [{"sequence_id": "restart-1"}],
            )
            add_mock(
                stack,
                shot_sequence_involvement_metrics,
                "prepare_shot_sequence_ranking",
                ranking,
            )
            add_mock(
                stack,
                defensive_contribution_metrics,
                "build_defensive_ranking",
                ranking,
            )
            add_mock(
                stack,
                player_pass_map_metrics,
                "classify_player_passes",
                pd.DataFrame([{"pass_type": "completed"}]),
            )
            add_mock(
                stack,
                player_pass_map_metrics,
                "player_pass_profile",
                {"passes": 1},
            )
            add_mock(
                stack,
                threat_reception_metrics,
                "team_player_options",
                [{"player_name": "Registry Home One"}],
            )
            add_mock(
                stack,
                threat_reception_metrics,
                "received_passes_for_player",
                pd.DataFrame([{"receiver": "Registry Home One"}]),
            )
            add_mock(
                stack,
                threat_reception_metrics,
                "reception_summary",
                {"receptions": 1},
            )
            add_mock(
                stack,
                defensive_contribution_metrics,
                "team_defensive_player_options",
                [{"player_name": "Registry Home One"}],
            )
            add_mock(
                stack,
                defensive_contribution_metrics,
                "classify_defensive_events",
                pd.DataFrame([{"type_name": "Tackle"}]),
            )
            add_mock(
                stack,
                defensive_contribution_metrics,
                "player_defensive_profile",
                {"defensive_actions": 1},
            )

            bundle = build_match_report_data_bundle(
                fixture,
                {
                    "hteamName": "Registry Home",
                    "ateamName": "Registry Away",
                    "hteamScore": 1,
                    "ateamScore": 0,
                },
            )

        errors = {
            section.id: (
                section.error_type,
                section.error_message,
            )
            for section in bundle.sections
            if section.status is ReportSectionStatus.ERROR
        }
        self.assertEqual(errors, {})
        self.assertEqual(
            {section.status for section in bundle.sections},
            {ReportSectionStatus.GENERATED},
        )

        overview = bundle.section("overview").data
        self.assertEqual(
            overview["scorers"][0]["team_name"],
            "Registry Home",
        )

def _increment(calls):
    calls["count"] += 1
    return calls["count"]


if __name__ == "__main__":
    unittest.main()
