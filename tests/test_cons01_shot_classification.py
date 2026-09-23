from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

import pandas as pd

from src import config
from src.data_processing import preprocess
from src.metrics.shot_classification import (
    SHOT_OUTCOME_BLOCKED,
    SHOT_OUTCOME_GOAL,
    SHOT_OUTCOME_OFF_TARGET,
    SHOT_OUTCOME_OWN_GOAL,
    SHOT_OUTCOME_POST,
    SHOT_OUTCOME_SAVED,
    SHOT_OUTCOME_UNKNOWN,
    classify_shots,
)
from src.metrics.shot_metrics import calculate_shot_stats
from src.reporting.analysis_summary import build_analysis_summary
from src.reporting.audit import MATCH_JSON_ENV_VAR, process_raw_opta_match
from src.reporting.bundle import build_match_report_data_bundle
from src.reporting.data_exports import build_events_core
from src.reporting.figure_catalog import build_report_figure_catalog
from src.reporting.pack_builder import build_pack_tables
from src.reporting.pdf_renderer import _overview_game_profile_rows
from src.utils import mapping_loader


class ShotClassificationTests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        rows = [
            {"id": 1, "team_name": "Home", "type_name": "Goal", "Own goal": 0, "Blocked": pd.NA, "x": 90, "y": 50},
            {"id": 2, "team_name": "Home", "type_name": "Attempt Saved", "Own goal": 0, "Blocked": pd.NA, "x": 85, "y": 45},
            {"id": 3, "team_name": "Home", "type_name": "Attempt Saved", "Own goal": 0, "Blocked": 1, "x": 80, "y": 55},
            {"id": 4, "team_name": "Home", "type_name": "Miss", "Own goal": 0, "Blocked": pd.NA, "x": 82, "y": 60},
            {"id": 5, "team_name": "Home", "type_name": "Post", "Own goal": 0, "Blocked": pd.NA, "x": 92, "y": 48},
            {"id": 6, "team_name": "Away", "type_name": "Attempt Saved", "Own goal": 0, "Blocked": pd.NA, "Keeper Saved": 1, "x": 78, "y": 50},
            {"id": 7, "team_name": "Away", "type_name": "Attempt Saved", "Own goal": 0, "Blocked": 1, "Keeper Saved": 1, "x": 78, "y": 50},
            {"id": 8, "team_name": "Away", "type_name": "Goal", "Own goal": 1, "Blocked": pd.NA, "x": 10, "y": 50},
        ]
        return pd.DataFrame(rows)

    def test_shared_classifier_disambiguates_saved_blocked_off_target_post_and_own_goal(self):
        shots = classify_shots(self._frame()).set_index("id")

        self.assertEqual(shots.loc[1, "shot_outcome"], SHOT_OUTCOME_GOAL)
        self.assertEqual(shots.loc[2, "shot_outcome"], SHOT_OUTCOME_SAVED)
        self.assertEqual(shots.loc[3, "shot_outcome"], SHOT_OUTCOME_BLOCKED)
        self.assertEqual(shots.loc[4, "shot_outcome"], SHOT_OUTCOME_OFF_TARGET)
        self.assertEqual(shots.loc[5, "shot_outcome"], SHOT_OUTCOME_POST)
        self.assertEqual(shots.loc[6, "shot_outcome"], SHOT_OUTCOME_OFF_TARGET)
        self.assertEqual(shots.loc[7, "shot_outcome"], SHOT_OUTCOME_UNKNOWN)
        self.assertEqual(shots.loc[8, "shot_outcome"], SHOT_OUTCOME_OWN_GOAL)

        self.assertTrue(bool(shots.loc[2, "shot_on_target"]))
        self.assertFalse(bool(shots.loc[3, "shot_on_target"]))
        self.assertTrue(pd.isna(shots.loc[7, "shot_on_target"]))
        self.assertTrue(pd.isna(shots.loc[8, "shot_on_target"]))
        self.assertTrue(bool(shots.loc[3, "shot_blocked"]))
        self.assertFalse(bool(shots.loc[2, "shot_blocked"]))

    def test_raw_opta_qualifier_82_is_mapped_to_blocked(self):
        root = Path(__file__).resolve().parents[1]
        fixture = root / "tests" / "fixtures" / "report11_real_shaped_match.json"
        raw = json.loads(fixture.read_text(encoding="utf-8"))

        attempt = next(
            event
            for event in raw["liveData"]["event"]
            if event.get("typeId") == 15
        )
        attempt.setdefault("qualifier", []).append({"qualifierId": 82})

        event_map = mapping_loader.load_opta_event_mapping(
            root / config.OPTA_EVENTS_XLSX
        )
        qualifier_map = mapping_loader.load_opta_qualifier_mapping(
            root / config.OPTA_QUALIFIERS_JSON
        )
        match_info = config.extract_match_info(raw)
        frame, _, _, _ = preprocess.process_opta_events(
            raw,
            event_map,
            qualifier_map,
            match_info,
        )
        classified = classify_shots(frame)
        row = classified.loc[classified["id"].eq(attempt["id"])].iloc[0]

        self.assertEqual(row["shot_outcome"], SHOT_OUTCOME_BLOCKED)
        self.assertTrue(bool(row["shot_blocked"]))
        self.assertFalse(bool(row["shot_on_target"]))

    def test_calculate_shot_stats_uses_canonical_classes(self):
        shots, home, away = calculate_shot_stats(
            self._frame(),
            "Home",
            "Away",
            None,
            None,
            None,
            None,
        )

        self.assertEqual(home["total_shots"], 5)
        self.assertEqual(home["shots_on_target"], 2)
        self.assertEqual(home["goals"], 1)
        self.assertEqual(home["saved_shots"], 1)
        self.assertEqual(home["blocked_shots"], 1)
        self.assertEqual(home["off_target_shots"], 1)
        self.assertEqual(home["woodwork_shots"], 1)

        # Away has two canonical attempts plus an own goal event. The own goal
        # remains explicit but is not credited as a shooting-team attempt.
        self.assertEqual(away["total_shots"], 2)
        self.assertEqual(away["shots_on_target"], 0)
        self.assertEqual(away["unknown_shots"], 1)
        self.assertEqual(away["off_target_shots"], 1)
        self.assertEqual(away["own_goals"], 1)
        self.assertEqual(len(shots), 8)


class ShotClassificationPackIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_report11_real_shaped_fixture import _processed_fixture

        frame, match_info = _processed_fixture()
        saved = frame.index[frame["type_name"].eq("Attempt Saved")].tolist()
        if len(saved) >= 2:
            frame.loc[saved[0], "Blocked"] = 1
        cls.frame = frame
        cls.match_info = match_info
        cls.bundle = build_match_report_data_bundle(frame, match_info)

    def test_overview_summary_and_csv_share_same_shot_values(self):
        profile = self.bundle.section("overview").data["game_profile"]
        tables = build_pack_tables(self.bundle)
        comparison = tables["tables/match-comparison.csv"].set_index("team_name")

        catalog = build_report_figure_catalog(self.bundle)
        generation = {
            "generation": {
                "pack_schema_version": "1.5",
                "status": "generated",
                "section_statuses": self.bundle.status_by_section(),
                "figures_expected": len(catalog.figures),
                "figures_generated": len(catalog.figures),
                "figures_empty": 0,
                "figures_skipped": 0,
                "figures_failed": 0,
                "required_figures_failed": 0,
            }
        }
        summary = build_analysis_summary(self.bundle, catalog, generation)
        summary_by_team = {
            row["team"]: row
            for row in summary["team_comparison"]
        }
        pdf_profile = _overview_game_profile_rows(self.bundle).set_index("Team")

        for team in self.bundle.teams:
            for field in (
                "shots",
                "shots_on_target",
                "saved_shots",
                "blocked_shots",
                "off_target_shots",
                "woodwork_shots",
                "unknown_shots",
                "own_goals",
            ):
                expected = profile[team][field]
                self.assertEqual(comparison.loc[team, field], expected)
                self.assertEqual(summary_by_team[team][field], expected)

            # The PDF Overview consumes the same game profile for its visible
            # shots-on-target value.
            self.assertEqual(
                int(pdf_profile.loc[team, "On target"]),
                int(profile[team]["shots_on_target"]),
            )

    def test_match_analysis_overview_uses_shared_classifier(self):
        app_source = Path("app.py").read_text(encoding="utf-8")
        self.assertIn("shot_metrics.calculate_shot_stats", app_source)
        self.assertIn('"Shots on target"', app_source)
        self.assertIn('"Blocked shots"', app_source)

    def test_events_core_retains_nullable_shot_audit_fields(self):
        core = build_events_core(self.bundle)
        for column in ("shot_outcome", "shot_on_target", "shot_blocked"):
            self.assertIn(column, core.columns)

        non_shots = core.loc[~core["is_shot"]]
        self.assertTrue(non_shots["shot_outcome"].isna().all())
        self.assertTrue(non_shots["shot_on_target"].isna().all())
        self.assertTrue(non_shots["shot_blocked"].isna().all())

        shots = core.loc[core["is_shot"]]
        self.assertTrue(shots["shot_outcome"].notna().all())
        blocked = shots.loc[shots["shot_outcome"].eq("blocked")]
        self.assertFalse(blocked.empty)
        self.assertTrue(blocked["shot_blocked"].eq(True).all())
        self.assertTrue(blocked["shot_on_target"].eq(False).all())


class NapoliUdineseExternalVerificationTests(unittest.TestCase):
    def test_external_napoli_udinese_is_6_2_with_3_7_blocked(self):
        raw_path = os.getenv(MATCH_JSON_ENV_VAR)
        if not raw_path:
            self.skipTest(
                f"Set {MATCH_JSON_ENV_VAR} to the external Napoli-Udinese JSON."
            )

        path = Path(raw_path)
        if not path.is_file():
            self.skipTest(f"External JSON not found: {path}")

        frame, match_info = process_raw_opta_match(path)
        home = match_info.get("hteamName")
        away = match_info.get("ateamName")
        self.assertEqual((home, away), ("Napoli", "Udinese"))

        _, home_stats, away_stats = calculate_shot_stats(
            frame,
            home,
            away,
            match_info.get("hxG"),
            match_info.get("axG"),
            match_info.get("hxGOT"),
            match_info.get("axGOT"),
        )

        self.assertEqual(home_stats["blocked_shots"], 3)
        self.assertEqual(away_stats["blocked_shots"], 7)
        self.assertEqual(home_stats["shots_on_target"], 6)
        self.assertEqual(away_stats["shots_on_target"], 2)

        bundle = build_match_report_data_bundle(frame, match_info)
        profile = bundle.section("overview").data["game_profile"]
        self.assertEqual(profile[home]["shots_on_target"], 6)
        self.assertEqual(profile[away]["shots_on_target"], 2)

        tables = build_pack_tables(bundle)
        comparison = tables["tables/match-comparison.csv"].set_index("team_name")
        self.assertEqual(int(comparison.loc[home, "shots_on_target"]), 6)
        self.assertEqual(int(comparison.loc[away, "shots_on_target"]), 2)


if __name__ == "__main__":
    unittest.main()
