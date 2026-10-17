"""REPORT-25: the window analysis inside the Match Analysis Pack (tables and analysis-summary)."""

from __future__ import annotations

import json
import unittest
from unittest import mock

import pandas as pd

import plotly.graph_objects as go

from src.reporting import data_exports as de
from src.reporting import window_analysis as wa
from src.reporting.analysis_summary import ANALYSIS_SUMMARY_SCHEMA_VERSION, build_analysis_summary
from src.reporting.bundle import build_match_report_data_bundle
from src.reporting.figure_catalog import RendererRegistry, build_report_figure_catalog
from src.reporting.pack_builder import PACK_SCHEMA_VERSION, TABLE_PATHS, build_pack_tables
from test_report11_real_shaped_fixture import _processed_fixture


def _summary(bundle):
    """Same recipe as the REPORT-19 tests: a catalog with dummy renderers and a generation record."""
    with mock.patch.object(RendererRegistry, "resolve", return_value=lambda *a, **k: go.Figure()):
        catalog = build_report_figure_catalog(bundle)
    generation = {"generation": {
        "pack_schema_version": PACK_SCHEMA_VERSION, "status": "generated", "section_statuses": bundle.status_by_section(),
        "figures_expected": len(catalog.figures), "figures_generated": len(catalog.figures), "figures_empty": 0,
        "figures_skipped": 0, "figures_failed": 0, "required_figures_failed": 0,
    }}
    return build_analysis_summary(bundle, catalog, generation)

NEW_TABLES = {
    "tables/match-phases.csv": de.MATCH_PHASE_COLUMNS,
    "tables/window-profile.csv": de.WINDOW_PROFILE_COLUMNS,
    "tables/pressure-profile.csv": de.PRESSURE_PROFILE_COLUMNS,
    "tables/pressure-touches.csv": de.PRESSURE_TOUCH_COLUMNS,
    "tables/window-positions.csv": de.WINDOW_POSITION_COLUMNS,
    "tables/shot-droughts.csv": de.SHOT_DROUGHT_COLUMNS,
    "tables/insight-candidates.csv": de.INSIGHT_CANDIDATE_COLUMNS,
}


class WindowExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame, info = _processed_fixture()
        cls.bundle = build_match_report_data_bundle(frame, info)
        cls.tables = build_pack_tables(cls.bundle)

    def test_pack_declares_and_builds_every_new_table_with_its_columns(self):
        self.assertEqual(PACK_SCHEMA_VERSION, "1.12")
        for path, columns in NEW_TABLES.items():
            self.assertIn(path, TABLE_PATHS)
            self.assertEqual(tuple(self.tables[path].columns), tuple(columns), path)
            self.assertIn(path, de.CSV_EXPORT_DOCUMENTATION)

    def test_both_teams_and_the_whole_match_scope_are_present(self):
        profile = self.tables["tables/window-profile.csv"]
        self.assertEqual(set(profile["team_name"]), set(self.bundle.teams))
        self.assertIn(0, set(profile["phase_id"]))
        phases = self.tables["tables/match-phases.csv"]
        self.assertGreaterEqual(len(phases), 1)
        self.assertEqual(set(profile["phase_id"]) - {0}, set(phases["phase_id"]))

    def test_phases_add_up_to_the_whole_match(self):
        profile = self.tables["tables/window-profile.csv"]
        for team, part in profile.groupby("team_name"):
            whole = part[part["phase_id"] == 0].iloc[0]
            phases = part[part["phase_id"] > 0]
            self.assertEqual(int(phases["touches"].sum()), int(whole["touches"]), team)
            self.assertEqual(int(phases["shots"].sum()), int(whole["shots"]), team)
            self.assertAlmostEqual(float(phases["play_minutes"].sum()), float(whole["play_minutes"]), delta=0.6)

    def test_pressure_groups_partition_the_touches(self):
        table = self.tables["tables/pressure-profile.csv"]
        for (team, phase), part in table.groupby(["team_name", "phase_id"]):
            total = int(part[part["group_type"] == "all"]["touches"].sum())
            for group_type in ("lane", "third", "player"):
                self.assertEqual(int(part[part["group_type"] == group_type]["touches"].sum()), total, (team, phase, group_type))
            self.assertTrue((part["pressed"] <= part["touches"]).all())

    def test_per_touch_table_matches_the_profile(self):
        touches = self.tables["tables/pressure-touches.csv"]
        profile = self.tables["tables/pressure-profile.csv"]
        whole = profile[(profile["phase_id"] == 0) & (profile["group_type"] == "all")].set_index("team_name")
        for team in self.bundle.teams:
            outfield = touches[(touches["team_name"] == team)]
            self.assertGreaterEqual(len(outfield), int(whole.loc[team, "touches"]))   # the per-touch table also keeps goalkeeper touches

    def test_positions_carry_a_shared_frame_with_the_home_team_unchanged(self):
        pos = self.tables["tables/window-positions.csv"]
        home = pos[pos["team_name"] == self.bundle.teams[0]]
        away = pos[pos["team_name"] == self.bundle.teams[1]]
        if len(home):
            self.assertTrue((home["x"] == home["x_shared"]).all())
        if len(away):
            self.assertTrue(((100.0 - away["x"]).round(1) == away["x_shared"]).all())

    def test_analysis_summary_exposes_the_new_sections(self):
        self.assertEqual(ANALYSIS_SUMMARY_SCHEMA_VERSION, "1.8")
        summary = _summary(self.bundle)
        for key in ("match_phases", "window_profile", "pressure_profile", "shot_droughts", "insight_candidates"):
            self.assertIn(key, summary)
        self.assertEqual([t["team"] for t in summary["window_profile"]["teams"]], list(self.bundle.teams))
        self.assertEqual([t["team"] for t in summary["pressure_profile"]["teams"]], list(self.bundle.teams))
        json.dumps(summary, allow_nan=False)    # strict JSON: no NaN anywhere
        for candidate in summary["insight_candidates"]:
            self.assertIsInstance(candidate["numbers"], dict)
            self.assertTrue(candidate["caveat"])

    def test_analysis_is_computed_once_per_bundle(self):
        first = de._window_analysis_for(self.bundle)
        self.assertIs(de._window_analysis_for(self.bundle), first)

    def test_a_failing_analysis_never_breaks_the_pack(self):
        frame, info = _processed_fixture()
        bundle = build_match_report_data_bundle(frame, info)
        with mock.patch.object(wa, "compute_window_analysis", side_effect=RuntimeError("boom")):
            tables = build_pack_tables(bundle)
            summary = _summary(bundle)
        self.assertTrue(tables["tables/window-profile.csv"].empty)
        self.assertEqual(tuple(tables["tables/window-profile.csv"].columns), tuple(de.WINDOW_PROFILE_COLUMNS))
        self.assertEqual(summary["match_phases"], [])
        self.assertEqual(summary["insight_candidates"], [])


if __name__ == "__main__":
    unittest.main()
