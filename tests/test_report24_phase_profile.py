"""REPORT-24: phase profile (recoveries and losses by pitch third)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.metrics import phase_profile as pp
from src.reporting.analysis_summary import build_analysis_summary  # noqa: F401  (import check)
from src.reporting.bundle import build_match_report_data_bundle
from src.reporting.data_exports import (
    PHASE_PROFILE_COLUMNS,
    PHASE_SEQUENCE_COLUMNS,
    build_phase_profile,
    build_phase_sequences,
)
from src.reporting.pack_builder import TABLE_PATHS, build_pack_tables
from test_report11_real_shaped_fixture import _processed_fixture


def _events(rows, **meta):
    """rows: (type, outcome, x, y, end_x, seconds). Adds the sequence metadata."""
    frame = pd.DataFrame(rows, columns=["type_name", "outcome", "x", "y", "end_x", "total_seconds"])
    frame["end_y"] = np.nan
    for key, value in meta.items():
        frame[key] = value
    return frame


def _recovery(rows, zone="Defensive Third", terminal="turnover", seq=1):
    return _events(rows, loss_zone=zone, terminal_outcome=terminal, loss_sequence_id=seq)


def _loss(rows, zone="Attacking Third", terminal="turnover", seq=1, loss_t=(10, 0), loss_x=80.0, loss_y=40.0):
    return _events(
        rows,
        loss_zone=zone, terminal_outcome=terminal, loss_sequence_id=seq,
        loss_x=loss_x, loss_y=loss_y, timeMin_at_loss=loss_t[0], timeSec_at_loss=loss_t[1],
        type_of_initial_loss="Unsuccessful Pass", opponent_pass_count=2,
    )


class RecoveryRowTests(unittest.TestCase):
    def test_fast_forward_attack_reaches_final_third_and_shoots(self):
        seq = _recovery([
            ("Ball recovery", "Successful", 20.0, 50.0, np.nan, 100),
            ("Pass", "Successful", 20.0, 50.0, 45.0, 101),   # +25 forward
            ("Pass", "Successful", 45.0, 50.0, 70.0, 104),   # arrives past 66.67
            ("Miss", "Successful", 85.0, 50.0, np.nan, 107),
        ], terminal="shot")
        row = pp.recovery_sequence_row(seq)
        self.assertEqual(row["start_zone"], "Defensive Third")
        self.assertTrue(row["first_pass_forward"])
        self.assertTrue(row["reached_final_third"])
        self.assertEqual(row["seconds_to_final_third"], 4.0)
        self.assertEqual(row["completed_passes"], 2)
        self.assertAlmostEqual(row["gain_m"], (85.0 - 20.0) * 1.05, places=1)
        self.assertEqual(row["duration_s"], 7.0)
        self.assertTrue(row["produced_shot"])

    def test_sideways_first_pass_is_not_forward_and_short_sequence_never_arrives(self):
        seq = _recovery([
            ("Ball recovery", "Successful", 50.0, 20.0, np.nan, 10),
            ("Pass", "Successful", 50.0, 20.0, 52.0, 11),
            ("Pass", "Unsuccessful", 52.0, 20.0, 60.0, 13),
        ], zone="Middle Third")
        row = pp.recovery_sequence_row(seq)
        self.assertFalse(row["first_pass_forward"])
        self.assertFalse(row["reached_final_third"])
        self.assertIsNone(row["seconds_to_final_third"])
        self.assertFalse(row["produced_shot"])

    def test_unsuccessful_forward_pass_does_not_count_as_reaching_the_zone(self):
        seq = _recovery([
            ("Ball recovery", "Successful", 55.0, 50.0, np.nan, 10),
            ("Pass", "Unsuccessful", 55.0, 50.0, 80.0, 12),
        ], zone="Middle Third", terminal="turnover")
        row = pp.recovery_sequence_row(seq)
        self.assertTrue(row["first_pass_forward"])      # the intent was forward
        self.assertFalse(row["reached_final_third"])    # but control never got there
        self.assertAlmostEqual(row["gain_m"], 0.0, places=1)

    def test_recovery_already_in_final_third_has_no_reach_metric(self):
        seq = _recovery([("Ball recovery", "Successful", 80.0, 50.0, np.nan, 5)], zone="Attacking Third")
        row = pp.recovery_sequence_row(seq)
        self.assertIsNone(row["reached_final_third"])
        self.assertIsNone(row["seconds_to_final_third"])
        self.assertIsNone(row["first_pass_forward"])

    def test_empty_or_unlocatable_sequences_are_skipped(self):
        self.assertIsNone(pp.recovery_sequence_row(pd.DataFrame()))
        self.assertIsNone(pp.recovery_sequence_row(_recovery([("Pass", "Successful", np.nan, np.nan, np.nan, 1)])))


class LossRowTests(unittest.TestCase):
    def test_regained_sequence_reports_seconds_and_opponent_progress(self):
        seq = _loss([
            ("Pass", "Successful", 30.0, 50.0, 55.0, 12),
            ("Pass", "Successful", 55.0, 50.0, 70.0, 14),
            ("Pass", "Unsuccessful", 70.0, 50.0, 80.0, 16),
        ], terminal="turnover", loss_t=(0, 10))
        row = pp.loss_sequence_row(seq)
        self.assertEqual(row["loss_zone"], "Attacking Third")
        self.assertTrue(row["regained"])
        self.assertEqual(row["seconds_to_regain"], 6.0)
        self.assertTrue(row["opponent_reached_final_third"])
        self.assertFalse(row["opponent_shot"])
        self.assertEqual(row["opponent_passes"], 2)

    def test_consolidated_sequence_is_not_regained_and_has_no_regain_time(self):
        seq = _loss([("Pass", "Successful", 20.0, 40.0, 30.0, 12)], terminal="consolidated", loss_t=(0, 10))
        row = pp.loss_sequence_row(seq)
        self.assertFalse(row["regained"])
        self.assertIsNone(row["seconds_to_regain"])
        self.assertFalse(row["opponent_reached_final_third"])

    def test_shot_conceded_is_flagged(self):
        seq = _loss([
            ("Pass", "Successful", 60.0, 50.0, 80.0, 12),
            ("Attempt Saved", "Successful", 85.0, 50.0, np.nan, 14),
        ], terminal="shot", loss_t=(0, 10))
        row = pp.loss_sequence_row(seq)
        self.assertTrue(row["opponent_shot"])
        self.assertFalse(row["regained"])


class SummaryTests(unittest.TestCase):
    def _recoveries(self):
        rows = []
        for i in range(10):  # exactly at the sample threshold: not low
            rows.append({"sequence_id": i, "start_zone": "Defensive Third", "start_x": 20.0, "start_y": 50.0, "duration_s": 6.0,
                         "completed_passes": 2, "first_pass_forward": i < 4, "reached_final_third": i < 2,
                         "seconds_to_final_third": 5.0 if i < 2 else np.nan, "gain_m": 20.0, "terminal_outcome": "turnover",
                         "produced_shot": i == 0})
        rows.append({"sequence_id": 99, "start_zone": "Middle Third", "start_x": 50.0, "start_y": 50.0, "duration_s": 3.0,
                     "completed_passes": 1, "first_pass_forward": True, "reached_final_third": True,
                     "seconds_to_final_third": 2.0, "gain_m": 30.0, "terminal_outcome": "shot", "produced_shot": True})
        return pd.DataFrame(rows, columns=pp.RECOVERY_SEQUENCE_COLUMNS)

    def test_recovery_summary_percentages_flags_and_all_row(self):
        summary = pp.summarize_recoveries(self._recoveries()).set_index("zone")
        d = summary.loc["Defensive Third"]
        self.assertEqual(d["n"], 10)
        self.assertFalse(d["low_sample"])
        self.assertEqual(d["first_pass_forward_pct"], 40.0)
        self.assertEqual(d["reached_final_third_pct"], 20.0)
        self.assertEqual(d["median_seconds_to_final_third"], 5.0)
        self.assertEqual(d["shot_pct"], 10.0)
        m = summary.loc["Middle Third"]
        self.assertEqual(m["n"], 1)
        self.assertTrue(m["low_sample"])
        self.assertEqual(summary.loc["All", "n"], 11)
        self.assertEqual(int(summary.loc["Attacking Third", "n"]), 0)
        self.assertTrue(pd.isna(summary.loc["Attacking Third", "first_pass_forward_pct"]))

    def test_loss_summary_counts_unregained_as_not_quick(self):
        rows = [
            {"sequence_id": 1, "loss_zone": "Attacking Third", "loss_x": 80.0, "loss_y": 40.0, "initial_loss_type": "x", "regained": True,
             "seconds_to_regain": 3.0, "opponent_passes": 1, "opponent_reached_final_third": False, "terminal_outcome": "turnover", "opponent_shot": False},
            {"sequence_id": 2, "loss_zone": "Attacking Third", "loss_x": 80.0, "loss_y": 40.0, "initial_loss_type": "x", "regained": True,
             "seconds_to_regain": 8.0, "opponent_passes": 3, "opponent_reached_final_third": True, "terminal_outcome": "turnover", "opponent_shot": False},
            {"sequence_id": 3, "loss_zone": "Attacking Third", "loss_x": 80.0, "loss_y": 40.0, "initial_loss_type": "x", "regained": False,
             "seconds_to_regain": np.nan, "opponent_passes": 5, "opponent_reached_final_third": True, "terminal_outcome": "shot", "opponent_shot": True},
            {"sequence_id": 4, "loss_zone": "Attacking Third", "loss_x": 80.0, "loss_y": 40.0, "initial_loss_type": "x", "regained": False,
             "seconds_to_regain": np.nan, "opponent_passes": 4, "opponent_reached_final_third": False, "terminal_outcome": "consolidated", "opponent_shot": False},
        ]
        summary = pp.summarize_losses(pd.DataFrame(rows, columns=pp.LOSS_SEQUENCE_COLUMNS)).set_index("zone")
        a = summary.loc["Attacking Third"]
        self.assertEqual(a["n"], 4)
        self.assertTrue(a["low_sample"])
        self.assertEqual(a["regained_pct"], 50.0)
        self.assertEqual(a["regained_within_5s_pct"], 25.0)    # denominator is every loss
        self.assertEqual(a["regained_within_10s_pct"], 50.0)
        self.assertEqual(a["median_seconds_to_regain"], 5.5)
        self.assertEqual(a["opponent_final_third_pct"], 50.0)
        self.assertEqual(a["opponent_shot_pct"], 25.0)

    def test_summary_of_concatenated_matches_weights_every_sequence(self):
        one = self._recoveries()
        both = pd.concat([one, one], ignore_index=True)
        s1 = pp.summarize_recoveries(one).set_index("zone")
        s2 = pp.summarize_recoveries(both).set_index("zone")
        self.assertEqual(s2.loc["All", "n"], 2 * s1.loc["All", "n"])
        self.assertEqual(s2.loc["Defensive Third", "first_pass_forward_pct"], s1.loc["Defensive Third", "first_pass_forward_pct"])

    def test_zone_boundaries(self):
        self.assertEqual(pp.zone_from_x(0), "Defensive Third")
        self.assertEqual(pp.zone_from_x(33.32), "Defensive Third")
        self.assertEqual(pp.zone_from_x(33.34), "Middle Third")
        self.assertEqual(pp.zone_from_x(66.67), "Attacking Third")
        self.assertIsNone(pp.zone_from_x(np.nan))


class PackIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame, info = _processed_fixture()
        cls.bundle = build_match_report_data_bundle(frame, info)

    def test_tables_are_registered_and_have_stable_columns(self):
        self.assertIn("tables/phase-sequences.csv", TABLE_PATHS)
        self.assertIn("tables/phase-profile.csv", TABLE_PATHS)
        tables = build_pack_tables(self.bundle)
        self.assertEqual(tuple(tables["tables/phase-sequences.csv"].columns), PHASE_SEQUENCE_COLUMNS)
        self.assertEqual(tuple(tables["tables/phase-profile.csv"].columns), PHASE_PROFILE_COLUMNS)

    def test_sequence_counts_match_the_transition_sections(self):
        seqs = build_phase_sequences(self.bundle)
        for team in self.bundle.teams:
            off = self.bundle.section("offensive-transitions").data[team]["stats"]["total"]
            de = self.bundle.section("defensive-transitions").data[team]["stats"]["total"]
            part = seqs[seqs["team_name"] == team]
            self.assertEqual(int((part["phase"] == "recovery").sum()), off)
            self.assertEqual(int((part["phase"] == "loss").sum()), de)

    def test_profile_agrees_with_the_apps_existing_kpis(self):
        # Shot rates and the opponent's final-third rate use the same
        # "controlled location" milestones as the transition KPIs, so the
        # new table can never silently disagree with the existing ones.
        profile = build_phase_profile(self.bundle)
        for team in self.bundle.teams:
            off_k = self.bundle.section("offensive-transitions").data[team]["kpis"]
            def_k = self.bundle.section("defensive-transitions").data[team]["kpis"]
            r = profile[(profile.team_name == team) & (profile.phase == "recovery") & (profile.zone == "All")].iloc[0]
            l = profile[(profile.team_name == team) & (profile.phase == "loss") & (profile.zone == "All")].iloc[0]
            self.assertAlmostEqual(r["shot_pct"], off_k["shot_pct"], delta=0.06)
            self.assertAlmostEqual(l["opponent_shot_pct"], def_k["shot_pct"], delta=0.06)
            self.assertAlmostEqual(l["opponent_final_third_pct"], def_k["final_third_pct"], delta=0.06)

    def test_profile_has_three_zones_plus_all_for_each_phase(self):
        profile = build_phase_profile(self.bundle)
        for team in self.bundle.teams:
            for phase in ("recovery", "loss"):
                zones = profile[(profile.team_name == team) & (profile.phase == phase)]["zone"].tolist()
                self.assertEqual(zones, ["Defensive Third", "Middle Third", "Attacking Third", "All"])

    def test_analysis_summary_carries_the_phase_profile(self):
        from src.reporting.analysis_summary import build_analysis_summary, ANALYSIS_SUMMARY_SCHEMA_VERSION
        self.assertEqual(ANALYSIS_SUMMARY_SCHEMA_VERSION, "1.8")
        try:
            summary = build_analysis_summary(self.bundle)
        except TypeError:
            from src.reporting.figure_catalog import build_report_figure_catalog
            summary = build_analysis_summary(self.bundle, build_report_figure_catalog(self.bundle))
        # The summary must also validate against the published JSON schema
        # (the real audit caught a stale version constant once).
        import json
        from pathlib import Path
        from jsonschema import Draft202012Validator
        schema = json.loads(Path("src/reporting/schemas/analysis-summary.schema.json").read_text(encoding="utf-8"))
        errors = list(Draft202012Validator(schema).iter_errors(summary))
        self.assertEqual([e.message for e in errors], [])
        teams = summary["phase_profile"]["teams"]
        self.assertEqual([t["team"] for t in teams], list(self.bundle.teams))
        for t in teams:
            self.assertEqual([r["zone"] for r in t["recoveries"]], ["Defensive Third", "Middle Third", "Attacking Third", "All"])
            self.assertIn("regained_within_5s_pct", t["losses"][0])


if __name__ == "__main__":
    unittest.main()
