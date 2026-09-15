from __future__ import annotations

from types import SimpleNamespace
import unittest

from src.reporting.audit import pdf_page_count, searchable_pdf_text
from src.reporting.bundle import build_match_report_data_bundle
from src.reporting.figure_catalog import build_report_figure_catalog
from src.reporting.manifest import REPORT_MANIFEST
from src.reporting.models import ReportManifest
from src.reporting.pdf_renderer import (
    _goal_context_rows,
    _overview_game_profile_rows,
    _overview_metadata_rows,
    _quality_strip_values,
    render_match_report_pdf,
)
from tests.test_report11_real_shaped_fixture import _processed_fixture


class Report17OverviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.match_info = _processed_fixture()
        cls.bundle = build_match_report_data_bundle(cls.frame, cls.match_info)

    def test_overview_contains_result_metadata_and_home_away_profile(self):
        metadata = _overview_metadata_rows(self.bundle)
        self.assertEqual(len(metadata), 1)
        self.assertEqual(metadata.iloc[0]["Home"], "Home FC")
        self.assertEqual(metadata.iloc[0]["Away"], "Away FC")
        self.assertEqual(metadata.iloc[0]["Score"], "1 - 1")
        self.assertEqual(metadata.iloc[0]["Competition"], "Serie A")

        profile = _overview_game_profile_rows(self.bundle)
        self.assertEqual(profile["Side"].tolist(), ["Home", "Away"])
        self.assertEqual(profile["Team"].tolist(), ["Home FC", "Away FC"])
        self.assertTrue({"Passes", "Shots", "Progressive passes", "Final third entries"}.issubset(profile.columns))

    def test_goal_rows_expose_origin_creation_and_linked_analysis(self):
        goals = _goal_context_rows(self.bundle)
        self.assertEqual(len(goals), 2)
        self.assertEqual(
            list(goals.columns),
            [
                "Goal",
                "Possession origin",
                "Attack type",
                "Decisive mechanism",
                "Creator",
                "Possession",
                "Passes",
                "Analysis / linked sequence",
            ],
        )

        transition_goal = goals.loc[
            goals["Attack type"].eq("Offensive Transition")
        ].iloc[0]
        self.assertIn("Away ST", transition_goal["Goal"])
        self.assertEqual(transition_goal["Possession origin"], "Ball recovery")
        self.assertEqual(transition_goal["Decisive mechanism"], "Fast progression")
        self.assertEqual(transition_goal["Possession"], "9.0s")
        self.assertEqual(int(transition_goal["Passes"]), 2)
        self.assertIn("Offensive Transition", transition_goal["Analysis / linked sequence"])
        self.assertIn("sequence", transition_goal["Analysis / linked sequence"])

        assisted = goals.loc[goals["Creator"].str.startswith("Assist")].iloc[0]
        self.assertIn("Home Sub", assisted["Creator"])
        self.assertNotIn("goal_type", " ".join(goals.columns).casefold())

        # When a canonical module sequence id cannot be resolved, the PDF
        # must describe the reconstructed chain without exposing raw Opta ids
        # that could be mistaken for a module sequence id.
        fallback = goals.loc[
            ~goals["Analysis / linked sequence"].str.contains("sequence", case=False, na=False)
        ]
        if not fallback.empty:
            text = fallback.iloc[0]["Analysis / linked sequence"]
            self.assertIn("reconstructed goal chain", text)
            self.assertNotIn("->", text)

    def test_unmapped_qualifiers_are_counted_in_quality_metadata(self):
        frame = self.frame.copy()
        frame["qualifier_999"] = 1
        bundle = build_match_report_data_bundle(frame, self.match_info)
        qualifiers = bundle.section("overview").data["data_coverage"]["qualifiers"]
        self.assertEqual(qualifiers["unmapped_count"], 1)
        self.assertEqual(tuple(qualifiers["unmapped_ids"]), (999,))

    def test_quality_strip_has_required_signals_and_aggregates_carries(self):
        coverage = self.bundle.section("overview").data["data_coverage"]
        original_carries = coverage["final_third_carries"]
        coverage["final_third_carries"] = {
            "Home FC": {"included": 2, "excluded": 1},
            "Away FC": {"included": 1, "excluded": 2},
        }
        try:
            values = _quality_strip_values(
                self.bundle,
                SimpleNamespace(figures=()),
            )
        finally:
            coverage["final_third_carries"] = original_carries

        by_label = {item["label"]: item for item in values}
        self.assertEqual(
            set(by_label),
            {
                "Receiver coverage",
                "Coordinate coverage",
                "Carry inference",
                "Outcome unknown",
                "Unmapped qualifiers",
                "Failures",
            },
        )
        self.assertEqual(by_label["Carry inference"]["value"], "3 in / 3 out")
        self.assertRegex(by_label["Failures"]["value"], r"^\d+ sections? / 0 plots$")
        self.assertNotIn(" sec ", by_label["Failures"]["value"])
        self.assertIn("%", by_label["Receiver coverage"]["value"])
        self.assertIn("%", by_label["Coordinate coverage"]["value"])

    def test_overview_pdf_is_at_most_two_pages_and_contains_no_technical_goal_type(self):
        overview_manifest = ReportManifest(
            id="report17-overview-only",
            schema_version=REPORT_MANIFEST.schema_version,
            sections=(REPORT_MANIFEST.sections[0],),
        )
        catalog = build_report_figure_catalog(self.bundle, overview_manifest)
        pdf = render_match_report_pdf(
            self.bundle,
            catalog,
            overview_manifest,
        )

        # Cover + Contents consume two pages. REPORT-17 allows at most two
        # additional pages for the Overview itself.
        self.assertLessEqual(pdf_page_count(pdf), 4)

        text = searchable_pdf_text(pdf)
        for required in (
            "Match snapshot",
            "Game profile",
            "Goals",
            "Possession origin",
            "Attack type",
            "Decisive mechanism",
            "Creator",
            "Analysis / linked sequence",
            "Quality strip",
            "Receiver coverage",
            "Coordinate coverage",
            "Unmapped qualifiers",
            "Failures",
        ):
            with self.subTest(required=required):
                self.assertIn(required, text)

        self.assertNotIn("goal_type", text.casefold())
        self.assertNotIn("Type G", text)

    def test_many_goal_rows_still_keep_overview_within_two_pages(self):
        from copy import deepcopy

        overview = self.bundle.section("overview").data
        original_origins = overview["goal_origins"]
        original_scorers = overview["scorers"]
        template = deepcopy(original_origins[0])
        origins = []
        scorers = []
        for index in range(12):
            goal = deepcopy(template)
            goal["goal_event_id"] = 1000 + index
            goal["minute"] = 1 + index * 7
            goal["second"] = 0
            goal["scorer"] = f"Player {index + 1}"
            goal["sequence_event_ids"] = [2000 + index * 3, 2001 + index * 3, 1000 + index]
            origins.append(goal)
            scorers.append(
                {
                    "goal_event_id": 1000 + index,
                    "team_name": goal["team_name"],
                    "scorer": goal["scorer"],
                    "minute": goal["minute"],
                    "second": 0,
                    "goal_type": "G",
                }
            )

        overview["goal_origins"] = origins
        overview["scorers"] = scorers
        try:
            overview_manifest = ReportManifest(
                id="report17-overview-many-goals",
                schema_version=REPORT_MANIFEST.schema_version,
                sections=(REPORT_MANIFEST.sections[0],),
            )
            catalog = build_report_figure_catalog(self.bundle, overview_manifest)
            pdf = render_match_report_pdf(self.bundle, catalog, overview_manifest)
        finally:
            overview["goal_origins"] = original_origins
            overview["scorers"] = original_scorers

        self.assertLessEqual(pdf_page_count(pdf), 4)
        text = searchable_pdf_text(pdf)
        self.assertIn("Player 12", text)
        self.assertNotIn("Type G", text)


if __name__ == "__main__":
    unittest.main()
