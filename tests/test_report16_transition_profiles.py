from __future__ import annotations

from io import BytesIO
import copy
import re
import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image
import plotly.graph_objects as go

from src.reporting.bundle import build_match_report_data_bundle
from src.reporting.figure_catalog import (
    RendererRegistry,
    build_report_figure_catalog,
)
from src.reporting.manifest import REPORT_MANIFEST
from src.reporting.pack_builder import build_pack_tables
from src.reporting.pdf_renderer import (
    MatchReportPdfConfig,
    _doc,
    _section_heading,
    _styles,
    _table_payloads,
    _transition_profile_rows,
    _transition_selection_rows,
    _transition_section_story,
    _transition_summary_rows,
    _transition_taxonomy_rows,
)
from tests.test_report11_real_shaped_fixture import _processed_fixture


def _tiny_png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (48, 27), "white").save(output, format="PNG")
    return output.getvalue()


def _dummy_renderer(*args, **kwargs):
    return go.Figure()


def _page_count(pdf: bytes) -> int:
    return len(re.findall(rb"/Type\s*/Page\b", pdf))


def _section_spec(section_id: str):
    return next(section for section in REPORT_MANIFEST.sections if section.id == section_id)


class Report16TransitionProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame, match_info = _processed_fixture()
        cls.bundle = build_match_report_data_bundle(frame, match_info)

    def test_overview_is_home_away_scalar_kpis(self):
        expected = [
            "Side",
            "Team",
            "Transitions",
            "Median duration (s)",
            "Final third %",
            "Penalty area %",
            "Shot %",
        ]
        for section_id in ("defensive-transitions", "offensive-transitions"):
            with self.subTest(section=section_id):
                frame = _transition_summary_rows(self.bundle, section_id)
                self.assertEqual(list(frame.columns), expected)
                self.assertEqual(frame["Side"].tolist(), ["Home", "Away"])
                self.assertEqual(frame["Team"].tolist(), list(self.bundle.teams))
                self.assertTrue((frame["Transitions"] > 0).all())
                for column in ("Final third %", "Penalty area %", "Shot %"):
                    self.assertTrue(frame[column].between(0, 100).all())

    def test_profile_uses_canonical_stats_table_not_combined_rows(self):
        raw, info = _processed_fixture()
        bundle = build_match_report_data_bundle(raw, info)
        expected_columns = [
            "Start zone",
            "Channel",
            "Home transitions",
            "Away transitions",
            "Home avg duration (s)",
            "Away avg duration (s)",
        ]
        for table_id in (
            "defensive-transitions-sequences",
            "offensive-transitions-sequences",
        ):
            section_id = (
                "defensive-transitions"
                if table_id.startswith("defensive")
                else "offensive-transitions"
            )
            section = bundle.section(section_id)
            for team in bundle.teams:
                # If the renderer regresses to combined, this sentinel will leak.
                section.data[team]["combined"] = pd.DataFrame(
                    [{"playerName": "EVENT_ROW_MUST_NOT_APPEAR", "nested": {"x": 1}}]
                )

            payloads = _table_payloads(bundle, table_id, selection_limit=10)
            self.assertEqual(len(payloads), 1)
            _, frame = payloads[0]
            self.assertEqual(list(frame.columns), expected_columns)
            self.assertNotIn("EVENT_ROW_MUST_NOT_APPEAR", frame.to_string())
            self.assertFalse(any(isinstance(value, (dict, list, tuple, set)) for value in frame.to_numpy().ravel()))

    def test_profile_taxonomy_is_deterministic_and_aligned(self):
        for section_id in ("defensive-transitions", "offensive-transitions"):
            with self.subTest(section=section_id):
                profile = _transition_profile_rows(self.bundle, section_id, limit=10)
                self.assertEqual(profile.iloc[0]["Start zone"], "Middle Third")
                self.assertEqual(profile.iloc[0]["Channel"], "Center")

                taxonomy = _transition_taxonomy_rows(self.bundle, section_id)
                self.assertEqual(
                    list(taxonomy.columns),
                    ["Dimension", "Category", "Home", "Away"],
                )
                self.assertTrue({"Outcome", "Terminal outcome", "Channel"}.issubset(set(taxonomy["Dimension"])))
                # One shared row per category means Home and Away necessarily use
                # the same taxonomy and category order.
                self.assertFalse(taxonomy[["Dimension", "Category"]].duplicated().any())
                channel_rows = taxonomy[taxonomy["Dimension"] == "Channel"]
                self.assertEqual(channel_rows["Category"].tolist(), ["Center"])

    def test_one_representative_sequence_per_team_with_reason(self):
        with patch.object(RendererRegistry, "resolve", return_value=_dummy_renderer):
            catalog = build_report_figure_catalog(self.bundle)

        for figure_id in (
            "defensive-transitions-top-sequence",
            "offensive-transitions-top-sequence",
        ):
            artifacts = catalog.by_id(figure_id)
            self.assertEqual(len(artifacts), 2)
            self.assertEqual(
                [artifact.team_name for artifact in artifacts],
                list(self.bundle.teams),
            )
            reasons = _transition_selection_rows(catalog, figure_id, self.bundle.teams)
            self.assertEqual(len(reasons), 2)
            self.assertTrue(reasons["Selection reason"].str.len().gt(10).all())
            self.assertFalse(reasons["Selection reason"].str.contains("stable_id", regex=False).any())

    def test_transition_sections_render_in_exactly_two_pages(self):
        png = _tiny_png()
        with patch.object(RendererRegistry, "resolve", return_value=_dummy_renderer), patch(
            "src.reporting.pdf_renderer.pio.to_image",
            return_value=png,
        ):
            catalog = build_report_figure_catalog(self.bundle)
            for section_id in ("defensive-transitions", "offensive-transitions"):
                with self.subTest(section=section_id):
                    config = MatchReportPdfConfig()
                    styles = _styles(config)
                    buffer = BytesIO()
                    doc = _doc(buffer, config)
                    spec = _section_spec(section_id)
                    story = _section_heading(spec, styles)
                    story.extend(
                        _transition_section_story(
                            self.bundle,
                            catalog,
                            spec,
                            styles,
                            config,
                        )
                    )
                    doc.multiBuild(story)
                    self.assertEqual(_page_count(buffer.getvalue()), 2)


    def test_realistic_full_taxonomy_still_fits_two_pages(self):
        png = _tiny_png()
        bundle = copy.deepcopy(self.bundle)

        for section_id in ("defensive-transitions", "offensive-transitions"):
            section = bundle.section(section_id)
            defensive = section_id == "defensive-transitions"
            for index, team in enumerate(bundle.teams):
                stats = section.data[team]["stats"]
                if defensive:
                    stats["outcomes"] = {
                        "Shots conceded": 4,
                        "Opponent Possession Consolidated": 25 + 7 * index,
                        "Regained Possessions": 32 - 11 * index,
                        "Out": 1 - index,
                        "Foul": 2 + index,
                        "Offside": index,
                    }
                else:
                    stats["outcomes"] = {
                        "Shots": 4,
                        "Possession Consolidated": 32 - 7 * index,
                        "Lost Possessions": 22 + 11 * index,
                        "Out": index,
                        "Foul": 3 - index,
                        "Offside": 1 - index,
                    }
                stats["terminal_outcomes"] = {
                    "shot": 4,
                    "consolidated": 25 + 7 * (1 - index),
                    "turnover": 32 - 11 * (1 - index),
                    "out": index,
                    "foul": 2 + index,
                    "offside": 1 - index,
                }
                stats["flanks"] = {
                    "Left": 16 + 8 * index,
                    "Center": 26 - 5 * index,
                    "Right": 22 - 6 * index,
                }

        with patch.object(RendererRegistry, "resolve", return_value=_dummy_renderer), patch(
            "src.reporting.pdf_renderer.pio.to_image",
            return_value=png,
        ):
            catalog = build_report_figure_catalog(bundle)
            for section_id in ("defensive-transitions", "offensive-transitions"):
                with self.subTest(section=section_id):
                    config = MatchReportPdfConfig()
                    styles = _styles(config)
                    buffer = BytesIO()
                    doc = _doc(buffer, config)
                    spec = _section_spec(section_id)
                    story = _section_heading(spec, styles)
                    story.extend(
                        _transition_section_story(
                            bundle, catalog, spec, styles, config
                        )
                    )
                    doc.multiBuild(story)
                    self.assertEqual(_page_count(buffer.getvalue()), 2)

    def test_full_event_rows_remain_in_transition_csvs(self):
        tables = build_pack_tables(self.bundle)
        for section_id, path in (
            ("defensive-transitions", "tables/defensive-transitions.csv"),
            ("offensive-transitions", "tables/offensive-transitions.csv"),
        ):
            section = self.bundle.section(section_id)
            expected_rows = sum(
                len(section.data[team]["combined"])
                for team in self.bundle.teams
            )
            frame = tables[path]
            self.assertEqual(len(frame), expected_rows)
            self.assertIn("report_team", frame.columns)
            self.assertIn("loss_sequence_id", frame.columns)


if __name__ == "__main__":
    unittest.main()
