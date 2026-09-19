from __future__ import annotations

from io import BytesIO
import re
import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image
import plotly.graph_objects as go

from src.metrics.cross_metrics import build_cross_flow_profile
from src.reporting.bundle import build_match_report_data_bundle
from src.reporting.figure_catalog import RendererRegistry, build_report_figure_catalog
from src.reporting.manifest import REPORT_MANIFEST
from src.reporting.pack_builder import build_pack_tables
from src.reporting.pdf_renderer import (
    MatchReportPdfConfig,
    _cross_flow_section_story,
    _cross_summary_rows,
    _doc,
    _section_heading,
    _styles,
    _table_payloads,
)
from tests.test_report11_real_shaped_fixture import _processed_fixture


def _tiny_png():
    output = BytesIO()
    Image.new("RGB", (48, 27), "white").save(output, format="PNG")
    return output.getvalue()


def _dummy_renderer(*args, **kwargs):
    return go.Figure()


def _page_count(pdf):
    return len(re.findall(rb"/Type\s*/Page\b", pdf))


def _section_spec(section_id):
    return next(section for section in REPORT_MANIFEST.sections if section.id == section_id)


def _sample_routes():
    rows = []
    origins = [
        "Left Deep", "Center Deep", "Right Deep", "Left Advanced", "Center Advanced",
        "Right Advanced", "Left Midfield", "Center Midfield", "Right Midfield", "Left Deep",
    ]
    destinations = [
        "Center Deep", "Right Deep", "Left Deep", "Center Advanced", "Right Advanced",
        "Left Advanced", "Center Midfield", "Right Midfield", "Left Midfield", "Right Advanced",
    ]
    for index, (origin, destination) in enumerate(zip(origins, destinations)):
        rows.append(
            {
                "Origin Zone": origin,
                "Destination Zone": destination,
                "Crosses": 10 - index,
                "Share %": 10.0,
                "Completed": 1,
                "Completion %": 50.0,
                "Retained": 1,
                "Retention %": 50.0,
                "Shots": 1 if index == 0 else 0,
                "Shot Rate %": 10.0 if index == 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _inject_cross_sample(bundle):
    section = bundle.section("cross-flow")
    routes = _sample_routes()
    for index, team in enumerate(bundle.teams):
        crosses = pd.DataFrame(
            [
                {
                    "playerName": "Home Crosser" if index == 0 else "Away Crosser",
                    "Outcome": "Completed" if number % 2 == 0 else "Incomplete",
                    "Retained": number < 6,
                    "Shot Generated": number < 2,
                }
                for number in range(10)
            ]
        )
        section.data[team] = {
            "crosses": crosses,
            "summary": {
                "total_crosses": 10,
                "retained_crosses": 6,
                "retention_pct": 60.0,
                "shot_crosses": 2,
                "shot_rate_pct": 20.0,
                "top_crosser": "Home Crosser" if index == 0 else "Away Crosser",
                "top_crosser_count": 6,
            },
            "routes": routes.copy(),
        }
    return bundle


class Report18CrossFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame, match_info = _processed_fixture()
        cls.bundle = _inject_cross_sample(build_match_report_data_bundle(frame, match_info))

    def test_manifest_has_only_two_static_flow_artifacts(self):
        spec = _section_spec("cross-flow")
        self.assertEqual([item.id for item in spec.figures], ["cross-flow-figure"])
        self.assertEqual([item.id for item in spec.tables], ["cross-summary", "cross-top-routes"])

    def test_complete_routes_can_be_retained_outside_pdf(self):
        analyzed = pd.DataFrame(
            [
                {
                    "Origin Zone": row["Origin Zone"],
                    "Destination Zone": row["Destination Zone"],
                    "Outcome": "Completed",
                    "Retained": True,
                    "Shot Generated": False,
                    "playerName": "Player",
                }
                for _, row in _sample_routes().iterrows()
            ]
        )
        _, full = build_cross_flow_profile(analyzed, limit=None)
        _, selected = build_cross_flow_profile(analyzed, limit=8)
        self.assertEqual(len(full), 10)
        self.assertEqual(len(selected), 8)

    def test_home_away_summary_contains_requested_metrics(self):
        frame = _cross_summary_rows(self.bundle)
        self.assertEqual(
            list(frame.columns),
            [
                "Side", "Team", "Crosses", "Completion %", "Retention %",
                "Shot rate %", "Top crosser",
            ],
        )
        self.assertEqual(frame["Side"].tolist(), ["Home", "Away"])
        self.assertEqual(frame["Crosses"].tolist(), [10, 10])
        self.assertEqual(frame["Completion %"].tolist(), [50.0, 50.0])
        self.assertEqual(frame["Retention %"].tolist(), [60.0, 60.0])
        self.assertEqual(frame["Shot rate %"].tolist(), [20.0, 20.0])

    def test_pdf_selects_eight_routes_but_csv_keeps_all(self):
        payloads = _table_payloads(self.bundle, "cross-top-routes", selection_limit=8)
        self.assertEqual(len(payloads), 2)
        self.assertTrue(all(len(frame) == 8 for _, frame in payloads))
        tables = build_pack_tables(self.bundle)
        self.assertIn("tables/cross-routes.csv", tables)
        self.assertEqual(len(tables["tables/cross-routes.csv"]), 20)

    def test_catalog_has_flow_only_for_both_teams(self):
        with patch.object(RendererRegistry, "resolve", return_value=_dummy_renderer):
            catalog = build_report_figure_catalog(self.bundle)
        self.assertEqual(len(catalog.by_id("cross-flow-figure")), 2)
        ids = {item.id for item in catalog.figures}
        self.assertNotIn("cross-origin-map", ids)
        self.assertNotIn("cross-destination-map", ids)

    def test_cross_flow_section_is_at_most_two_pages(self):
        png = _tiny_png()
        with patch.object(RendererRegistry, "resolve", return_value=_dummy_renderer), patch(
            "src.reporting.pdf_renderer.pio.to_image", return_value=png
        ):
            catalog = build_report_figure_catalog(self.bundle)
            config = MatchReportPdfConfig()
            styles = _styles(config)
            buffer = BytesIO()
            doc = _doc(buffer, config)
            spec = _section_spec("cross-flow")
            story = _section_heading(spec, styles)
            story.extend(_cross_flow_section_story(self.bundle, catalog, spec, styles, config))
            doc.multiBuild(story)
        self.assertLessEqual(_page_count(buffer.getvalue()), 2)


if __name__ == "__main__":
    unittest.main()
