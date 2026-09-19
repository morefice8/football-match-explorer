from __future__ import annotations

from io import BytesIO
import json
import unittest
from unittest.mock import patch
import zipfile

from jsonschema import Draft202012Validator
from PIL import Image
import plotly.graph_objects as go

from src.reporting.analysis_summary import (
    ANALYSIS_SUMMARY_MAX_BYTES,
    ANALYSIS_SUMMARY_SCHEMA_FILE,
    build_analysis_summary,
)
from src.reporting.bundle import build_match_report_data_bundle
from src.reporting.figure_catalog import (
    RendererRegistry,
    build_report_figure_catalog,
)
from src.reporting.pack_builder import build_match_analysis_pack
from tests.test_report11_real_shaped_fixture import _processed_fixture


def _dummy_renderer(*args, **kwargs):
    return go.Figure()


def _tiny_png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (48, 27), "white").save(output, format="PNG")
    return output.getvalue()


class Report19AnalysisSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame, match_info = _processed_fixture()
        cls.bundle = build_match_report_data_bundle(frame, match_info)
        with patch.object(
            RendererRegistry,
            "resolve",
            return_value=_dummy_renderer,
        ):
            cls.catalog = build_report_figure_catalog(cls.bundle)

        cls.generation = {
            "generation": {
                "pack_schema_version": "1.3",
                "status": "generated",
                "section_statuses": cls.bundle.status_by_section(),
                "figures_expected": len(cls.catalog.figures),
                "figures_generated": len(cls.catalog.figures),
                "figures_empty": 0,
                "figures_skipped": 0,
                "figures_failed": 0,
                "required_figures_failed": 0,
            }
        }
        cls.summary = build_analysis_summary(
            cls.bundle,
            cls.catalog,
            cls.generation,
        )
        cls.schema = json.loads(
            ANALYSIS_SUMMARY_SCHEMA_FILE.read_text(encoding="utf-8")
        )

    def test_schema_is_valid_and_summary_validates(self):
        Draft202012Validator.check_schema(self.schema)
        Draft202012Validator(self.schema).validate(self.summary)

    def test_top_level_contract_is_exact_and_stable(self):
        expected = {
            "schema_version",
            "match",
            "score_and_goals",
            "data_quality",
            "team_comparison",
            "formations",
            "passing",
            "progression",
            "final_third_entries",
            "crosses",
            "buildup",
            "defensive_shape",
            "ppda",
            "defensive_transitions",
            "offensive_transitions",
            "restarts",
            "top_players",
            "representative_sequences",
            "generation_status",
        }
        self.assertEqual(set(self.summary), expected)

    def test_summary_is_strict_json_native_and_under_one_mib(self):
        encoded = json.dumps(
            self.summary,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        self.assertLess(len(encoded), ANALYSIS_SUMMARY_MAX_BYTES)
        self.assertNotIn(b"NaN", encoded)

    def test_rankings_are_capped(self):
        for family in ("passing", "shooting", "defending"):
            for team_payload in self.summary["top_players"][family]:
                self.assertLessEqual(len(team_payload["ranking"]), 10)

        for team_payload in self.summary["progression"]["teams"]:
            self.assertLessEqual(len(team_payload["top_passers"]), 10)

        for team_payload in self.summary["crosses"]["teams"]:
            self.assertLessEqual(len(team_payload["top_routes"]), 8)

    def test_representative_events_are_globally_deduplicated(self):
        representative = self.summary["representative_sequences"]
        event_keys = [
            event["event_key"]
            for event in representative["events"]
        ]
        self.assertEqual(len(event_keys), len(set(event_keys)))
        pool = set(event_keys)
        for sequence in representative["sequences"]:
            self.assertLessEqual(len(sequence["event_ids"]), 120)
            self.assertTrue(set(sequence["event_ids"]).issubset(pool))

    def test_no_bulk_event_tables_leak_into_summary(self):
        text = json.dumps(
            self.summary,
            ensure_ascii=False,
            allow_nan=False,
        )
        for forbidden in (
            '"classified_passes"',
            '"combined"',
            '"map_datasets"',
            '"event_explorer"',
        ):
            self.assertNotIn(forbidden, text)

    def test_match_date_and_formation_minutes_are_ai_ready(self):
        self.assertIsNotNone(self.summary["match"]["date"])
        timeline = self.summary["formations"]["timeline"]
        self.assertTrue(timeline)
        self.assertTrue(
            all(item["minute"] is not None for item in timeline)
        )

    def test_representative_events_use_global_ids_and_receivers(self):
        events = self.summary["representative_sequences"]["events"]
        self.assertTrue(events)
        for event in events:
            if event["event_id"] is not None:
                self.assertEqual(
                    event["event_key"],
                    str(event["event_id"]),
                )

        pass_events = [
            event
            for event in events
            if event.get("event_type") == "Pass"
        ]
        self.assertTrue(pass_events)
        self.assertTrue(
            any(event.get("receiver") for event in pass_events)
        )

    def test_pack_contains_summary_and_schema(self):
        png = _tiny_png()
        with patch.object(
            RendererRegistry,
            "resolve",
            return_value=_dummy_renderer,
        ), patch(
            "src.reporting.pdf_renderer.pio.to_image",
            return_value=png,
        ):
            payload = build_match_analysis_pack(self.bundle)

        with zipfile.ZipFile(BytesIO(payload)) as archive:
            self.assertIn("analysis-summary.json", archive.namelist())
            self.assertIn("analysis-summary.schema.json", archive.namelist())
            summary = json.loads(archive.read("analysis-summary.json"))
            schema = json.loads(
                archive.read("analysis-summary.schema.json")
            )

        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(summary)


if __name__ == "__main__":
    unittest.main()
