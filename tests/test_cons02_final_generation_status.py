from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import json
import unittest
from unittest.mock import patch
import zipfile

import pandas as pd

from src.reporting import pdf_renderer
from src.reporting.download_service import (
    MatchAnalysisDownload,
    MatchAnalysisDownloadStatus,
    _result_message,
    _status_and_base_message,
)
from src.reporting.figure_catalog import MatchReportFigureCatalog
from src.reporting.manifest import REPORT_MANIFEST
from src.reporting.pack_builder import build_match_analysis_pack
from src.reporting.pdf_renderer import MatchReportPdfConfig, render_match_report_pdf
from src.reporting.render_audit import (
    prepare_render_audit,
    summarize_render_audit,
    table_key,
    section_key,
)
from src.reporting.bundle import ReportSectionStatus
from tests.test_report_pdf_renderer import _bundle


def _catalog() -> MatchReportFigureCatalog:
    return MatchReportFigureCatalog(
        manifest_id=REPORT_MANIFEST.id,
        manifest_version=REPORT_MANIFEST.schema_version,
        figures=(),
    )


def _render_summary(bundle, *, patcher=None):
    catalog = _catalog()
    audit = prepare_render_audit(bundle, catalog, REPORT_MANIFEST)
    config = MatchReportPdfConfig(
        enable_brand_font=False,
        render_audit=audit,
    )

    if patcher is None:
        pdf = render_match_report_pdf(bundle, catalog, config=config)
    else:
        with patcher:
            pdf = render_match_report_pdf(bundle, catalog, config=config)

    return pdf, audit, summarize_render_audit(audit.values())


class FinalGenerationStatusTests(unittest.TestCase):
    def test_overview_composition_error_is_required_failure(self):
        bundle = _bundle(
            section_statuses={
                "overview": ReportSectionStatus.GENERATED,
            },
            section_payloads={
                "overview": {
                    "metadata": {},
                    "result": {},
                    "game_profile": {},
                    "data_coverage": {},
                }
            },
        )

        pdf, audit, summary = _render_summary(
            bundle,
            patcher=patch.object(
                pdf_renderer,
                "_overview_section_story",
                side_effect=RuntimeError("synthetic overview internals"),
            ),
        )

        self.assertTrue(pdf.startswith(b"%PDF"))
        overview = audit[section_key("overview")]
        self.assertEqual(overview["data_status"], "generated")
        self.assertEqual(overview["composition_status"], "error")
        self.assertEqual(overview["error_type"], "CompositionError")
        self.assertNotIn("synthetic", overview["error_message"] or "")
        self.assertGreaterEqual(summary["required_sections_failed"], 1)
        self.assertGreater(summary["required_content_failed"], 0)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["status"], "error")

    def test_required_table_composition_error_is_failure(self):
        nodes = pd.DataFrame(
            [
                {
                    "playerName": "Player A",
                    "jersey_number": 8,
                    "pass_sent": 10,
                    "pass_received": 9,
                    "pass_involvement": 19,
                    "minutes": 90.0,
                }
            ]
        )
        bundle = _bundle(
            section_statuses={
                "pass-network": ReportSectionStatus.GENERATED,
            },
            section_payloads={
                "pass-network": {
                    "teams": {
                        "Home FC": {"nodes": nodes},
                        "Away FC": {"nodes": nodes},
                    }
                }
            },
        )

        real_table_story = pdf_renderer._table_story_for_spec

        def flaky(*args, **kwargs):
            table_spec = args[1]
            if table_spec.id == "pass-network-leaders":
                raise RuntimeError("synthetic table internals")
            return real_table_story(*args, **kwargs)

        pdf, audit, summary = _render_summary(
            bundle,
            patcher=patch.object(
                pdf_renderer,
                "_table_story_for_spec",
                side_effect=flaky,
            ),
        )

        self.assertTrue(pdf.startswith(b"%PDF"))
        record = audit[table_key("pass-network", "pass-network-leaders")]
        self.assertEqual(record["data_status"], "generated")
        self.assertEqual(record["composition_status"], "error")
        self.assertTrue(record["required"])
        self.assertGreaterEqual(summary["required_tables_failed"], 1)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["status"], "error")

    def test_optional_composition_error_is_warning(self):
        bundle = _bundle(
            section_statuses={
                "player-highlights": ReportSectionStatus.GENERATED,
            },
            section_payloads={"player-highlights": {}},
        )
        real_section_story = pdf_renderer._section_story

        def flaky(*args, **kwargs):
            section_spec = args[2]
            if section_spec.id == "player-highlights":
                raise RuntimeError("synthetic optional internals")
            return real_section_story(*args, **kwargs)

        _pdf, _audit, summary = _render_summary(
            bundle,
            patcher=patch.object(
                pdf_renderer,
                "_section_story",
                side_effect=flaky,
            ),
        )

        self.assertEqual(summary["required_content_failed"], 0)
        self.assertGreater(summary["optional_content_failed"], 0)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["status"], "warning")

    def test_legitimately_empty_required_content_is_not_failure(self):
        _pdf, audit, summary = _render_summary(_bundle())

        overview = audit[section_key("overview")]
        self.assertEqual(overview["data_status"], "empty")
        self.assertEqual(overview["composition_status"], "empty")
        self.assertEqual(summary["required_content_failed"], 0)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["status"], "generated")

    def test_pack_manifest_is_incomplete_after_required_overview_failure(self):
        bundle = _bundle(
            section_statuses={
                "overview": ReportSectionStatus.GENERATED,
            },
            section_payloads={
                "overview": {
                    "metadata": {},
                    "result": {},
                    "game_profile": {},
                    "data_coverage": {},
                }
            },
        )

        with patch(
            "src.reporting.pack_builder.build_report_figure_catalog",
            return_value=_catalog(),
        ), patch.object(
            pdf_renderer,
            "_overview_section_story",
            side_effect=RuntimeError("synthetic overview internals"),
        ):
            payload = build_match_analysis_pack(
                bundle,
                pdf_config=MatchReportPdfConfig(enable_brand_font=False),
            )

        with zipfile.ZipFile(BytesIO(payload), "r") as archive:
            manifest = json.loads(archive.read("report-manifest.json"))

        generation = manifest["generation"]
        self.assertEqual(generation["status"], "error")
        self.assertFalse(generation["complete"])
        self.assertEqual(generation["pack_kind"], "incomplete-diagnostic")
        self.assertGreater(generation["required_content_failed"], 0)
        overview = next(
            item for item in manifest["sections"] if item["id"] == "overview"
        )
        self.assertEqual(
            overview["generation"]["data_status"],
            "generated",
        )
        self.assertEqual(
            overview["generation"]["composition_status"],
            "error",
        )
        self.assertNotIn(
            "synthetic overview internals",
            json.dumps(manifest),
        )

    def test_ui_status_failure_warning_and_diagnostic_label(self):
        failure, _ = _status_and_base_message(
            {
                "required_content_failed": 1,
                "optional_content_failed": 0,
                "complete": False,
            }
        )
        warning, _ = _status_and_base_message(
            {
                "required_content_failed": 0,
                "optional_content_failed": 1,
                "complete": True,
            }
        )

        self.assertIs(failure, MatchAnalysisDownloadStatus.FAILURE)
        self.assertIs(warning, MatchAnalysisDownloadStatus.WARNING)

        message = _result_message(
            MatchAnalysisDownload(
                filename="sample-match-analysis-pack.zip",
                payload=b"PK",
                status=MatchAnalysisDownloadStatus.FAILURE,
                warnings=("1 required report item(s) failed final generation",),
            )
        )
        self.assertIn("Incomplete diagnostic pack", message)
        self.assertNotIn("RuntimeError", message)


if __name__ == "__main__":
    unittest.main()
