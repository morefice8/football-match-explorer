import ast
from dataclasses import replace
from io import BytesIO
import json
import logging
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile

import dash
from dash import dcc, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from PIL import Image

from src.reporting import download_service as service
from src.reporting.figure_catalog import (
    MatchReportFigureCatalog, ReportFigureArtifact, ReportFigureStatus,
)
from src.reporting.manifest import REPORT_MANIFEST
from src.reporting.pack_builder import build_match_analysis_pack
from tests.test_report_pdf_renderer import _bundle
from tests.test_report_download_integration import stored_match, successful_preflight


class RenderAuditTests(unittest.TestCase):
    def make_pack(self, *, required=True, fail=False, status=ReportFigureStatus.GENERATED):
        section = next(s for s in REPORT_MANIFEST.sections if s.id == "mean-positions")
        figure_spec = replace(section.figures[0], required=required)
        manifest = replace(REPORT_MANIFEST, sections=(replace(section, figures=(figure_spec,)),))
        artifact = ReportFigureArtifact(
            id=figure_spec.id, section_id=section.id, title="Test", variant="summary",
            filename="test.png", width_px=100, height_px=60, status=status,
            figure=go.Figure(), team_name="Home FC", renderer_id="mean-positions",
            source_section_status="generated",
            selection={"selected_id": "7", "selected_name": "Player A", "category": "passing"},
        )
        catalog = MatchReportFigureCatalog(manifest.id, manifest.schema_version, (artifact,))
        output = BytesIO()
        Image.new("RGB", (100, 60), "white").save(output, format="PNG")
        with patch("src.reporting.pack_builder.build_report_figure_catalog", return_value=catalog), patch(
            "src.reporting.pdf_renderer.pio.to_image",
            side_effect=RuntimeError("private renderer trace") if fail else None,
            return_value=output.getvalue(),
        ):
            pack = build_match_analysis_pack(_bundle(), manifest)
        with zipfile.ZipFile(BytesIO(pack)) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(len(archive.namelist()), 15)
            generation = json.loads(archive.read("report-manifest.json"))["generation"]
        return pack, generation

    def callback(self, download):
        source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding='utf-8')
        fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                  and n.name == "generate_match_analysis_pack_download")
        fn.decorator_list = []
        namespace = {
            "dash": dash, "dcc": dcc, "dbc": dbc, "no_update": no_update,
            "logger": logging.getLogger(__name__),
            "build_match_analysis_download": lambda _: download,
            "MatchAnalysisDownloadStatus": service.MatchAnalysisDownloadStatus,
            "MatchAnalysisDownloadError": service.MatchAnalysisDownloadError,
            "MatchAnalysisPreflightError": service.MatchAnalysisPreflightError,
        }
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "callback", "exec"), namespace)
        return namespace[fn.name](1, stored_match())

    def download(self, pack):
        with patch.object(service, "preflight_match_report_export", return_value=successful_preflight()), patch.object(
            service, "build_match_report_data_bundle", return_value=_bundle()
        ), patch.object(service, "build_match_analysis_pack", return_value=pack):
            return service.build_match_analysis_download(stored_match())

    def test_actual_png_error_optional_is_manifest_error_and_ui_warning(self):
        pack, generation = self.make_pack(required=False, fail=True)
        row = generation["artifacts"][0]
        self.assertEqual(row["data_status"], "generated")
        self.assertEqual(row["render_status"], "error")
        self.assertEqual(row["error_type"], "RuntimeError")
        self.assertNotIn("private renderer trace", row["error_message"])
        self.assertEqual(generation["figures_expected"], 1)
        self.assertEqual(generation["figures_generated"], 0)
        self.assertEqual(generation["figures_failed"], 1)
        self.assertEqual(generation["required_figures_failed"], 0)
        download = self.download(pack)
        data, alert = self.callback(download)
        self.assertTrue(data["content"])
        self.assertEqual(alert.color, "warning")
        self.assertIn("optional plots", alert.children)

    def test_actual_png_error_required_is_ui_failure_not_success(self):
        pack, generation = self.make_pack(fail=True)
        self.assertEqual(generation["required_figures_failed"], 1)
        self.assertEqual(generation["status"], "error")
        _, alert = self.callback(self.download(pack))
        self.assertEqual(alert.color, "danger")
        self.assertIn("required plots", alert.children)
        self.assertNotIn("successfully", alert.children)

    def test_success_preserves_artifact_identity_and_selection(self):
        pack, generation = self.make_pack()
        row = generation["artifacts"][0]
        self.assertEqual(row["section_id"], "mean-positions")
        self.assertEqual(row["team_name"], "Home FC")
        self.assertEqual(row["variant"], "summary")
        self.assertEqual(row["selection"]["selected_id"], "7")
        self.assertEqual(row["renderer_id"], "mean-positions")
        self.assertEqual(generation["figures_generated"], 1)
        self.assertEqual(generation["figures_failed"], 0)
        _, alert = self.callback(self.download(pack))
        self.assertEqual(alert.color, "success")

    def test_empty_and_skipped_placeholders_are_not_generated_figures(self):
        for state, count in ((ReportFigureStatus.EMPTY, "figures_empty"),
                             (ReportFigureStatus.SKIPPED, "figures_skipped")):
            with self.subTest(state=state):
                _, generation = self.make_pack(status=state)
                self.assertEqual(generation[count], 1)
                self.assertEqual(generation["figures_generated"], 0)
                self.assertEqual(generation["figures_failed"], 0)

    def test_section_composition_failure_overrides_image_success(self):
        with patch("src.reporting.pdf_renderer._table_payloads", side_effect=RuntimeError("table error")):
            _, generation = self.make_pack()
        self.assertEqual(generation["figures_failed"], 1)

    def test_successful_placeholder_export_does_not_hide_catalog_error(self):
        _, generation = self.make_pack(status=ReportFigureStatus.ERROR)
        self.assertEqual(generation["figures_generated"], 0)
        self.assertEqual(generation["required_figures_failed"], 1)
        self.assertEqual(generation["artifacts"][0]["error_type"], "FigureConstructionError")


if __name__ == "__main__":
    unittest.main()
