from __future__ import annotations

import ast
from io import BytesIO
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import plotly.graph_objects as go
from PIL import Image as PillowImage

from src.reporting.bundle import (
    MatchReportDataBundle,
    ReportSectionBundle,
    ReportSectionStatus,
)
from src.reporting.figure_catalog import (
    MatchReportFigureCatalog,
    ReportFigureArtifact,
    ReportFigureStatus,
)
from src.reporting.manifest import REPORT_MANIFEST
from src.reporting.models import ReportScope
from src.reporting.pdf_renderer import (
    MatchReportPdfConfig,
    _image_flowable,
    _styles,
    render_match_report_pdf,
    render_match_report_pdf_buffer,
)


ROOT = Path(__file__).resolve().parents[1]


def _page_count(pdf: bytes) -> int:
    return len(re.findall(rb"/Type\s*/Page\b", pdf))


def _bundle(
    *,
    section_statuses: dict[str, ReportSectionStatus] | None = None,
    section_payloads: dict[str, object] | None = None,
    match_info: dict | None = None,
) -> MatchReportDataBundle:
    section_statuses = section_statuses or {}
    section_payloads = section_payloads or {}

    sections = []
    for spec in REPORT_MANIFEST.sections:
        status = section_statuses.get(spec.id, ReportSectionStatus.EMPTY)
        data = section_payloads.get(spec.id, {})
        sections.append(
            ReportSectionBundle(
                id=spec.id,
                status=status,
                data=data,
                error_type=("RuntimeError" if status is ReportSectionStatus.ERROR else None),
                error_message=(
                    "synthetic isolated section failure"
                    if status is ReportSectionStatus.ERROR
                    else None
                ),
            )
        )

    info = {
        "hteamName": "Home FC",
        "ateamName": "Away FC",
        "hteamScore": 2,
        "ateamScore": 1,
        "competitionName": "Synthetic League",
        "game_date": "2026-08-29",
    }
    if match_info:
        info.update(match_info)

    return MatchReportDataBundle(
        manifest_id=REPORT_MANIFEST.id,
        manifest_version=REPORT_MANIFEST.schema_version,
        source_signature="report-05-test",
        scope=ReportScope.FULL_MATCH,
        teams=("Home FC", "Away FC"),
        match_info=info,
        sections=tuple(sections),
    )


def _overview_payload() -> dict:
    return {
        "metadata": {
            "competitionName": "Synthetic League",
            "game_date": "2026-08-29",
        },
        "result": {
            "home_team": "Home FC",
            "away_team": "Away FC",
            "home_score": 2,
            "away_score": 1,
        },
        "scorers": [
            {
                "team_name": "Home FC",
                "scorer": "Home Striker",
                "minute": 12,
                "second": 0,
                "goal_type": "Open Play",
            }
        ],
        "game_profile": {
            "Home FC": {
                "passes": 510,
                "successful_passes": 455,
                "shots": 13,
                "progressive_passes": 38,
            },
            "Away FC": {
                "passes": 390,
                "successful_passes": 320,
                "shots": 8,
                "progressive_passes": 24,
            },
        },
        "data_coverage": {
            "event_rows": 1678,
            "receiver": {"eligible": 420, "resolved": 399},
            "coordinates": {"eligible": 1678, "valid": 1650},
        },
    }


def _catalog(
    figures: tuple[ReportFigureArtifact, ...] = (),
) -> MatchReportFigureCatalog:
    return MatchReportFigureCatalog(
        manifest_id=REPORT_MANIFEST.id,
        manifest_version=REPORT_MANIFEST.schema_version,
        figures=figures,
    )


def _artifact(
    *,
    figure_id: str = "mean-positions-figure",
    section_id: str = "mean-positions",
    team_name: str = "Home FC",
    selection_reason: str | None = None,
) -> ReportFigureArtifact:
    figure = go.Figure()
    figure.add_scatter(x=[0, 1], y=[0, 1])
    return ReportFigureArtifact(
        id=figure_id,
        section_id=section_id,
        title="Synthetic figure",
        variant="summary",
        filename="synthetic.png",
        width_px=1600,
        height_px=900,
        status=ReportFigureStatus.GENERATED,
        figure=figure,
        team_name=team_name,
        is_away=(team_name == "Away FC"),
        renderer_id="synthetic",
        selection_reason=selection_reason,
        source_section_status="generated",
    )


class MatchReportPdfRendererTests(unittest.TestCase):
    def setUp(self):
        self.config = MatchReportPdfConfig(enable_brand_font=False)

    def test_pdf_signature_and_minimum_content(self):
        bundle = _bundle(
            section_statuses={"overview": ReportSectionStatus.GENERATED},
            section_payloads={"overview": _overview_payload()},
        )

        pdf = render_match_report_pdf(
            bundle,
            _catalog(),
            config=self.config,
        )

        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertGreater(len(pdf), 10_000)
        self.assertIn(b"Match Analysis Report", pdf)
        self.assertGreaterEqual(_page_count(pdf), 18)

    def test_bytesio_api_is_rewound_and_contains_pdf(self):
        output = render_match_report_pdf_buffer(
            _bundle(),
            _catalog(),
            config=self.config,
        )

        self.assertIsInstance(output, BytesIO)
        self.assertEqual(output.tell(), 0)
        self.assertEqual(output.read(4), b"%PDF")

    def test_empty_sections_render_as_valid_pdf_placeholders(self):
        pdf = render_match_report_pdf(
            _bundle(),
            _catalog(),
            config=self.config,
        )

        self.assertTrue(pdf.startswith(b"%PDF"))
        # Cover + contents + one explicit page per canonical section.
        self.assertGreaterEqual(_page_count(pdf), 18)

    def test_long_native_table_splits_across_pages(self):
        baseline = render_match_report_pdf(
            _bundle(),
            _catalog(),
            config=self.config,
        )
        baseline_pages = _page_count(baseline)

        moments = [
            {
                "minute": minute,
                "home_formation": "4-3-3",
                "away_formation": "4-2-3-1",
                "reason": "synthetic long-table row",
            }
            for minute in range(220)
        ]
        bundle = _bundle(
            section_statuses={
                "formation-timeline": ReportSectionStatus.GENERATED,
            },
            section_payloads={
                "formation-timeline": {"moments": moments},
            },
        )

        pdf = render_match_report_pdf(
            bundle,
            _catalog(),
            config=self.config,
        )

        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertGreater(_page_count(pdf), baseline_pages)

    def test_failed_data_section_does_not_block_later_sections(self):
        bundle = _bundle(
            section_statuses={
                "pass-network": ReportSectionStatus.ERROR,
                "methodology-appendix": ReportSectionStatus.GENERATED,
            },
            section_payloads={
                "methodology-appendix": {
                    "contract": "Synthetic methodology remains reachable.",
                    "scope": "full-match",
                },
            },
        )

        pdf = render_match_report_pdf(
            bundle,
            _catalog(),
            config=self.config,
        )

        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertGreaterEqual(_page_count(pdf), 18)

    def test_composition_exception_is_isolated_to_one_section(self):
        from src.reporting import pdf_renderer as module

        real_story = module._section_story

        def flaky(bundle, catalog, section_spec, styles, config, manifest):
            if section_spec.id == "pass-network":
                raise RuntimeError("synthetic reportlab composition failure")
            return real_story(
                bundle,
                catalog,
                section_spec,
                styles,
                config,
                manifest,
            )

        with patch.object(module, "_section_story", side_effect=flaky):
            pdf = render_match_report_pdf(
                _bundle(),
                _catalog(),
                config=self.config,
            )

        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertGreaterEqual(_page_count(pdf), 18)

    def test_figure_export_failure_becomes_pdf_placeholder(self):
        artifact = _artifact(
            selection_reason="Highest canonical passing contribution.",
        )
        bundle = _bundle(
            section_statuses={"mean-positions": ReportSectionStatus.GENERATED},
            section_payloads={
                "mean-positions": {
                    "Home FC": {"players": pd.DataFrame(), "summary": {}},
                    "Away FC": {"players": pd.DataFrame(), "summary": {}},
                }
            },
        )

        with patch(
            "src.reporting.pdf_renderer.pio.to_image",
            side_effect=RuntimeError("synthetic Kaleido failure"),
        ):
            pdf = render_match_report_pdf(
                bundle,
                _catalog((artifact,)),
                config=self.config,
            )

        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertGreaterEqual(_page_count(pdf), 18)

    def test_image_scaling_preserves_aspect_ratio(self):
        # Minimal 2x1 PNG; the PDF renderer must scale, never crop/stretch.
        png_buffer = BytesIO()
        PillowImage.new("RGB", (2, 1), "white").save(png_buffer, format="PNG")
        png = png_buffer.getvalue()
        artifact = _artifact()
        styles = _styles(self.config)

        with patch(
            "src.reporting.pdf_renderer.pio.to_image",
            return_value=png,
        ):
            cell = _image_flowable(
                artifact,
                styles,
                self.config,
                max_width=100,
                max_height=100,
            )

        images = [item for item in cell.flowables if item.__class__.__name__ == "Image"]
        self.assertEqual(len(images), 1)
        image = images[0]
        self.assertAlmostEqual(
            image.drawWidth / image.drawHeight,
            image.imageWidth / image.imageHeight,
            places=6,
        )

    def test_renderer_has_no_dash_app_html_or_browser_imports(self):
        source = (ROOT / "src" / "reporting" / "pdf_renderer.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
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
                    or module.startswith("selenium")
                    or module.startswith("playwright")
                    or module.startswith("weasyprint")
                    or module.startswith("pdfkit")
                ):
                    forbidden.append(module)

        self.assertEqual(forbidden, [])

    def test_render_does_not_leave_files_in_working_directory(self):
        bundle = _bundle()
        catalog = _catalog()

        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            workdir = Path(directory)
            before = set(workdir.iterdir())
            try:
                os.chdir(workdir)
                pdf = render_match_report_pdf(
                    bundle,
                    catalog,
                    config=self.config,
                )
                after = set(workdir.iterdir())
            finally:
                os.chdir(original_cwd)

        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
