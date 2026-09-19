from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image, ImageDraw
import plotly.graph_objects as go
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas

from src.reporting.bundle import (
    ReportSectionStatus,
    build_match_report_data_bundle,
)
from src.reporting.figure_catalog import RendererRegistry
from src.reporting.pack_builder import build_match_analysis_pack
from src.reporting.pdf_visual_regression import (
    PdfVisualAuditConfig,
    poppler_available,
    run_pdf_visual_regression,
)
from tests.test_report11_real_shaped_fixture import _processed_fixture


def _visual_png() -> bytes:
    image = Image.new("RGB", (480, 270), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 471, 261), outline="black", width=5)
    draw.line((240, 8, 240, 261), fill="black", width=4)
    draw.ellipse((205, 100, 275, 170), outline="black", width=4)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _dummy_renderer(*args, **kwargs):
    return go.Figure()


def _replace_section(bundle, section_id, *, data=None, status=None):
    return replace(
        bundle,
        sections=tuple(
            replace(
                section,
                data=section.data if data is None else data,
                status=section.status if status is None else status,
            )
            if section.id == section_id
            else section
            for section in bundle.sections
        ),
    )


def _zero_zero(bundle):
    overview = deepcopy(bundle.section("overview").data)
    result = dict(overview.get("result", {}))
    result["home_score"] = 0
    result["away_score"] = 0
    overview["result"] = result
    overview["scorers"] = []
    overview["goal_origins"] = []
    return _replace_section(bundle, "overview", data=overview)


def _without_reliable_carries(bundle):
    overview = deepcopy(bundle.section("overview").data)
    coverage = dict(overview.get("data_coverage", {}))
    carries = {}
    for team in bundle.teams:
        carries[team] = {
            "eligible": 0,
            "resolved": 0,
            "coverage_pct": 0.0,
        }
    coverage["final_third_carries"] = carries
    overview["data_coverage"] = coverage
    bundle = _replace_section(bundle, "overview", data=overview)

    final_third = deepcopy(
        bundle.section("final-third-entries").data
    )
    teams = final_third.get("teams", {})
    if isinstance(teams, dict):
        for payload in teams.values():
            if not isinstance(payload, dict):
                continue
            stats = dict(payload.get("stats", {}))
            for key in list(stats):
                if "carry" in str(key).casefold():
                    value = stats[key]
                    stats[key] = [] if isinstance(value, list) else 0
            payload["stats"] = stats
    return _replace_section(
        bundle,
        "final-third-entries",
        data=final_third,
    )


def _optional_section_empty(bundle):
    return _replace_section(
        bundle,
        "player-highlights",
        data={},
        status=ReportSectionStatus.EMPTY,
    )


def _red_card_many_shape_changes(bundle):
    timeline = deepcopy(
        bundle.section("formation-timeline").data
    )
    moments = list(timeline.get("moments", []))
    if not moments:
        return bundle

    base = deepcopy(moments[0])
    additions = []
    shapes = (
        ("4-3-3", "3-4-2-1"),
        ("4-2-3-1", "3-4-2-1"),
        ("3-5-2", "4-4-2"),
        ("4-4-2", "4-4-1"),
        ("3-4-3", "4-4-1"),
        ("4-3-3", "4-4-1"),
        ("5-3-2", "4-4-1"),
    )

    for index, minute in enumerate(
        (12, 24, 38, 51, 63, 76, 86)
    ):
        moment = deepcopy(base)
        moment["time_seconds"] = minute * 60
        moment["time_label"] = f"{minute}′"
        moment["home_formation_name"] = shapes[index][0]
        moment["away_formation_name"] = shapes[index][1]
        events = [
            {
                "kind": "formation_change",
                "team": "home" if index % 2 == 0 else "away",
                "description": (
                    f"Synthetic tactical change {index + 1}"
                ),
                "event_time_label": f"{minute}′",
            }
        ]
        if minute == 63:
            events.append(
                {
                    "kind": "dismissal",
                    "team": "away",
                    "description": (
                        "Red card · synthetic dismissal"
                    ),
                    "event_time_label": "63′",
                }
            )
        moment["events"] = events
        additions.append(moment)

    timeline["moments"] = sorted(
        moments + additions,
        key=lambda item: int(
            item.get("time_seconds", 0) or 0
        ),
    )
    return _replace_section(
        bundle,
        "formation-timeline",
        data=timeline,
    )


def _pdf_and_summary(bundle):
    png = _visual_png()
    with patch.object(
        RendererRegistry,
        "resolve",
        return_value=_dummy_renderer,
    ), patch(
        "src.reporting.pdf_renderer.pio.to_image",
        return_value=png,
    ):
        pack = build_match_analysis_pack(bundle)

    with zipfile.ZipFile(BytesIO(pack)) as archive:
        pdf_name = next(
            name
            for name in archive.namelist()
            if name.lower().endswith(".pdf")
        )
        pdf_bytes = archive.read(pdf_name)
        summary = json.loads(
            archive.read("analysis-summary.json")
        )
    return pdf_bytes, summary


def _synthetic_pdf(*lines: str) -> bytes:
    output = BytesIO()
    writer = canvas.Canvas(
        output,
        pagesize=landscape(A4),
    )
    for line in lines:
        writer.setFont("Helvetica", 10)
        writer.drawString(40, 520, line)
        writer.showPage()
    writer.save()
    return output.getvalue()


@unittest.skipUnless(
    poppler_available(),
    "REPORT-21 requires Poppler pdftoppm and pdftotext",
)
class Report21PdfVisualRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame, match_info = _processed_fixture()
        cls.base_bundle = build_match_report_data_bundle(
            frame,
            match_info,
        )

    def test_visual_regression_matrix(self):
        cases = {
            "full_match": self.base_bundle,
            "zero_zero": _zero_zero(self.base_bundle),
            "no_reliable_carries": _without_reliable_carries(
                self.base_bundle
            ),
            "optional_section_empty": _optional_section_empty(
                self.base_bundle
            ),
            "red_card_many_shape_changes": (
                _red_card_many_shape_changes(
                    self.base_bundle
                )
            ),
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            for name, bundle in cases.items():
                with self.subTest(case=name):
                    pdf, summary = _pdf_and_summary(bundle)
                    snapshot_dir = root / name
                    result = run_pdf_visual_regression(
                        pdf,
                        analysis_summary=summary,
                        snapshot_dir=snapshot_dir,
                        config=PdfVisualAuditConfig(
                            min_pages=24,
                            max_pages=32,
                            save_snapshots=(
                                name == "full_match"
                            ),
                        ),
                    )

                    errors = [
                        issue
                        for issue in result.issues
                        if issue.severity == "error"
                    ]
                    self.assertFalse(
                        errors,
                        msg="\n".join(
                            f"{issue.code}: {issue.message}"
                            for issue in errors
                        ),
                    )

                    if name == "full_match":
                        self.assertEqual(
                            set(result.snapshots),
                            {
                                "cover",
                                "overview",
                                "pitch",
                                "transitions",
                                "player-highlights",
                            },
                        )
                        saved = list(snapshot_dir.glob("*.png"))
                        self.assertEqual(len(saved), 5)

    def test_technical_plot_placeholder_fails(self):
        pdf = _synthetic_pdf(
            "Figure unavailable: synthetic renderer failure"
        )
        result = run_pdf_visual_regression(
            pdf,
            config=PdfVisualAuditConfig(
                min_pages=1,
                max_pages=5,
                require_headers=False,
                save_snapshots=False,
            ),
        )
        self.assertTrue(
            any(
                issue.code == "technical-placeholder-visible"
                for issue in result.issues
            )
        )
        self.assertTrue(result.failed)

    def test_nearly_empty_table_page_fails(self):
        pdf = _synthetic_pdf(
            "Home-Away summary"
        )
        result = run_pdf_visual_regression(
            pdf,
            config=PdfVisualAuditConfig(
                min_pages=1,
                max_pages=5,
                require_headers=False,
                save_snapshots=False,
            ),
        )
        self.assertTrue(
            any(
                issue.code == "nearly-empty-table-page"
                for issue in result.issues
            )
        )
        self.assertTrue(result.failed)


if __name__ == "__main__":
    unittest.main()
