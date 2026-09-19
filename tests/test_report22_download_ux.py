from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import pandas as pd

from src.reporting import download_service
from src.reporting.download_service import (
    MatchAnalysisDownloadStatus,
    build_match_analysis_download,
    clear_match_analysis_download_cache,
)
from src.reporting.figure_export import (
    StaticExportPreflightCode,
    StaticExportPreflightResult,
    StaticExportPreflightStatus,
)
from src.reporting.render_audit import MatchAnalysisPackBytes


ROOT = Path(__file__).resolve().parents[1]


def successful_preflight():
    return StaticExportPreflightResult(
        status=StaticExportPreflightStatus.SUCCESS,
        code=StaticExportPreflightCode.READY,
        user_message="The report export engine is ready.",
    )


def stored_match(*, x=10.0, match_info=None):
    frame = pd.DataFrame(
        [
            {
                "id": 1001,
                "eventId": 1,
                "periodId": 1,
                "timeMin": 1,
                "timeSec": 2,
                "team_name": "Home FC",
                "playerName": "Player A",
                "type_name": "Pass",
                "outcome": "Successful",
                "x": x,
                "y": 50.0,
                "end_x": 20.0,
                "end_y": 50.0,
            }
        ]
    )
    return {
        "df": frame.to_json(
            orient="split",
            date_format="iso",
        ),
        "match_info": json.dumps(
            match_info
            or {
                "hteamName": "Home FC",
                "ateamName": "Away FC",
            }
        ),
    }


def valid_pack(*, pages=29, plots=38):
    pdf = (
        b"%PDF-1.4\n"
        + (b"<< /Type /Page >>\n" * pages)
        + b"%%EOF\n"
    )
    output = BytesIO()
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr("sample-report.pdf", pdf)
        archive.writestr("analysis-summary.json", "{}")

    return MatchAnalysisPackBytes(
        output.getvalue(),
        {
            "figures_expected": plots,
            "figures_generated": plots,
            "figures_failed": 0,
            "required_figures_failed": 0,
        },
        {
            "figure_catalog": 1.1,
            "png_export": 2.2,
            "pdf": 3.3,
            "csv_json": 4.4,
            "zip": 0.5,
        },
    )


class Report22DownloadUxTests(unittest.TestCase):
    def setUp(self):
        clear_match_analysis_download_cache()

    def _patch_generation(self):
        bundle = SimpleNamespace(
            source_signature="unused-by-mock",
        )
        return (
            bundle,
            patch.object(
                download_service,
                "preflight_match_report_export",
                return_value=successful_preflight(),
            ),
            patch.object(
                download_service,
                "build_match_report_data_bundle",
                return_value=bundle,
            ),
            patch.object(
                download_service,
                "build_match_analysis_pack",
                return_value=valid_pack(),
            ),
            patch.object(
                download_service,
                "match_analysis_pack_filename",
                return_value=(
                    "home-fc-vs-away-fc-match-analysis-pack.zip"
                ),
            ),
        )

    def test_second_identical_click_reuses_bounded_source_signature_cache(self):
        source = stored_match()
        signature = download_service.dataframe_signature(
            pd.read_json(
                BytesIO(source["df"].encode("utf-8")),
                orient="split",
            )
        )

        bundle, preflight, build_bundle, build_pack, filename = (
            self._patch_generation()
        )
        bundle.source_signature = signature

        with preflight, build_bundle as bundle_mock, build_pack as pack_mock, filename:
            first = build_match_analysis_download(source)
            second = build_match_analysis_download(source)

        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(bundle_mock.call_count, 1)
        self.assertEqual(pack_mock.call_count, 1)
        self.assertEqual(first.payload, second.payload)
        self.assertIn("cache: reused", second.message)

    def test_event_change_invalidates_cache(self):
        first_source = stored_match(x=10.0)
        second_source = stored_match(x=11.0)

        signatures = []
        for source in (first_source, second_source):
            frame = pd.read_json(
                BytesIO(source["df"].encode("utf-8")),
                orient="split",
            )
            signatures.append(
                download_service.dataframe_signature(frame)
            )

        bundles = [
            SimpleNamespace(source_signature=signatures[0]),
            SimpleNamespace(source_signature=signatures[1]),
        ]

        with patch.object(
            download_service,
            "preflight_match_report_export",
            return_value=successful_preflight(),
        ), patch.object(
            download_service,
            "build_match_report_data_bundle",
            side_effect=bundles,
        ) as build_bundle, patch.object(
            download_service,
            "build_match_analysis_pack",
            side_effect=[valid_pack(), valid_pack()],
        ), patch.object(
            download_service,
            "match_analysis_pack_filename",
            return_value="sample.zip",
        ):
            first = build_match_analysis_download(first_source)
            second = build_match_analysis_download(second_source)

        self.assertFalse(first.cache_hit)
        self.assertFalse(second.cache_hit)
        self.assertEqual(build_bundle.call_count, 2)

    def test_match_metadata_change_invalidates_cache(self):
        source_a = stored_match(
            match_info={
                "hteamName": "Home FC",
                "ateamName": "Away FC",
                "competitionName": "League A",
            }
        )
        source_b = stored_match(
            match_info={
                "hteamName": "Home FC",
                "ateamName": "Away FC",
                "competitionName": "League B",
            }
        )

        frame = pd.read_json(
            BytesIO(source_a["df"].encode("utf-8")),
            orient="split",
        )
        signature = download_service.dataframe_signature(frame)
        bundles = [
            SimpleNamespace(source_signature=signature),
            SimpleNamespace(source_signature=signature),
        ]

        with patch.object(
            download_service,
            "preflight_match_report_export",
            return_value=successful_preflight(),
        ), patch.object(
            download_service,
            "build_match_report_data_bundle",
            side_effect=bundles,
        ) as build_bundle, patch.object(
            download_service,
            "build_match_analysis_pack",
            side_effect=[valid_pack(), valid_pack()],
        ), patch.object(
            download_service,
            "match_analysis_pack_filename",
            return_value="sample.zip",
        ):
            build_match_analysis_download(source_a)
            build_match_analysis_download(source_b)

        self.assertEqual(build_bundle.call_count, 2)

    def test_final_message_contains_requested_generation_metadata(self):
        source = stored_match()
        frame = pd.read_json(
            BytesIO(source["df"].encode("utf-8")),
            orient="split",
        )
        signature = download_service.dataframe_signature(frame)
        bundle = SimpleNamespace(source_signature=signature)

        with patch.object(
            download_service,
            "preflight_match_report_export",
            return_value=successful_preflight(),
        ), patch.object(
            download_service,
            "build_match_report_data_bundle",
            return_value=bundle,
        ), patch.object(
            download_service,
            "build_match_analysis_pack",
            return_value=valid_pack(),
        ), patch.object(
            download_service,
            "match_analysis_pack_filename",
            return_value="sample-match-analysis-pack.zip",
        ):
            result = build_match_analysis_download(source)

        self.assertEqual(
            result.status,
            MatchAnalysisDownloadStatus.SUCCESS,
        )
        self.assertEqual(result.page_count, 29)
        self.assertEqual(result.plots_generated, 38)
        self.assertEqual(result.warnings, ())
        self.assertGreater(result.zip_size_bytes, 0)

        for expected in (
            "sample-match-analysis-pack.zip",
            "29 pages",
            "38 plots generated",
            "warnings: none",
            "ZIP:",
        ):
            self.assertIn(expected, result.message)

    def test_service_logs_all_requested_phase_names(self):
        source = stored_match()
        frame = pd.read_json(
            BytesIO(source["df"].encode("utf-8")),
            orient="split",
        )
        signature = download_service.dataframe_signature(frame)
        bundle = SimpleNamespace(source_signature=signature)

        with patch.object(
            download_service,
            "preflight_match_report_export",
            return_value=successful_preflight(),
        ), patch.object(
            download_service,
            "build_match_report_data_bundle",
            return_value=bundle,
        ), patch.object(
            download_service,
            "build_match_analysis_pack",
            return_value=valid_pack(),
        ), patch.object(
            download_service,
            "match_analysis_pack_filename",
            return_value="sample.zip",
        ), self.assertLogs(
            "src.reporting.performance",
            level="INFO",
        ) as captured:
            result = build_match_analysis_download(source)

        log = "\n".join(captured.output)
        for phase in (
            "bundle=",
            "figure_catalog=",
            "png_export=",
            "pdf=",
            "csv_json=",
            "zip=",
        ):
            self.assertIn(phase, log)

        self.assertEqual(
            set(result.timings),
            {
                "bundle",
                "figure_catalog",
                "png_export",
                "pdf",
                "csv_json",
                "zip",
                "total",
            },
        )

    def test_ui_copy_and_callback_remain_filter_independent(self):
        page_source = (
            ROOT / "pages" / "match_analysis.py"
        ).read_text(encoding="utf-8")
        app_source = (
            ROOT / "app.py"
        ).read_text(encoding="utf-8")

        self.assertIn("Download Analysis Pack", page_source)
        self.assertIn(
            "Generating the complete report. "
            "This may take 30–60 seconds.",
            app_source,
        )

        marker = "# REPORT-07 — Match Analysis Pack download"
        start = app_source.index(marker)
        end = app_source.index(
            "# -----------------------------------------------------------------------------\n"
            "# CALLBACKS PER LA PAGINA DEL DATABASE",
            start,
        )
        callback = app_source[start:end]

        self.assertIn(
            'Input("generate-report-button", "n_clicks")',
            callback,
        )
        self.assertIn(
            'State("store-df-match", "data")',
            callback,
        )

        for forbidden in (
            "cross-filter-store",
            "store-def-transition-filter",
            "store-off-transition-filter",
            "store-set-piece-filter",
            "cross-selection-store",
            "active_tab",
        ):
            self.assertNotIn(forbidden, callback)

    def test_pack_builder_and_pdf_renderer_expose_phase_timing_hooks(self):
        pack_source = (
            ROOT / "src" / "reporting" / "pack_builder.py"
        ).read_text(encoding="utf-8")
        pdf_source = (
            ROOT / "src" / "reporting" / "pdf_renderer.py"
        ).read_text(encoding="utf-8")

        for phase in (
            '"figure_catalog"',
            '"png_export"',
            '"pdf"',
            '"csv_json"',
            '"zip"',
        ):
            self.assertIn(phase, pack_source)

        self.assertIn("timing_sink", pdf_source)
        self.assertIn("time.perf_counter()", pdf_source)


if __name__ == "__main__":
    unittest.main()
