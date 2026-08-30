from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from src.reporting import download_service
from src.reporting.download_service import (
    MatchAnalysisDownloadError,
    build_match_analysis_download,
)


ROOT = Path(__file__).resolve().parents[1]


def stored_match(*, match_info=None):
    frame = pd.DataFrame(
        [
            {
                "eventId": 1,
                "team_name": "Home FC",
                "playerName": "Player A",
                "type_displayName": "Pass",
                "x": 10.0,
                "y": 50.0,
            }
        ]
    )
    payload = {
        "df": frame.to_json(
            orient="split",
            date_format="iso",
        ),
    }
    if match_info is not None:
        payload["match_info"] = json.dumps(match_info)
    return payload


class MatchAnalysisDownloadServiceTests(unittest.TestCase):
    def test_success_uses_one_bundle_for_pack_and_filename(self):
        fake_bundle = object()
        source = stored_match(
            match_info={
                "hteamName": "Home FC",
                "ateamName": "Away FC",
            }
        )

        with patch.object(
            download_service,
            "build_match_report_data_bundle",
            return_value=fake_bundle,
        ) as build_bundle, patch.object(
            download_service,
            "build_match_analysis_pack",
            return_value=b"PK-report-pack",
        ) as build_pack, patch.object(
            download_service,
            "match_analysis_pack_filename",
            return_value="home-fc-vs-away-fc-match-analysis-pack.zip",
        ) as build_filename:
            result = build_match_analysis_download(source)

        self.assertEqual(
            result.filename,
            "home-fc-vs-away-fc-match-analysis-pack.zip",
        )
        self.assertEqual(result.payload, b"PK-report-pack")
        self.assertEqual(build_bundle.call_count, 1)
        self.assertEqual(build_pack.call_count, 1)
        self.assertEqual(build_filename.call_count, 1)

        built_bundle = build_bundle.return_value
        self.assertIs(build_pack.call_args.args[0], built_bundle)
        self.assertIs(build_filename.call_args.args[0], built_bundle)
        self.assertEqual(len(build_bundle.call_args.args), 2)
        self.assertIsInstance(
            build_bundle.call_args.args[0],
            pd.DataFrame,
        )
        self.assertEqual(
            build_bundle.call_args.args[1]["hteamName"],
            "Home FC",
        )

    def test_missing_store_has_readable_error(self):
        with self.assertRaisesRegex(
            MatchAnalysisDownloadError,
            "Load a match",
        ):
            build_match_analysis_download(None)

    def test_missing_dataframe_has_readable_error(self):
        with self.assertRaisesRegex(
            MatchAnalysisDownloadError,
            "Load a match",
        ):
            build_match_analysis_download(
                {"match_info": "{}"}
            )

    def test_malformed_dataframe_has_readable_error(self):
        with self.assertRaisesRegex(
            MatchAnalysisDownloadError,
            "could not be read",
        ):
            build_match_analysis_download(
                {
                    "df": "not-a-split-dataframe",
                    "match_info": "{}",
                }
            )

    def test_renderer_failure_is_readable(self):
        fake_bundle = object()

        with patch.object(
            download_service,
            "build_match_report_data_bundle",
            return_value=fake_bundle,
        ), patch.object(
            download_service,
            "build_match_analysis_pack",
            side_effect=RuntimeError(
                "Kaleido renderer unavailable"
            ),
        ):
            with self.assertRaisesRegex(
                MatchAnalysisDownloadError,
                "Could not render.*Kaleido renderer unavailable",
            ):
                build_match_analysis_download(
                    stored_match(match_info={})
                )

    def test_missing_match_metadata_is_allowed(self):
        fake_bundle = object()

        with patch.object(
            download_service,
            "build_match_report_data_bundle",
            return_value=fake_bundle,
        ) as build_bundle, patch.object(
            download_service,
            "build_match_analysis_pack",
            return_value=b"PK-pack",
        ), patch.object(
            download_service,
            "match_analysis_pack_filename",
            return_value="match-analysis-pack.zip",
        ):
            result = build_match_analysis_download(
                stored_match()
            )

        self.assertEqual(
            result.filename,
            "match-analysis-pack.zip",
        )
        self.assertEqual(
            build_bundle.call_args.args[1],
            {},
        )

    def test_service_has_no_dash_ui_or_filesystem_dependency(self):
        path = (
            ROOT
            / "src"
            / "reporting"
            / "download_service.py"
        )
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(
                    alias.name
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")

        for module in imported:
            self.assertFalse(
                module == "dash"
                or module.startswith("dash.")
                or module == "app"
                or module.startswith("app.")
            )

        for forbidden in (
            "active_tab",
            "callback_context",
            "cross-filter-store",
            "store-def-transition-filter",
            "store-off-transition-filter",
            "store-set-piece-filter",
            "tempfile",
            "write_bytes(",
            "write_text(",
        ):
            self.assertNotIn(forbidden, source)


class MatchAnalysisDashIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_source = (
            ROOT / "app.py"
        ).read_text(encoding="utf-8")
        cls.page_source = (
            ROOT / "pages" / "match_analysis.py"
        ).read_text(encoding="utf-8")

        marker = "# REPORT-07 — Match Analysis Pack download"
        start = cls.app_source.find(marker)
        if start < 0:
            raise AssertionError(
                "REPORT-07 callback marker not found in app.py"
            )
        # The applicator inserts REPORT-07 immediately before the
        # database callback section. Limit assertions to this block.
        end_marker = (
            "# -----------------------------------------------------------------------------\n"
            "# CALLBACKS PER LA PAGINA DEL DATABASE"
        )
        end = cls.app_source.find(end_marker, start)
        cls.callback_source = cls.app_source[
            start:
            end if end >= 0 else None
        ]

    def test_existing_generate_report_button_is_preserved(self):
        self.assertIn(
            "generate-report-button",
            self.page_source,
        )

    def test_layout_has_dedicated_download_and_status(self):
        self.assertIn(
            'dcc.Download(id="match-analysis-pack-download")',
            self.app_source,
        )
        self.assertIn(
            'id="report-generation-status"',
            self.app_source,
        )

    def test_callback_uses_only_click_and_store_as_report_inputs(self):
        block = self.callback_source
        self.assertIn(
            'Input("generate-report-button", "n_clicks")',
            block,
        )
        self.assertIn(
            'State("store-df-match", "data")',
            block,
        )

        for forbidden in (
            'State("url",',
            'Input("url",',
            "active_tab",
            "cross-filter-store",
            "store-def-transition-filter",
            "store-off-transition-filter",
            "store-set-piece-filter",
            "cross-selection-store",
        ):
            self.assertNotIn(forbidden, block)

    def test_running_guard_disables_button_and_restores_label(self):
        block = self.callback_source
        self.assertIn(
            'Output("generate-report-button", "disabled")',
            block,
        )
        self.assertIn(
            'Output("generate-report-button", "children")',
            block,
        )
        self.assertIn('"Generating…"', block)
        self.assertIn('"Generate Report"', block)

    def test_successful_callback_uses_send_bytes(self):
        block = self.callback_source
        self.assertIn(
            "build_match_analysis_download(",
            block,
        )
        self.assertIn("dcc.send_bytes(", block)
        self.assertIn("download.payload", block)
        self.assertIn("download.filename", block)

    def test_no_data_and_renderer_errors_are_presented_as_alerts(self):
        block = self.callback_source
        self.assertIn(
            "Match data is not available",
            block,
        )
        self.assertIn(
            "MatchAnalysisDownloadError",
            block,
        )
        self.assertIn("dbc.Alert(", block)

    def test_legacy_report_store_is_not_executable_when_unused(self):
        executable_lines = [
            line
            for line in self.app_source.splitlines()
            if not line.lstrip().startswith("#")
        ]
        executable = "\n".join(executable_lines)
        self.assertNotIn(
            "report-html-content-store",
            executable,
        )

    def test_send_bytes_smoke(self):
        from dash import dcc

        response = dcc.send_bytes(
            b"PK-report-pack",
            "sample-match-analysis-pack.zip",
        )
        self.assertEqual(
            response["filename"],
            "sample-match-analysis-pack.zip",
        )
        self.assertTrue(response["base64"])
        self.assertTrue(response["content"])


if __name__ == "__main__":
    unittest.main()
