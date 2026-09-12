from __future__ import annotations

import unittest
from unittest.mock import patch

import plotly.graph_objects as go

from src.reporting.figure_export import (
    StaticExportPreflightCode,
    StaticExportPreflightStatus,
    preflight_match_report_export,
)
from src.reporting.pdf_renderer import MatchReportPdfConfig


PNG_BYTES = b"\x89PNG\r\n\x1a\nsynthetic"


class MatchReportExportPreflightTests(unittest.TestCase):
    def test_success_uses_pdf_plotly_configuration(self):
        config = MatchReportPdfConfig(
            image_scale=1.75,
            enable_brand_font=False,
        )

        with patch(
            "src.reporting.figure_export._static_export_diagnostic",
            return_value=(True, None),
        ), patch(
            "src.reporting.figure_export.pio.to_image",
            return_value=PNG_BYTES,
        ) as to_image:
            result = preflight_match_report_export(config)

        self.assertTrue(result.passed)
        self.assertIs(
            result.status,
            StaticExportPreflightStatus.SUCCESS,
        )
        self.assertIs(result.code, StaticExportPreflightCode.READY)
        self.assertIsInstance(to_image.call_args.args[0], go.Figure)
        self.assertEqual(to_image.call_args.kwargs["format"], "png")
        self.assertEqual(to_image.call_args.kwargs["scale"], 1.75)
        self.assertLessEqual(to_image.call_args.kwargs["width"], 100)
        self.assertLessEqual(to_image.call_args.kwargs["height"], 100)

    def test_missing_kaleido_returns_operational_warning(self):
        technical = (
            "Static report export requires Kaleido. "
            "ModuleNotFoundError: No module named 'kaleido'"
        )

        with patch(
            "src.reporting.figure_export._static_export_diagnostic",
            return_value=(False, technical),
        ), patch(
            "src.reporting.figure_export.pio.to_image",
        ) as to_image:
            result = preflight_match_report_export()

        self.assertFalse(result.passed)
        self.assertIs(
            result.status,
            StaticExportPreflightStatus.WARNING,
        )
        self.assertIs(
            result.code,
            StaticExportPreflightCode.KALEIDO_MISSING,
        )
        self.assertIn("pip install --upgrade kaleido", result.user_message)
        self.assertNotIn("ModuleNotFoundError", result.user_message)
        self.assertEqual(result.technical_detail, technical)
        to_image.assert_not_called()

    def test_missing_chrome_returns_plotly_install_command(self):
        error = RuntimeError(
            "Kaleido requires Google Chrome to be installed."
        )

        with patch(
            "src.reporting.figure_export._static_export_diagnostic",
            return_value=(True, None),
        ), patch(
            "src.reporting.figure_export.pio.to_image",
            side_effect=error,
        ):
            result = preflight_match_report_export()

        self.assertFalse(result.passed)
        self.assertIs(
            result.status,
            StaticExportPreflightStatus.WARNING,
        )
        self.assertIs(
            result.code,
            StaticExportPreflightCode.CHROME_MISSING,
        )
        self.assertIn("plotly_get_chrome", result.user_message)
        self.assertNotIn(str(error), result.user_message)
        self.assertIn(str(error), result.technical_detail or "")

    def test_unexpected_renderer_error_is_safe_and_logged(self):
        error = RuntimeError("synthetic renderer internals")

        with patch(
            "src.reporting.figure_export._static_export_diagnostic",
            return_value=(True, None),
        ), patch(
            "src.reporting.figure_export.pio.to_image",
            side_effect=error,
        ), self.assertLogs(
            "src.reporting.figure_export",
            level="ERROR",
        ) as captured:
            result = preflight_match_report_export()

        self.assertFalse(result.passed)
        self.assertIs(
            result.status,
            StaticExportPreflightStatus.FAILURE,
        )
        self.assertIs(
            result.code,
            StaticExportPreflightCode.RENDERER_ERROR,
        )
        self.assertNotIn("synthetic renderer internals", result.user_message)
        self.assertIn(
            "synthetic renderer internals",
            "\n".join(captured.output),
        )


if __name__ == "__main__":
    unittest.main()
