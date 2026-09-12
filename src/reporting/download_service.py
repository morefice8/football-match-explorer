"""Application-neutral bridge from the Dash match store to REPORT-06.

REPORT-07 keeps the UI callback deliberately thin. This module converts
``store-df-match`` into the canonical Full Match report bundle and hands that
same bundle to REPORT-06. No active tab, filter, selection, browser, or
filesystem state is part of the contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from io import StringIO
import json
import logging
from typing import Any, Mapping

import pandas as pd

from src.reporting.bundle import (
    MatchReportBundleConfig,
    build_match_report_data_bundle,
)
from src.reporting.pack_builder import (
    build_match_analysis_pack,
    match_analysis_pack_filename,
)
from src.reporting.figure_export import (
    StaticExportPreflightStatus,
    preflight_match_report_export,
)
from src.reporting.pdf_renderer import MatchReportPdfConfig


logger = logging.getLogger(__name__)


class MatchAnalysisDownloadStatus(str, Enum):
    SUCCESS = "success"
    WARNING = "warning"
    FAILURE = "failure"


@dataclass(frozen=True)
class MatchAnalysisDownload:
    filename: str
    payload: bytes
    status: MatchAnalysisDownloadStatus = (
        MatchAnalysisDownloadStatus.SUCCESS
    )
    message: str = "Match Analysis Pack generated successfully."


class MatchAnalysisDownloadError(RuntimeError):
    """Readable REPORT-07 generation failure."""

    status = MatchAnalysisDownloadStatus.FAILURE


class MatchAnalysisPreflightError(MatchAnalysisDownloadError):
    """Safe UI message for a failed static-export preflight."""

    def __init__(
        self,
        message: str,
        *,
        status: MatchAnalysisDownloadStatus,
    ) -> None:
        super().__init__(message)
        self.status = status


def _parse_match_info(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}

    if isinstance(value, Mapping):
        return dict(value)

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise MatchAnalysisDownloadError(
                "Stored match metadata is not valid JSON."
            ) from exc

        if not isinstance(parsed, dict):
            raise MatchAnalysisDownloadError(
                "Stored match metadata must be a JSON object."
            )
        return parsed

    raise MatchAnalysisDownloadError(
        "Stored match metadata has an unsupported format."
    )


def _read_match_dataframe(
    stored_match_data: Mapping[str, Any],
) -> pd.DataFrame:
    df_json = stored_match_data.get("df")

    if not df_json:
        raise MatchAnalysisDownloadError(
            "Match data is not available. "
            "Load a match before generating the report."
        )

    try:
        frame = pd.read_json(
            StringIO(str(df_json)),
            orient="split",
        )
    except Exception as exc:
        raise MatchAnalysisDownloadError(
            "Stored match data could not be read."
        ) from exc

    if frame.empty:
        raise MatchAnalysisDownloadError(
            "Match data is empty. "
            "Load a match with events before generating the report."
        )

    return frame


def build_match_analysis_download(
    stored_match_data: Mapping[str, Any] | None,
) -> MatchAnalysisDownload:
    """Build one canonical Full Match pack from ``store-df-match``."""

    if not isinstance(stored_match_data, Mapping):
        raise MatchAnalysisDownloadError(
            "Match data is not available. "
            "Load a match before generating the report."
        )

    frame = _read_match_dataframe(stored_match_data)
    match_info = _parse_match_info(
        stored_match_data.get("match_info")
    )

    pdf_config = MatchReportPdfConfig()
    preflight = preflight_match_report_export(pdf_config)
    if not preflight.passed:
        status = (
            MatchAnalysisDownloadStatus.WARNING
            if preflight.status is StaticExportPreflightStatus.WARNING
            else MatchAnalysisDownloadStatus.FAILURE
        )
        raise MatchAnalysisPreflightError(
            preflight.user_message,
            status=status,
        )

    default_config = MatchReportBundleConfig()
    raw_scope = getattr(default_config, "scope", None)
    scope = getattr(raw_scope, "value", raw_scope)
    if scope != "full-match":
        raise MatchAnalysisDownloadError(
            "Canonical report configuration is not Full Match."
        )

    try:
        bundle = build_match_report_data_bundle(
            frame,
            match_info,
        )
    except Exception as exc:
        logger.exception(
            "Could not build the canonical Full Match report data."
        )
        raise MatchAnalysisDownloadError(
            "Could not build the canonical Full Match report data. "
            "Review the application logs, then try again."
        ) from exc

    try:
        payload = build_match_analysis_pack(
            bundle,
            pdf_config=pdf_config,
        )
    except Exception as exc:
        logger.exception("Could not render the Match Analysis Pack.")
        raise MatchAnalysisDownloadError(
            "Could not render the Match Analysis Pack. "
            "Review the application logs, then try again."
        ) from exc

    summary = getattr(payload, "render_summary", {})
    status = MatchAnalysisDownloadStatus.SUCCESS
    message = "Match Analysis Pack generated successfully."
    if summary.get("required_figures_failed", 0):
        status = MatchAnalysisDownloadStatus.FAILURE
        message = (
            "Report incomplete: required plots failed to render. "
            "A diagnostic pack is available; review report-manifest.json and the logs."
        )
    elif summary.get("figures_failed", 0):
        status = MatchAnalysisDownloadStatus.WARNING
        message = (
            "Pack generated with warnings: optional plots failed to render. "
            "Review report-manifest.json for details."
        )

    if hasattr(payload, "getvalue"):
        payload = payload.getvalue()

    if not isinstance(payload, (bytes, bytearray)):
        raise MatchAnalysisDownloadError(
            "The Match Analysis Pack renderer returned an invalid payload."
        )

    payload_bytes = bytes(payload)
    if not payload_bytes:
        raise MatchAnalysisDownloadError(
            "The Match Analysis Pack renderer returned an empty payload."
        )

    try:
        filename = match_analysis_pack_filename(bundle)
    except Exception as exc:
        raise MatchAnalysisDownloadError(
            "Could not determine the Match Analysis Pack filename: "
            f"{exc}"
        ) from exc

    if not filename:
        raise MatchAnalysisDownloadError(
            "The Match Analysis Pack filename is empty."
        )

    return MatchAnalysisDownload(
        filename=str(filename),
        payload=payload_bytes,
        status=status,
        message=message,
    )
