"""Application-neutral bridge from the Dash match store to the Match Analysis Pack.

REPORT-22 keeps the UI callback deliberately thin, exposes generation metadata
for the final UX message, records phase timings, and reuses an identical pack
from a small bounded in-process cache keyed by the canonical source signature.

No active tab, filter, selection, browser state, or filesystem state is part of
the report-generation contract.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
from io import BytesIO, StringIO
import json
import logging
import os
import re
from threading import RLock
import time
from typing import Any, Mapping
import zipfile

import pandas as pd

from src.reporting.bundle import (
    MatchReportBundleConfig,
    build_match_report_data_bundle,
)
from src.reporting.figure_export import (
    StaticExportPreflightStatus,
    preflight_match_report_export,
)
from src.reporting.manifest import REPORT_MANIFEST
from src.reporting.pack_builder import (
    PACK_SCHEMA_VERSION,
    build_match_analysis_pack,
    match_analysis_pack_filename,
)
from src.reporting.pdf_renderer import MatchReportPdfConfig
from src.utils.derived_cache import dataframe_signature


logger = logging.getLogger(__name__)
performance_logger = logging.getLogger("src.reporting.performance")


DOWNLOAD_CACHE_MAX_ENTRIES = max(
    int(os.getenv("MATCH_ANALYSIS_DOWNLOAD_CACHE_MAX_ENTRIES", "3")),
    1,
)

_DOWNLOAD_CACHE_VERSION = "REPORT-22-v1"
_DOWNLOAD_CACHE: OrderedDict[tuple[str, ...], "MatchAnalysisDownload"] = (
    OrderedDict()
)
_DOWNLOAD_CACHE_LOCK = RLock()
_DOWNLOAD_KEY_LOCKS: dict[tuple[str, ...], RLock] = {}


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
    duration_seconds: float = 0.0
    page_count: int = 0
    plots_generated: int = 0
    warnings: tuple[str, ...] = ()
    zip_size_bytes: int = 0
    source_signature: str = ""
    cache_hit: bool = False
    timings: Mapping[str, float] = field(default_factory=dict)


class MatchAnalysisDownloadError(RuntimeError):
    """Readable report-generation failure."""

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


def _metadata_signature(match_info: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(match_info or {}),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=12).hexdigest()


def _cache_key(
    source_signature: str,
    match_info: Mapping[str, Any],
) -> tuple[str, ...]:
    return (
        str(source_signature),
        _metadata_signature(match_info),
        str(PACK_SCHEMA_VERSION),
        str(REPORT_MANIFEST.schema_version),
        _DOWNLOAD_CACHE_VERSION,
    )


def clear_match_analysis_download_cache() -> None:
    with _DOWNLOAD_CACHE_LOCK:
        _DOWNLOAD_CACHE.clear()
        _DOWNLOAD_KEY_LOCKS.clear()


def match_analysis_download_cache_info() -> dict[str, int]:
    with _DOWNLOAD_CACHE_LOCK:
        return {
            "entries": len(_DOWNLOAD_CACHE),
            "max_entries": DOWNLOAD_CACHE_MAX_ENTRIES,
        }


def _cache_get(
    key: tuple[str, ...],
) -> MatchAnalysisDownload | None:
    with _DOWNLOAD_CACHE_LOCK:
        cached = _DOWNLOAD_CACHE.pop(key, None)
        if cached is None:
            return None
        _DOWNLOAD_CACHE[key] = cached
        return cached


def _cache_put(
    key: tuple[str, ...],
    value: MatchAnalysisDownload,
) -> None:
    with _DOWNLOAD_CACHE_LOCK:
        _DOWNLOAD_CACHE.pop(key, None)
        _DOWNLOAD_CACHE[key] = value
        while len(_DOWNLOAD_CACHE) > DOWNLOAD_CACHE_MAX_ENTRIES:
            _DOWNLOAD_CACHE.popitem(last=False)


def _key_lock(key: tuple[str, ...]) -> RLock:
    with _DOWNLOAD_CACHE_LOCK:
        return _DOWNLOAD_KEY_LOCKS.setdefault(key, RLock())


def _release_key_lock(key: tuple[str, ...]) -> None:
    with _DOWNLOAD_CACHE_LOCK:
        _DOWNLOAD_KEY_LOCKS.pop(key, None)


def _format_bytes(value: int) -> str:
    number = float(max(int(value), 0))
    for unit in ("B", "KiB", "MiB", "GiB"):
        if number < 1024 or unit == "GiB":
            if unit == "B":
                return f"{int(number)} B"
            return f"{number:.1f} {unit}"
        number /= 1024
    return f"{int(value)} B"


def _page_count_from_pack(payload: bytes) -> int:
    try:
        with zipfile.ZipFile(BytesIO(payload), "r") as archive:
            pdf_names = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".pdf")
            ]
            if len(pdf_names) != 1:
                return 0
            pdf = archive.read(pdf_names[0])
    except (OSError, zipfile.BadZipFile, KeyError):
        return 0

    return len(re.findall(rb"/Type\s*/Page\b", pdf))


def _valid_zip_payload(payload: bytes) -> bool:
    try:
        with zipfile.ZipFile(BytesIO(payload), "r") as archive:
            return archive.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def _warning_messages(
    summary: Mapping[str, Any],
) -> tuple[str, ...]:
    messages: list[str] = []

    required_failed = int(
        summary.get("required_figures_failed", 0) or 0
    )
    optional_failed = max(
        int(summary.get("figures_failed", 0) or 0)
        - required_failed,
        0,
    )

    if required_failed:
        messages.append(
            f"{required_failed} required plot(s) failed to render"
        )
    if optional_failed:
        messages.append(
            f"{optional_failed} optional plot(s) failed to render"
        )

    return tuple(messages)


def _status_and_base_message(
    summary: Mapping[str, Any],
) -> tuple[MatchAnalysisDownloadStatus, str]:
    if int(summary.get("required_figures_failed", 0) or 0):
        return (
            MatchAnalysisDownloadStatus.FAILURE,
            "Report incomplete: required plots failed to render.",
        )

    if int(summary.get("figures_failed", 0) or 0):
        return (
            MatchAnalysisDownloadStatus.WARNING,
            "Pack generated with warnings.",
        )

    return (
        MatchAnalysisDownloadStatus.SUCCESS,
        "Match Analysis Pack generated successfully.",
    )


def _result_message(
    result: MatchAnalysisDownload,
) -> str:
    warnings = (
        "; ".join(result.warnings)
        if result.warnings
        else "none"
    )
    cache_note = " · cache: reused" if result.cache_hit else ""

    return (
        f"{result.filename} · "
        f"{result.duration_seconds:.1f}s{cache_note} · "
        f"{result.page_count} pages · "
        f"{result.plots_generated} plots generated · "
        f"warnings: {warnings} · "
        f"ZIP: {_format_bytes(result.zip_size_bytes)}"
    )


def _with_request_duration(
    cached: MatchAnalysisDownload,
    elapsed: float,
) -> MatchAnalysisDownload:
    result = replace(
        cached,
        duration_seconds=float(elapsed),
        cache_hit=True,
    )
    return replace(
        result,
        message=_result_message(result),
    )


def _log_timing(
    *,
    source_signature: str,
    timings: Mapping[str, float],
    cache_hit: bool,
) -> None:
    performance_logger.info(
        (
            "REPORT-22 timing source_signature=%s "
            "bundle=%.3fs figure_catalog=%.3fs "
            "png_export=%.3fs pdf=%.3fs "
            "csv_json=%.3fs zip=%.3fs total=%.3fs "
            "cache_hit=%d"
        ),
        source_signature,
        float(timings.get("bundle", 0.0) or 0.0),
        float(timings.get("figure_catalog", 0.0) or 0.0),
        float(timings.get("png_export", 0.0) or 0.0),
        float(timings.get("pdf", 0.0) or 0.0),
        float(timings.get("csv_json", 0.0) or 0.0),
        float(timings.get("zip", 0.0) or 0.0),
        float(timings.get("total", 0.0) or 0.0),
        1 if cache_hit else 0,
    )


def _generate_uncached(
    frame: pd.DataFrame,
    match_info: Mapping[str, Any],
    *,
    source_signature: str,
    request_started: float,
) -> MatchAnalysisDownload:
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

    bundle_started = time.perf_counter()
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
    bundle_seconds = time.perf_counter() - bundle_started

    bundle_signature = str(
        getattr(bundle, "source_signature", "")
        or source_signature
    )
    if bundle_signature != source_signature:
        raise MatchAnalysisDownloadError(
            "Source signature changed during report generation. "
            "Reload the match and try again."
        )

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

    summary = dict(
        getattr(payload, "render_summary", {})
        or {}
    )
    pack_timings = dict(
        getattr(payload, "timings", {})
        or {}
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

    status, _ = _status_and_base_message(summary)
    warnings = _warning_messages(summary)
    elapsed = time.perf_counter() - request_started

    timings = {
        "bundle": float(bundle_seconds),
        "figure_catalog": float(
            pack_timings.get("figure_catalog", 0.0) or 0.0
        ),
        "png_export": float(
            pack_timings.get("png_export", 0.0) or 0.0
        ),
        "pdf": float(
            pack_timings.get("pdf", 0.0) or 0.0
        ),
        "csv_json": float(
            pack_timings.get("csv_json", 0.0) or 0.0
        ),
        "zip": float(
            pack_timings.get("zip", 0.0) or 0.0
        ),
        "total": float(elapsed),
    }

    result = MatchAnalysisDownload(
        filename=str(filename),
        payload=payload_bytes,
        status=status,
        duration_seconds=float(elapsed),
        page_count=_page_count_from_pack(payload_bytes),
        plots_generated=int(
            summary.get("figures_generated", 0) or 0
        ),
        warnings=warnings,
        zip_size_bytes=len(payload_bytes),
        source_signature=source_signature,
        cache_hit=False,
        timings=timings,
    )
    result = replace(
        result,
        message=_result_message(result),
    )

    _log_timing(
        source_signature=source_signature,
        timings=timings,
        cache_hit=False,
    )
    return result


def build_match_analysis_download(
    stored_match_data: Mapping[str, Any] | None,
) -> MatchAnalysisDownload:
    """Build or reuse one canonical Full Match Analysis Pack.

    The only source is ``store-df-match``. UI tab/filter state never
    participates in report generation or cache identity.
    """

    request_started = time.perf_counter()

    if not isinstance(stored_match_data, Mapping):
        raise MatchAnalysisDownloadError(
            "Match data is not available. "
            "Load a match before generating the report."
        )

    frame = _read_match_dataframe(stored_match_data)
    match_info = _parse_match_info(
        stored_match_data.get("match_info")
    )

    source_signature = dataframe_signature(frame)
    key = _cache_key(source_signature, match_info)

    cached = _cache_get(key)
    if cached is not None:
        elapsed = time.perf_counter() - request_started
        result = _with_request_duration(cached, elapsed)
        _log_timing(
            source_signature=source_signature,
            timings={"total": elapsed},
            cache_hit=True,
        )
        return result

    lock = _key_lock(key)
    try:
        with lock:
            cached = _cache_get(key)
            if cached is not None:
                elapsed = time.perf_counter() - request_started
                result = _with_request_duration(cached, elapsed)
                _log_timing(
                    source_signature=source_signature,
                    timings={"total": elapsed},
                    cache_hit=True,
                )
                return result

            result = _generate_uncached(
                frame,
                match_info,
                source_signature=source_signature,
                request_started=request_started,
            )

            # A required-render failure can be transient, so never cache it.
            if (
                result.status is not MatchAnalysisDownloadStatus.FAILURE
                and _valid_zip_payload(result.payload)
            ):
                _cache_put(key, result)

            return result
    finally:
        _release_key_lock(key)
