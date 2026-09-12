"""End-to-end diagnostics for Match Analysis Pack generation.

REPORT-11 deliberately stays outside Dash. It starts from a raw Opta JSON,
uses the production preprocessing/bundle/figure/pack pipeline, writes a real ZIP
artifact, re-opens every exported file, and returns a structured diagnostic
result suitable for a CLI and automated tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
import base64
import csv
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any, Mapping
import zipfile
import zlib

import pandas as pd

from src import config
from src.data_processing import preprocess
from src.reporting.bundle import build_match_report_data_bundle
from src.reporting.figure_catalog import (
    ReportFigureStatus,
    build_report_figure_catalog,
)
from src.reporting.figure_export import (
    FigureExportStatus,
    export_report_figure_catalog,
    preflight_match_report_export,
)
from src.reporting.manifest import REPORT_MANIFEST
from src.reporting.pack_builder import (
    TABLE_PATHS,
    build_match_analysis_pack,
    match_analysis_pack_filename,
)
from src.reporting.pdf_renderer import MatchReportPdfConfig
from src.utils import mapping_loader


MATCH_JSON_ENV_VAR = "MATCH_REPORT_AUDIT_JSON"
FORBIDDEN_PDF_TEXT = (
    "Figure unavailable",
    "Section composition failed",
    "Traceback (most recent call last)",
    "Traceback",
)


@dataclass(frozen=True)
class AuditStage:
    name: str
    seconds: float
    status: str
    detail: str = ""


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    code: str
    message: str


@dataclass
class MatchReportAuditResult:
    source_path: str
    pack_path: str | None = None
    temporary_output: bool = False
    stages: list[AuditStage] = field(default_factory=list)
    issues: list[AuditIssue] = field(default_factory=list)
    section_statuses: dict[str, str] = field(default_factory=dict)
    plot_statuses: list[dict[str, Any]] = field(default_factory=list)
    artifact_sizes: dict[str, int] = field(default_factory=dict)
    event_count: int = 0
    processed_columns: int = 0
    pack_bytes: int = 0

    @property
    def failed(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    @property
    def exit_code(self) -> int:
        return 1 if self.failed else 0

    def add_issue(self, severity: str, code: str, message: str) -> None:
        self.issues.append(AuditIssue(severity, code, message))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _repo_root() / path


def resolve_match_json_path(
    argument: str | Path | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve CLI argument first, then MATCH_REPORT_AUDIT_JSON."""

    env = os.environ if environ is None else environ
    raw_value = argument or env.get(MATCH_JSON_ENV_VAR)
    if not raw_value:
        raise ValueError(
            "Provide path/to/match.json or set "
            f"{MATCH_JSON_ENV_VAR}."
        )

    path = Path(raw_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Opta JSON not found: {path}")
    return path


def _load_raw_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Opta JSON root must be an object.")
    return payload


def process_raw_opta_match(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the same mapping + preprocessing path used by uploaded matches."""

    raw = _load_raw_json(path)
    event_mapping_path = _resolve_repo_path(config.OPTA_EVENTS_XLSX)
    qualifier_mapping_path = _resolve_repo_path(config.OPTA_QUALIFIERS_JSON)

    event_map = mapping_loader.load_opta_event_mapping(event_mapping_path)
    qualifier_map = mapping_loader.load_opta_qualifier_mapping(
        qualifier_mapping_path
    )
    if not event_map:
        raise RuntimeError(
            f"Opta event mapping could not be loaded: {event_mapping_path}"
        )
    if not qualifier_map:
        raise RuntimeError(
            "Opta qualifier mapping could not be loaded: "
            f"{qualifier_mapping_path}"
        )

    match_info = config.extract_match_info(raw)
    frame, _, _, _ = preprocess.process_opta_events(
        raw,
        event_map,
        qualifier_map,
        match_info,
    )
    if frame is None or frame.empty:
        raise RuntimeError("Opta preprocessing returned no events.")
    if not frame.columns.is_unique:
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise RuntimeError(
            "Processed event schema still contains duplicate columns: "
            + ", ".join(map(str, duplicates))
        )
    return frame, match_info


def _strict_json_loads(text: str, *, label: str) -> Any:
    def reject_constant(value: str):
        raise ValueError(f"Invalid JSON constant {value!r} in {label}")

    return json.loads(text, parse_constant=reject_constant)


def _decode_pdf_stream(raw: bytes, header: bytes) -> bytes | None:
    """Best-effort decode of ReportLab content streams without extra deps."""

    payload = raw.strip(b"\r\n")
    try:
        if b"/ASCII85Decode" in header or b"/A85" in header:
            payload = base64.a85decode(payload, adobe=True)
        if b"/FlateDecode" in header or b"/Fl" in header:
            payload = zlib.decompress(payload)
    except Exception:
        return None
    return payload


def searchable_pdf_text(pdf_bytes: bytes) -> str:
    """Return raw + decoded PDF stream text for deterministic phrase checks.

    The report is produced by ReportLab. Its page streams are normally Flate
    compressed (and may also be ASCII85 encoded), so scanning only the raw PDF
    bytes would miss visible fallback text. This decoder is intentionally small:
    it is not a general PDF parser, only an audit aid for our own generated PDF.
    """

    chunks: list[bytes] = [pdf_bytes]
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", pdf_bytes, re.S):
        header_start = max(0, match.start() - 768)
        header = pdf_bytes[header_start:match.start()]
        decoded = _decode_pdf_stream(match.group(1), header)
        if decoded:
            chunks.append(decoded)
    return "\n".join(chunk.decode("latin-1", errors="ignore") for chunk in chunks)


def validate_pdf_bytes(pdf_bytes: bytes) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    if not pdf_bytes.startswith(b"%PDF"):
        return [
            AuditIssue("error", "pdf-invalid-header", "PDF does not start with %PDF.")
        ]
    if b"%%EOF" not in pdf_bytes[-4096:]:
        issues.append(
            AuditIssue("error", "pdf-missing-eof", "PDF is missing its %%EOF marker.")
        )

    searchable = searchable_pdf_text(pdf_bytes)
    for phrase in FORBIDDEN_PDF_TEXT:
        if phrase.casefold() in searchable.casefold():
            issues.append(
                AuditIssue(
                    "error",
                    "pdf-forbidden-text",
                    f"PDF contains forbidden fallback text: {phrase!r}.",
                )
            )
    return issues


def _required_figure_policy() -> dict[tuple[str, str], bool]:
    return {
        (section.id, figure.id): (
            section.required if figure.required is None else figure.required
        )
        for section in REPORT_MANIFEST.sections
        for figure in section.figures
    }


def validate_pack_file(pack_path: Path) -> tuple[
    list[AuditIssue],
    dict[str, int],
    dict[str, str],
    list[dict[str, Any]],
]:
    """Re-open ZIP, PDF, both JSON files and every CSV table."""

    issues: list[AuditIssue] = []
    sizes: dict[str, int] = {}
    section_statuses: dict[str, str] = {}
    plot_statuses: list[dict[str, Any]] = []

    try:
        with zipfile.ZipFile(pack_path, "r") as archive:
            corrupt = archive.testzip()
            if corrupt:
                issues.append(
                    AuditIssue(
                        "error",
                        "zip-corrupt-entry",
                        f"ZIP CRC check failed for {corrupt}.",
                    )
                )

            names = archive.namelist()
            pdf_names = [name for name in names if name.lower().endswith(".pdf")]
            required_entries = {
                "report-data.json",
                "report-manifest.json",
                *TABLE_PATHS,
            }
            missing = sorted(required_entries - set(names))
            if missing:
                issues.append(
                    AuditIssue(
                        "error",
                        "zip-missing-entry",
                        "Pack is missing: " + ", ".join(missing),
                    )
                )
            if len(pdf_names) != 1:
                issues.append(
                    AuditIssue(
                        "error",
                        "zip-pdf-count",
                        f"Expected exactly one PDF, found {len(pdf_names)}.",
                    )
                )

            for info in archive.infolist():
                sizes[info.filename] = info.file_size

            for json_name in ("report-data.json", "report-manifest.json"):
                if json_name not in names:
                    continue
                text = archive.read(json_name).decode("utf-8")
                try:
                    parsed = _strict_json_loads(text, label=json_name)
                except Exception as exc:
                    issues.append(
                        AuditIssue(
                            "error",
                            "json-invalid",
                            f"{json_name} cannot be parsed: {exc}",
                        )
                    )
                    continue
                if json_name == "report-manifest.json" and isinstance(parsed, dict):
                    generation = parsed.get("generation", {}) or {}
                    section_statuses = {
                        str(key): str(value)
                        for key, value in (
                            generation.get("section_statuses", {}) or {}
                        ).items()
                    }
                    plot_statuses = list(generation.get("artifacts", []) or [])
                    if int(generation.get("required_figures_failed", 0) or 0) > 0:
                        issues.append(
                            AuditIssue(
                                "error",
                                "required-figure-failure",
                                "One or more required report figures failed final PDF rendering.",
                            )
                        )
                    if generation.get("status") == "error":
                        issues.append(
                            AuditIssue(
                                "error",
                                "manifest-generation-error",
                                "report-manifest.json marks generation as error.",
                            )
                        )

            for csv_name in TABLE_PATHS:
                if csv_name not in names:
                    continue
                try:
                    text = archive.read(csv_name).decode("utf-8")
                    rows = list(csv.reader(StringIO(text)))
                except Exception as exc:
                    issues.append(
                        AuditIssue(
                            "error",
                            "csv-invalid",
                            f"{csv_name} cannot be reopened: {exc}",
                        )
                    )
                    continue
                if not rows or not rows[0]:
                    issues.append(
                        AuditIssue(
                            "error",
                            "csv-missing-header",
                            f"{csv_name} has no header row.",
                        )
                    )
                    continue
                if len(rows[0]) != len(set(rows[0])):
                    issues.append(
                        AuditIssue(
                            "error",
                            "csv-duplicate-header",
                            f"{csv_name} has duplicate CSV headers.",
                        )
                    )

            if len(pdf_names) == 1:
                issues.extend(validate_pdf_bytes(archive.read(pdf_names[0])))

    except zipfile.BadZipFile as exc:
        issues.append(AuditIssue("error", "zip-invalid", f"Invalid ZIP: {exc}"))

    return issues, sizes, section_statuses, plot_statuses


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    number = float(value)
    for unit in units:
        if number < 1024 or unit == units[-1]:
            return f"{number:.1f} {unit}" if unit != "B" else f"{int(number)} B"
        number /= 1024
    return f"{value} B"


def run_match_report_audit(
    match_json: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> MatchReportAuditResult:
    source = Path(match_json).expanduser().resolve()
    result = MatchReportAuditResult(source_path=str(source))

    managed_tmp: str | None = None
    if output_dir is None:
        managed_tmp = tempfile.mkdtemp(prefix="match-report-audit-")
        destination = Path(managed_tmp)
        result.temporary_output = True
    else:
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)

    try:
        started = time.perf_counter()
        try:
            frame, match_info = process_raw_opta_match(source)
        except Exception as exc:
            result.stages.append(
                AuditStage("process-opta", time.perf_counter() - started, "FAIL", str(exc))
            )
            result.add_issue("error", "process-opta-failed", str(exc))
            return result
        result.event_count = len(frame)
        result.processed_columns = len(frame.columns)
        result.stages.append(
            AuditStage(
                "process-opta",
                time.perf_counter() - started,
                "OK",
                f"{len(frame)} events, {len(frame.columns)} columns",
            )
        )

        started = time.perf_counter()
        bundle = build_match_report_data_bundle(frame, match_info)
        result.section_statuses = bundle.status_by_section()
        required_sections = {s.id for s in REPORT_MANIFEST.sections if s.required}
        errored_required = sorted(
            section_id
            for section_id, status in result.section_statuses.items()
            if section_id in required_sections and status == "error"
        )
        if errored_required:
            result.add_issue(
                "error",
                "required-section-failure",
                "Required bundle sections failed: " + ", ".join(errored_required),
            )
        result.stages.append(
            AuditStage(
                "build-bundle",
                time.perf_counter() - started,
                "FAIL" if errored_required else "OK",
                ", ".join(
                    f"{state}={sum(value == state for value in result.section_statuses.values())}"
                    for state in ("generated", "empty", "skipped", "error")
                ),
            )
        )

        started = time.perf_counter()
        catalog = build_report_figure_catalog(bundle)
        catalog_errors = [
            artifact
            for artifact in catalog.figures
            if artifact.status is ReportFigureStatus.ERROR
        ]
        policy = _required_figure_policy()
        required_catalog_errors = [
            artifact
            for artifact in catalog_errors
            if policy.get((artifact.section_id, artifact.id), True)
        ]
        if required_catalog_errors:
            result.add_issue(
                "error",
                "required-plot-construction-failure",
                "Required plots failed construction: "
                + ", ".join(
                    f"{a.section_id}/{a.id}" for a in required_catalog_errors
                ),
            )
        elif catalog_errors:
            result.add_issue(
                "warning",
                "optional-plot-construction-failure",
                f"{len(catalog_errors)} optional plot(s) failed construction.",
            )
        result.stages.append(
            AuditStage(
                "build-plots",
                time.perf_counter() - started,
                "FAIL" if required_catalog_errors else "OK",
                f"{len(catalog.figures)} catalog artifacts",
            )
        )

        pdf_config = MatchReportPdfConfig()
        started = time.perf_counter()
        preflight = preflight_match_report_export(pdf_config)
        if not preflight.passed:
            result.stages.append(
                AuditStage(
                    "static-export-preflight",
                    time.perf_counter() - started,
                    "FAIL",
                    preflight.user_message,
                )
            )
            result.add_issue(
                "error",
                "static-export-preflight-failed",
                preflight.user_message,
            )
            return result
        result.stages.append(
            AuditStage(
                "static-export-preflight",
                time.perf_counter() - started,
                "OK",
                preflight.user_message,
            )
        )

        started = time.perf_counter()
        figure_dir = destination / "figures"
        exports = export_report_figure_catalog(
            catalog,
            figure_dir,
            format="png",
            scale=pdf_config.image_scale,
        )
        export_errors = [
            item for item in exports.items if item.status is FigureExportStatus.ERROR
        ]
        if export_errors:
            result.add_issue(
                "error",
                "plot-static-export-failure",
                f"{len(export_errors)} plot(s) failed PNG export.",
            )
        result.stages.append(
            AuditStage(
                "export-plots",
                time.perf_counter() - started,
                "FAIL" if export_errors else "OK",
                f"{len(exports.items) - len(export_errors)}/{len(exports.items)} exported",
            )
        )

        started = time.perf_counter()
        try:
            pack = build_match_analysis_pack(bundle, pdf_config=pdf_config)
        except Exception as exc:
            result.stages.append(
                AuditStage("build-pack", time.perf_counter() - started, "FAIL", str(exc))
            )
            result.add_issue("error", "pack-build-failed", str(exc))
            return result
        result.pack_bytes = len(pack)
        pack_path = destination / match_analysis_pack_filename(bundle)
        pack_path.write_bytes(bytes(pack))
        result.pack_path = str(pack_path)
        result.stages.append(
            AuditStage(
                "build-pack",
                time.perf_counter() - started,
                "OK",
                _format_bytes(len(pack)),
            )
        )

        started = time.perf_counter()
        issues, sizes, final_sections, plot_statuses = validate_pack_file(pack_path)
        result.issues.extend(issues)
        result.artifact_sizes = sizes
        if final_sections:
            result.section_statuses = final_sections
        result.plot_statuses = plot_statuses
        result.stages.append(
            AuditStage(
                "reopen-artifacts",
                time.perf_counter() - started,
                "FAIL" if any(i.severity == "error" for i in issues) else "OK",
                f"{len(sizes)} ZIP entries validated",
            )
        )
        return result
    finally:
        if managed_tmp and result.pack_path is None:
            shutil.rmtree(managed_tmp, ignore_errors=True)


def print_audit_result(result: MatchReportAuditResult) -> None:
    print("MATCH REPORT AUDIT")
    print(f"Source: {result.source_path}")
    print()

    for stage in result.stages:
        detail = f" — {stage.detail}" if stage.detail else ""
        print(f"[{stage.status:4}] {stage.name:<24} {stage.seconds:8.3f}s{detail}")

    if result.section_statuses:
        print("\nSections")
        for section in REPORT_MANIFEST.sections:
            status = result.section_statuses.get(section.id, "missing")
            required = "required" if section.required else "optional"
            print(f"  {section.order:02d}. {section.id:<26} {status:<10} {required}")

    if result.plot_statuses:
        print("\nPlots")
        for row in result.plot_statuses:
            identity = "/".join(
                str(value)
                for value in (
                    row.get("section_id"),
                    row.get("figure_id"),
                    row.get("team_name"),
                    row.get("variant"),
                )
                if value not in (None, "", "summary")
            )
            print(
                f"  {str(row.get('render_status', 'missing')).upper():<10} "
                f"{identity}"
            )

    if result.artifact_sizes:
        print("\nArtifacts")
        for name, size in sorted(result.artifact_sizes.items()):
            print(f"  {_format_bytes(size):>10}  {name}")

    if result.issues:
        print("\nIssues")
        for issue in result.issues:
            print(f"  [{issue.severity.upper()}] {issue.code}: {issue.message}")

    if result.pack_path:
        if result.temporary_output:
            print(f"\nPack validated at temporary path: {result.pack_path}")
            print("Use --output-dir to keep audit artifacts in a stable directory.")
        else:
            print(f"\nPack: {result.pack_path}")

    print("\nRESULT: " + ("FAIL" if result.failed else "PASS"))
