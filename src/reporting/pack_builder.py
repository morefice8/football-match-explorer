"""In-memory Match Analysis Pack builder.

REPORT-06 packages the already-built neutral report bundle, the REPORT-04
figure catalog and the REPORT-05 PDF into one compressed ZIP. It deliberately
does not calculate football metrics, touch Dash/app.py, or persist temp files.
"""

from __future__ import annotations

from io import BytesIO, StringIO
import json
import re
import unicodedata
import zipfile
from typing import Any, Mapping, Sequence

import pandas as pd

from src.reporting.bundle import normalize_for_json
from src.reporting.figure_catalog import (
    FigureCatalogConfig,
    build_report_figure_catalog,
)
from src.reporting.manifest import REPORT_MANIFEST
from src.reporting.models import ReportManifest
from src.reporting.pdf_renderer import (
    MatchReportPdfConfig,
    render_match_report_pdf,
)


PACK_SCHEMA_VERSION = "1.0"
MAX_RECOMMENDED_PACK_BYTES = 50 * 1024 * 1024

TABLE_PATHS: tuple[str, ...] = (
    "tables/match-comparison.csv",
    "tables/data-coverage.csv",
    "tables/pass-network-connections.csv",
    "tables/progressive-passers.csv",
    "tables/final-third-entries.csv",
    "tables/cross-routes.csv",
    "tables/buildup-summary.csv",
    "tables/defensive-transitions.csv",
    "tables/offensive-transitions.csv",
    "tables/restarts.csv",
    "tables/player-rankings.csv",
    "tables/event-explorer.csv",
)

_ALLOWED_SECTION_STATUSES = {
    "generated",
    "empty",
    "skipped",
    "error",
}


def _safe_slug(
    value: Any,
    *,
    fallback: str,
    max_length: int = 48,
) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    text = text[:max_length].rstrip("-")
    return text or fallback


def match_slug(bundle) -> str:
    teams = tuple(getattr(bundle, "teams", ()) or ())
    home = teams[0] if len(teams) > 0 else "home"
    away = teams[1] if len(teams) > 1 else "away"
    return (
        f"{_safe_slug(home, fallback='home')}"
        f"-vs-{_safe_slug(away, fallback='away')}"
    )


def match_report_pdf_filename(bundle) -> str:
    return f"{match_slug(bundle)}-report.pdf"


def match_analysis_pack_filename(bundle) -> str:
    return f"{match_slug(bundle)}-match-analysis-pack.zip"


def _section(bundle, section_id: str):
    try:
        return bundle.section(section_id)
    except Exception:
        return None


def _section_data(bundle, section_id: str) -> Any:
    section = _section(bundle, section_id)
    if section is None:
        return {}
    return getattr(section, "data", None) or {}


def _as_frame(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()

    if isinstance(value, pd.DataFrame):
        frame = value.copy()
        if (
            not isinstance(frame.index, pd.RangeIndex)
            or frame.index.name is not None
        ):
            index_name = frame.index.name or "index"
            if index_name in frame.columns:
                index_name = "_index"
            frame = frame.reset_index(names=index_name)
        return frame

    if isinstance(value, pd.Series):
        return value.to_frame().T.reset_index(drop=True)

    if isinstance(value, Mapping):
        return pd.DataFrame([dict(value)])

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        if not value:
            return pd.DataFrame()
        if all(isinstance(item, Mapping) for item in value):
            return pd.DataFrame([dict(item) for item in value])

    return pd.DataFrame()


def _with_report_team(
    frame: pd.DataFrame,
    team_name: str,
) -> pd.DataFrame:
    frame = _as_frame(frame)
    if frame.empty:
        return frame

    if "report_team" in frame.columns:
        frame["report_team"] = frame["report_team"].fillna(team_name)
    else:
        frame.insert(0, "report_team", team_name)

    if "team_name" not in frame.columns:
        frame.insert(1, "team_name", team_name)

    return frame


def _with_source(
    frame: pd.DataFrame,
    source_dataset: str,
) -> pd.DataFrame:
    frame = _as_frame(frame)
    if frame.empty:
        return frame

    if "source_dataset" in frame.columns:
        frame["source_dataset"] = frame["source_dataset"].fillna(
            source_dataset
        )
    else:
        frame.insert(0, "source_dataset", source_dataset)
    return frame


def _concat(
    frames: Sequence[pd.DataFrame],
    *,
    required_columns: Sequence[str],
) -> pd.DataFrame:
    usable = [
        frame
        for frame in frames
        if isinstance(frame, pd.DataFrame) and not frame.empty
    ]
    if not usable:
        return pd.DataFrame(columns=list(required_columns))

    result = pd.concat(
        usable,
        ignore_index=True,
        sort=False,
    )
    for column in reversed(tuple(required_columns)):
        if column not in result.columns:
            result.insert(0, column, None)
    return result


def _strict_json_text(value: Any) -> str:
    return json.dumps(
        normalize_for_json(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _csv_cell(value: Any) -> Any:
    normalized = normalize_for_json(value)
    if normalized is None:
        return ""
    if isinstance(normalized, (dict, list)):
        return _strict_json_text(normalized)
    return normalized


def _csv_bytes(
    frame: pd.DataFrame,
    *,
    required_columns: Sequence[str],
) -> bytes:
    frame = _as_frame(frame)

    for column in reversed(tuple(required_columns)):
        if column not in frame.columns:
            frame.insert(0, column, None)

    if frame.empty:
        frame = pd.DataFrame(columns=list(dict.fromkeys(frame.columns)))

    safe = frame.copy()
    for column in safe.columns:
        safe[column] = safe[column].map(_csv_cell)

    buffer = StringIO(newline="")
    safe.to_csv(
        buffer,
        index=False,
        lineterminator="\n",
    )
    return buffer.getvalue().encode("utf-8")


def _flatten_mapping_rows(
    value: Any,
    *,
    prefix: str = "",
) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        rows: list[dict[str, Any]] = []
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, Mapping):
                rows.extend(
                    _flatten_mapping_rows(
                        item,
                        prefix=path,
                    )
                )
            else:
                rows.append(
                    {
                        "metric": path,
                        "value": item,
                    }
                )
        return rows

    return [
        {
            "metric": prefix or "value",
            "value": value,
        }
    ]


def _match_comparison_table(bundle) -> pd.DataFrame:
    overview = _section_data(bundle, "overview")
    profile = (
        overview.get("game_profile", {})
        if isinstance(overview, Mapping)
        else {}
    )

    rows = []
    for team in tuple(getattr(bundle, "teams", ()) or ()):
        values = profile.get(team, {}) if isinstance(profile, Mapping) else {}
        row = {"team_name": team}
        if isinstance(values, Mapping):
            row.update(dict(values))
        elif values not in (None, ""):
            row["value"] = values
        rows.append(row)

    return _as_frame(rows)


def _data_coverage_table(bundle) -> pd.DataFrame:
    overview = _section_data(bundle, "overview")
    coverage = (
        overview.get("data_coverage", {})
        if isinstance(overview, Mapping)
        else {}
    )
    return _as_frame(_flatten_mapping_rows(coverage))


def _pass_network_table(bundle) -> pd.DataFrame:
    data = _section_data(bundle, "pass-network")
    teams_data = (
        data.get("teams", {})
        if isinstance(data, Mapping)
        else {}
    )
    frames = []
    for team in tuple(getattr(bundle, "teams", ()) or ()):
        payload = teams_data.get(team, {}) if isinstance(teams_data, Mapping) else {}
        edges = payload.get("edges") if isinstance(payload, Mapping) else None
        frames.append(_with_report_team(_as_frame(edges), team))
    return _concat(frames, required_columns=("report_team", "team_name"))


def _progressive_passers_table(bundle) -> pd.DataFrame:
    data = _section_data(bundle, "progressive-passes")
    teams_data = (
        data.get("teams", {})
        if isinstance(data, Mapping)
        else {}
    )
    frames = []
    for team in tuple(getattr(bundle, "teams", ()) or ()):
        payload = teams_data.get(team, {}) if isinstance(teams_data, Mapping) else {}
        ranking = (
            payload.get("player_ranking")
            if isinstance(payload, Mapping)
            else None
        )
        frames.append(_with_report_team(_as_frame(ranking), team))
    return _concat(frames, required_columns=("report_team", "team_name"))


def _final_third_entries_table(bundle) -> pd.DataFrame:
    data = _section_data(bundle, "final-third-entries")
    teams_data = (
        data.get("teams", {})
        if isinstance(data, Mapping)
        else {}
    )
    frames = []
    for team in tuple(getattr(bundle, "teams", ()) or ()):
        payload = teams_data.get(team, {}) if isinstance(teams_data, Mapping) else {}
        entries = payload.get("entries") if isinstance(payload, Mapping) else None
        frames.append(_with_report_team(_as_frame(entries), team))
    return _concat(frames, required_columns=("report_team", "team_name"))


def _cross_routes_table(bundle) -> pd.DataFrame:
    data = _section_data(bundle, "cross-flow")
    frames = []
    for team in tuple(getattr(bundle, "teams", ()) or ()):
        payload = data.get(team, {}) if isinstance(data, Mapping) else {}
        routes = payload.get("routes") if isinstance(payload, Mapping) else None
        frames.append(_with_report_team(_as_frame(routes), team))
    return _concat(frames, required_columns=("report_team", "team_name"))


def _buildup_summary_table(bundle) -> pd.DataFrame:
    data = _section_data(bundle, "build-up")
    teams_data = (
        data.get("teams", {})
        if isinstance(data, Mapping)
        else {}
    )
    frames = []
    for team in tuple(getattr(bundle, "teams", ()) or ()):
        payload = teams_data.get(team, {}) if isinstance(teams_data, Mapping) else {}
        summary = payload.get("summary") if isinstance(payload, Mapping) else None
        frames.append(_with_report_team(_as_frame(summary), team))
    return _concat(frames, required_columns=("report_team", "team_name"))


def _transition_table(
    bundle,
    section_id: str,
) -> pd.DataFrame:
    data = _section_data(bundle, section_id)
    frames = []

    for team in tuple(getattr(bundle, "teams", ()) or ()):
        payload = data.get(team, {}) if isinstance(data, Mapping) else {}
        combined = (
            payload.get("combined")
            if isinstance(payload, Mapping)
            else None
        )
        frame = _with_report_team(_as_frame(combined), team)

        if frame.empty and isinstance(payload, Mapping):
            stats = payload.get("stats")
            if isinstance(stats, Mapping) and stats:
                fallback = pd.DataFrame(
                    [
                        {
                            "report_team": team,
                            "team_name": team,
                            "metric": row["metric"],
                            "value": row["value"],
                        }
                        for row in _flatten_mapping_rows(stats)
                    ]
                )
                frame = fallback

        frames.append(frame)

    return _concat(
        frames,
        required_columns=("report_team", "team_name"),
    )


def _restarts_table(bundle) -> pd.DataFrame:
    data = _section_data(bundle, "restarts")
    frames = []

    for team in tuple(getattr(bundle, "teams", ()) or ()):
        payload = data.get(team, {}) if isinstance(data, Mapping) else {}
        records = payload.get("records") if isinstance(payload, Mapping) else None
        frames.append(_with_report_team(_as_frame(records), team))

    return _concat(frames, required_columns=("report_team", "team_name"))


def _player_rankings_table(bundle) -> pd.DataFrame:
    data = _section_data(bundle, "player-highlights")
    if not isinstance(data, Mapping):
        data = {}

    families = (
        ("passing", data.get("player_stats")),
        ("shooting", data.get("shot_sequence_ranking")),
        ("defending", data.get("defensive_ranking")),
    )

    frames = []
    for family, source in families:
        frame = _as_frame(source)
        if frame.empty:
            continue
        frame.insert(0, "ranking_family", family)
        frames.append(frame)

    return _concat(
        frames,
        required_columns=("ranking_family",),
    )


def _event_explorer_table(bundle) -> pd.DataFrame:
    frames = []

    pass_locations = _section_data(bundle, "pass-locations")
    if isinstance(pass_locations, Mapping):
        frames.append(
            _with_source(
                _as_frame(pass_locations.get("passes")),
                "passes",
            )
        )

    overview = _section_data(bundle, "overview")
    if isinstance(overview, Mapping):
        frames.append(
            _with_source(
                _as_frame(overview.get("shots")),
                "shots",
            )
        )

    final_third = _section_data(bundle, "final-third-entries")
    if isinstance(final_third, Mapping):
        frames.append(
            _with_source(
                _as_frame(final_third.get("carries")),
                "carries",
            )
        )

    defensive_shape = _section_data(bundle, "defensive-shape")
    if isinstance(defensive_shape, Mapping):
        for team in tuple(getattr(bundle, "teams", ()) or ()):
            profile = defensive_shape.get(team, {})
            actions = (
                profile.get("actions")
                if isinstance(profile, Mapping)
                else None
            )
            frame = _with_report_team(_as_frame(actions), team)
            frames.append(
                _with_source(
                    frame,
                    "defensive-actions",
                )
            )

    return _concat(
        frames,
        required_columns=("source_dataset",),
    )


def build_pack_tables(bundle) -> dict[str, pd.DataFrame]:
    """Build all CSV views from an existing REPORT-02 bundle only."""

    return {
        "tables/match-comparison.csv": _match_comparison_table(bundle),
        "tables/data-coverage.csv": _data_coverage_table(bundle),
        "tables/pass-network-connections.csv": _pass_network_table(bundle),
        "tables/progressive-passers.csv": _progressive_passers_table(bundle),
        "tables/final-third-entries.csv": _final_third_entries_table(bundle),
        "tables/cross-routes.csv": _cross_routes_table(bundle),
        "tables/buildup-summary.csv": _buildup_summary_table(bundle),
        "tables/defensive-transitions.csv": _transition_table(
            bundle,
            "defensive-transitions",
        ),
        "tables/offensive-transitions.csv": _transition_table(
            bundle,
            "offensive-transitions",
        ),
        "tables/restarts.csv": _restarts_table(bundle),
        "tables/player-rankings.csv": _player_rankings_table(bundle),
        "tables/event-explorer.csv": _event_explorer_table(bundle),
    }


def _generation_manifest(
    bundle,
    manifest: ReportManifest,
    *,
    pdf_filename: str,
) -> dict[str, Any]:
    payload = manifest.to_dict()
    section_lookup = {
        getattr(section, "id", ""): section
        for section in tuple(getattr(bundle, "sections", ()) or ())
    }

    statuses: dict[str, str] = {}

    for section_spec in payload.get("sections", []):
        section_id = section_spec.get("id")
        section_bundle = section_lookup.get(section_id)

        if section_bundle is None:
            status = "skipped"
            error_type = None
            error_message = "Section missing from data bundle."
        else:
            raw_status = getattr(section_bundle, "status", "skipped")
            status = getattr(raw_status, "value", str(raw_status))
            if status not in _ALLOWED_SECTION_STATUSES:
                status = "error"
            error_type = getattr(section_bundle, "error_type", None)
            error_message = getattr(section_bundle, "error_message", None)

        statuses[str(section_id)] = status
        section_spec["generation"] = {
            "status": status,
            "error_type": error_type,
            "error_message": error_message,
        }

    payload["generation"] = {
        "pack_schema_version": PACK_SCHEMA_VERSION,
        "source_signature": getattr(bundle, "source_signature", None),
        "scope": getattr(bundle, "scope", None),
        "teams": tuple(getattr(bundle, "teams", ()) or ()),
        "pdf_filename": pdf_filename,
        "section_statuses": statuses,
    }

    return normalize_for_json(payload)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        normalize_for_json(value),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")


def _write_zip_entry(
    archive: zipfile.ZipFile,
    name: str,
    data: bytes,
) -> None:
    info = zipfile.ZipInfo(
        filename=name,
        date_time=(1980, 1, 1, 0, 0, 0),
    )
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    archive.writestr(info, data)


def build_match_analysis_pack(
    bundle,
    manifest: ReportManifest = REPORT_MANIFEST,
    *,
    pdf_config: MatchReportPdfConfig | None = None,
    figure_config: FigureCatalogConfig | None = None,
) -> bytes:
    """Return the complete Match Analysis Pack as compressed ZIP bytes.

    All exported content is derived from the supplied REPORT-02 bundle.
    No raw source JSON is included and no file is written to disk.
    """

    pdf_filename = match_report_pdf_filename(bundle)

    catalog = build_report_figure_catalog(
        bundle,
        manifest,
        config=figure_config,
    )
    pdf_bytes = render_match_report_pdf(
        bundle,
        catalog,
        manifest,
        config=pdf_config,
    )
    if not isinstance(pdf_bytes, (bytes, bytearray)):
        raise TypeError("REPORT-05 renderer must return PDF bytes.")
    pdf_bytes = bytes(pdf_bytes)
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("REPORT-05 renderer did not return a valid PDF.")

    report_data = (
        bundle.to_json(indent=2).encode("utf-8")
        if hasattr(bundle, "to_json")
        else _json_bytes(bundle)
    )
    manifest_data = _json_bytes(
        _generation_manifest(
            bundle,
            manifest,
            pdf_filename=pdf_filename,
        )
    )

    tables = build_pack_tables(bundle)

    buffer = BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        _write_zip_entry(
            archive,
            pdf_filename,
            pdf_bytes,
        )
        _write_zip_entry(
            archive,
            "report-data.json",
            report_data,
        )
        _write_zip_entry(
            archive,
            "report-manifest.json",
            manifest_data,
        )

        for path in TABLE_PATHS:
            frame = tables.get(path, pd.DataFrame())
            required = {
                "tables/match-comparison.csv": ("team_name",),
                "tables/data-coverage.csv": ("metric", "value"),
                "tables/pass-network-connections.csv": (
                    "report_team",
                    "team_name",
                ),
                "tables/progressive-passers.csv": (
                    "report_team",
                    "team_name",
                ),
                "tables/final-third-entries.csv": (
                    "report_team",
                    "team_name",
                ),
                "tables/cross-routes.csv": (
                    "report_team",
                    "team_name",
                ),
                "tables/buildup-summary.csv": (
                    "report_team",
                    "team_name",
                ),
                "tables/defensive-transitions.csv": (
                    "report_team",
                    "team_name",
                ),
                "tables/offensive-transitions.csv": (
                    "report_team",
                    "team_name",
                ),
                "tables/restarts.csv": (
                    "report_team",
                    "team_name",
                ),
                "tables/player-rankings.csv": ("ranking_family",),
                "tables/event-explorer.csv": ("source_dataset",),
            }[path]
            _write_zip_entry(
                archive,
                path,
                _csv_bytes(
                    frame,
                    required_columns=required,
                ),
            )

    return buffer.getvalue()


def build_match_analysis_pack_buffer(
    bundle,
    manifest: ReportManifest = REPORT_MANIFEST,
    *,
    pdf_config: MatchReportPdfConfig | None = None,
    figure_config: FigureCatalogConfig | None = None,
) -> BytesIO:
    """Return a rewound BytesIO containing the complete Match Analysis Pack."""

    buffer = BytesIO(
        build_match_analysis_pack(
            bundle,
            manifest,
            pdf_config=pdf_config,
            figure_config=figure_config,
        )
    )
    buffer.seek(0)
    return buffer
