"""Final Match Analysis composition accounting.

The report bundle describes data availability.  This module separately tracks
what actually reached final PDF composition: sections, tables and figures.
The two states stay distinct so an item can have valid data but still fail
composition without the pack being reported as complete.
"""

from __future__ import annotations

from typing import Any, Mapping


_TERMINAL_COMPOSITION_STATES = {
    "generated",
    "empty",
    "skipped",
    "error",
}


def artifact_key(artifact):
    """Backward-compatible key for a concrete figure artifact."""

    return (
        artifact.section_id,
        artifact.id,
        artifact.team_name,
        artifact.variant,
    )


def section_key(section_id: str):
    return ("section", str(section_id))


def table_key(section_id: str, table_id: str):
    return ("table", str(section_id), str(table_id))


def _status_value(value: Any, default: str = "missing") -> str:
    raw = getattr(value, "value", value)
    text = str(raw or default)
    return text


def _bundle_section_statuses(bundle) -> dict[str, str]:
    return {
        str(getattr(section, "id", "")): _status_value(
            getattr(section, "status", None)
        )
        for section in tuple(getattr(bundle, "sections", ()) or ())
    }


def prepare_render_audit(bundle, catalog=None, manifest=None):
    """Create mutable per-request final-composition records.

    The historical two-argument ``(catalog, manifest)`` call is still
    supported and returns the original figure-only accounting.  Pack
    generation supplies ``(bundle, catalog, manifest)`` and therefore receives
    section/table accounting as well.
    """

    legacy_call = manifest is None
    if legacy_call:
        manifest = catalog
        catalog = bundle
        bundle = None

    section_data = (
        _bundle_section_statuses(bundle)
        if bundle is not None
        else {}
    )
    records: dict[Any, dict[str, Any]] = {}

    figure_policies = {
        (section.id, figure.id): (
            section.required
            if figure.required is None
            else figure.required
        )
        for section in manifest.sections
        for figure in section.figures
    }

    if not legacy_call:
        for section in manifest.sections:
            data_status = section_data.get(section.id, "missing")
            records[section_key(section.id)] = {
                "kind": "section",
                "section_id": section.id,
                "content_id": section.id,
                "title": section.title,
                "required": bool(section.required),
                "data_status": data_status,
                "composition_status": "pending",
                # ``render_status`` is kept as an alias for consumers written
                # before section/table accounting existed.
                "render_status": "pending",
                "error_type": None,
                "error_message": None,
            }

            for table in section.tables:
                records[table_key(section.id, table.id)] = {
                    "kind": "table",
                    "section_id": section.id,
                    "content_id": table.id,
                    "table_id": table.id,
                    "title": table.title,
                    "required": bool(section.required),
                    # Refined by the PDF renderer before final composition.
                    "data_status": data_status,
                    "composition_status": "pending",
                    "render_status": "pending",
                    "error_type": None,
                    "error_message": None,
                }

    for artifact in tuple(getattr(catalog, "figures", ()) or ()):
        status = _status_value(getattr(artifact, "status", None), "error")
        initial_composition_status = (
            ("skipped" if status == "skipped" else "error")
            if legacy_call
            else "pending"
        )
        records[artifact_key(artifact)] = {
            "kind": "figure",
            "section_id": artifact.section_id,
            "content_id": artifact.id,
            "figure_id": artifact.id,
            "team_name": artifact.team_name,
            "variant": artifact.variant,
            "selection": getattr(artifact, "selection", None),
            "data_status": _status_value(
                getattr(artifact, "source_section_status", None),
                "missing",
            ),
            "composition_status": initial_composition_status,
            "render_status": initial_composition_status,
            "renderer_id": artifact.renderer_id,
            "required": bool(
                figure_policies.get(
                    (artifact.section_id, artifact.id),
                    True,
                )
            ),
            "artifact_status": status,
            "error_type": (
                "ArtifactNotRendered" if legacy_call and status != "skipped" else None
            ),
            "error_message": (
                "Artifact was not included in PDF composition."
                if legacy_call and status != "skipped"
                else None
            ),
        }

    return records


def update_composition_record(
    record: dict[str, Any] | None,
    *,
    status: str,
    data_status: str | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    if record is None:
        return
    record["composition_status"] = status
    record["render_status"] = status
    if data_status is not None:
        record["data_status"] = data_status
    record["error_type"] = error_type
    record["error_message"] = error_message


def finalize_render_audit(records) -> None:
    """Turn every unresolved item into an explicit final state.

    A pending item means final composition never accounted for it.  Treating it
    as an error is intentional: silent omission must never produce a successful
    pack.
    """

    for record in records:
        status = str(record.get("composition_status") or "pending")
        if status in _TERMINAL_COMPOSITION_STATES:
            continue

        data_status = str(record.get("data_status") or "missing")
        if data_status == "empty":
            update_composition_record(
                record,
                status="empty",
                error_message=None,
            )
        elif data_status == "skipped" and not bool(record.get("required")):
            update_composition_record(
                record,
                status="skipped",
                error_message=None,
            )
        else:
            update_composition_record(
                record,
                status="error",
                error_type="CompositionNotRecorded",
                error_message="Final composition was not completed.",
            )


def _failed(record: Mapping[str, Any]) -> bool:
    data_status = str(record.get("data_status") or "missing")
    composition_status = str(
        record.get("composition_status") or "pending"
    )

    if data_status == "error" or composition_status == "error":
        return True

    # Required content may be legitimately empty, and a figure can also be
    # explicitly skipped by its selector without making the section itself
    # incomplete. Sections/tables, however, must not silently disappear.
    if bool(record.get("required")):
        if data_status == "missing" or composition_status == "pending":
            return True
        if record.get("kind") in {"section", "table"}:
            return (
                data_status == "skipped"
                or composition_status == "skipped"
            )

    return False


def _kind_rows(records, kind: str) -> list[Mapping[str, Any]]:
    return [row for row in records if row.get("kind") == kind]


def _kind_summary(records, kind: str, prefix: str) -> dict[str, int]:
    rows = _kind_rows(records, kind)
    return {
        f"{prefix}_expected": len(rows),
        f"{prefix}_generated": sum(
            row.get("composition_status") == "generated"
            for row in rows
        ),
        f"{prefix}_empty": sum(
            row.get("composition_status") == "empty"
            for row in rows
        ),
        f"{prefix}_skipped": sum(
            row.get("composition_status") == "skipped"
            for row in rows
        ),
        f"{prefix}_failed": sum(_failed(row) for row in rows),
        f"required_{prefix}_failed": sum(
            _failed(row) and bool(row.get("required"))
            for row in rows
        ),
    }


def summarize_render_audit(records):
    rows = list(records)

    summary: dict[str, Any] = {}
    summary.update(_kind_summary(rows, "section", "sections"))
    summary.update(_kind_summary(rows, "table", "tables"))
    summary.update(_kind_summary(rows, "figure", "figures"))

    required_failed = sum(
        _failed(row) and bool(row.get("required"))
        for row in rows
    )
    optional_failed = sum(
        _failed(row) and not bool(row.get("required"))
        for row in rows
    )

    summary["required_content_failed"] = required_failed
    summary["optional_content_failed"] = optional_failed
    summary["content_failed"] = required_failed + optional_failed
    summary["complete"] = required_failed == 0
    summary["status"] = (
        "error"
        if required_failed
        else "warning"
        if optional_failed
        else "generated"
    )
    return summary


class MatchAnalysisPackBytes(bytes):
    """Bytes API plus final composition summary and phase timings."""

    def __new__(
        cls,
        payload,
        summary,
        timings=None,
    ):
        result = super().__new__(cls, payload)
        result.render_summary = dict(summary)
        result.timings = dict(timings or {})
        return result
