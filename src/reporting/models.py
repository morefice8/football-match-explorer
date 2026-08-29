"""Neutral, serializable contracts for match-report composition."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import json
from typing import Any


class ReportScope(str, Enum):
    """Data scope supported by the report contract."""

    FULL_MATCH = "full-match"


class ExportOrientation(str, Enum):
    """Intended orientation for a future export renderer."""

    LANDSCAPE = "landscape"
    PORTRAIT = "portrait"


class MissingDataBehavior(str, Enum):
    """Declarative behavior when an expected report item has no data."""

    PLACEHOLDER = "show-placeholder"
    OMIT = "omit"


@dataclass(frozen=True)
class ReportExportSpec:
    """Neutral export dimensions; no rendering backend is implied."""

    width_px: int
    height_px: int
    orientation: ExportOrientation

    def __post_init__(self) -> None:
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("Export dimensions must be positive.")
        if (
            self.orientation is ExportOrientation.LANDSCAPE
            and self.width_px < self.height_px
        ):
            raise ValueError(
                "Landscape export width must be >= height."
            )
        if (
            self.orientation is ExportOrientation.PORTRAIT
            and self.height_px < self.width_px
        ):
            raise ValueError(
                "Portrait export height must be >= width."
            )


@dataclass(frozen=True)
class ReportSelectionSpec:
    """Machine-readable, non-executable rule for choosing report content."""

    rule: str
    limit: int | None = None
    sort_by: str | None = None
    descending: bool = True
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.rule.strip():
            raise ValueError("Selection rule must not be empty.")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("Selection limit must be positive.")


@dataclass(frozen=True)
class ReportFigureSpec:
    """Contract for a figure expected by a report section."""

    id: str
    title: str
    selection: ReportSelectionSpec
    export: ReportExportSpec
    missing_data: MissingDataBehavior = MissingDataBehavior.PLACEHOLDER


@dataclass(frozen=True)
class ReportTableSpec:
    """Contract for a table expected by a report section."""

    id: str
    title: str
    selection: ReportSelectionSpec
    export: ReportExportSpec
    missing_data: MissingDataBehavior = MissingDataBehavior.PLACEHOLDER


@dataclass(frozen=True)
class ReportSectionSpec:
    """Canonical report-section definition."""

    id: str
    title: str
    order: int
    scope: ReportScope
    selection: ReportSelectionSpec
    export: ReportExportSpec
    missing_data: MissingDataBehavior
    required: bool
    figures: tuple[ReportFigureSpec, ...] = ()
    tables: tuple[ReportTableSpec, ...] = ()


@dataclass(frozen=True)
class ReportManifest:
    """Serializable report contract with validation helpers."""

    id: str
    schema_version: str
    sections: tuple[ReportSectionSpec, ...]

    def all_ids(self) -> tuple[str, ...]:
        """Return every manifest, section, figure and table ID."""
        ids: list[str] = [self.id]
        for section in self.sections:
            ids.append(section.id)
            ids.extend(figure.id for figure in section.figures)
            ids.extend(table.id for table in section.tables)
        return tuple(ids)

    def required_section_ids(self) -> tuple[str, ...]:
        """Return required section IDs in report order."""
        return tuple(
            section.id
            for section in self.sections
            if section.required
        )

    def validate(self) -> None:
        """Validate ordering and globally unique stable IDs."""
        ids = self.all_ids()
        duplicates = sorted(
            {
                item_id
                for item_id in ids
                if ids.count(item_id) > 1
            }
        )
        if duplicates:
            raise ValueError(
                "Duplicate report IDs: " + ", ".join(duplicates)
            )

        expected_order = list(range(1, len(self.sections) + 1))
        actual_order = [section.order for section in self.sections]
        if actual_order != expected_order:
            raise ValueError(
                "Report section order must be contiguous and start at 1."
            )

        for item_id in ids:
            if not item_id or item_id != item_id.strip():
                raise ValueError("Report IDs must be non-empty and trimmed.")
            if any(char.isspace() for char in item_id):
                raise ValueError(
                    f"Report ID must not contain whitespace: {item_id!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation."""
        return _to_serializable(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the manifest without Dash or application dependencies."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=False,
        )


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: _to_serializable(
                getattr(value, field.name)
            )
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _to_serializable(item)
            for key, item in value.items()
        }
    return value
