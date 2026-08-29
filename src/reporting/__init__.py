"""Neutral report contracts and canonical match-report manifest."""

from .manifest import (
    REPORT_MANIFEST,
    REPORT_SECTIONS,
    REQUIRED_SECTION_IDS,
)
from .models import (
    ExportOrientation,
    MissingDataBehavior,
    ReportExportSpec,
    ReportFigureSpec,
    ReportManifest,
    ReportScope,
    ReportSectionSpec,
    ReportSelectionSpec,
    ReportTableSpec,
)

__all__ = [
    "ExportOrientation",
    "MissingDataBehavior",
    "REPORT_MANIFEST",
    "REPORT_SECTIONS",
    "REQUIRED_SECTION_IDS",
    "ReportExportSpec",
    "ReportFigureSpec",
    "ReportManifest",
    "ReportScope",
    "ReportSectionSpec",
    "ReportSelectionSpec",
    "ReportTableSpec",
]
