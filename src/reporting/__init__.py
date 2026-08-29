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
    "normalize_for_json",
    "build_match_report_data_bundle",
    "ReportSectionStatus",
    "ReportSectionBundle",
    "MatchReportDataBundle",
    "MatchReportBundleConfig",
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

from .bundle import (
    MatchReportBundleConfig,
    MatchReportDataBundle,
    ReportSectionBundle,
    ReportSectionStatus,
    build_match_report_data_bundle,
    normalize_for_json,
)
