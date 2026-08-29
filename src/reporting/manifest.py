"""Canonical, neutral manifest for the automatic match report."""

from __future__ import annotations

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


LANDSCAPE_FIGURE = ReportExportSpec(
    width_px=1600,
    height_px=900,
    orientation=ExportOrientation.LANDSCAPE,
)

LANDSCAPE_TABLE = ReportExportSpec(
    width_px=1600,
    height_px=1100,
    orientation=ExportOrientation.LANDSCAPE,
)

PORTRAIT_PAGE = ReportExportSpec(
    width_px=1200,
    height_px=1600,
    orientation=ExportOrientation.PORTRAIT,
)

FULL_MATCH = ReportSelectionSpec(
    rule="full-match",
)

ALL_QUALIFIED = ReportSelectionSpec(
    rule="all-qualified-events",
)

TOP_ROUTES = ReportSelectionSpec(
    rule="top-routes-by-crosses",
    limit=8,
    sort_by="crosses",
    descending=True,
)

TOP_SEQUENCE = ReportSelectionSpec(
    rule="top-sequence-by-outcome-priority",
    limit=1,
)

TOP_SEQUENCES = ReportSelectionSpec(
    rule="top-sequences-by-outcome-priority",
    limit=10,
)

TOP_TAKERS = ReportSelectionSpec(
    rule="top-takers-by-restart-count",
    limit=6,
    sort_by="restart-count",
    descending=True,
)

TOP_PLAYER = ReportSelectionSpec(
    rule="top-player-by-metric-family",
    limit=1,
)


def figure(
    item_id: str,
    title: str,
    selection: ReportSelectionSpec = FULL_MATCH,
    *,
    export: ReportExportSpec = LANDSCAPE_FIGURE,
    missing_data: MissingDataBehavior = MissingDataBehavior.PLACEHOLDER,
) -> ReportFigureSpec:
    return ReportFigureSpec(
        id=item_id,
        title=title,
        selection=selection,
        export=export,
        missing_data=missing_data,
    )


def table(
    item_id: str,
    title: str,
    selection: ReportSelectionSpec = FULL_MATCH,
    *,
    export: ReportExportSpec = LANDSCAPE_TABLE,
    missing_data: MissingDataBehavior = MissingDataBehavior.PLACEHOLDER,
) -> ReportTableSpec:
    return ReportTableSpec(
        id=item_id,
        title=title,
        selection=selection,
        export=export,
        missing_data=missing_data,
    )


def section(
    section_id: str,
    title: str,
    order: int,
    *,
    required: bool = True,
    selection: ReportSelectionSpec = FULL_MATCH,
    export: ReportExportSpec = LANDSCAPE_FIGURE,
    missing_data: MissingDataBehavior | None = None,
    figures: tuple[ReportFigureSpec, ...] = (),
    tables: tuple[ReportTableSpec, ...] = (),
) -> ReportSectionSpec:
    resolved_missing_data = (
        missing_data
        if missing_data is not None
        else (
            MissingDataBehavior.PLACEHOLDER
            if required
            else MissingDataBehavior.OMIT
        )
    )
    return ReportSectionSpec(
        id=section_id,
        title=title,
        order=order,
        scope=ReportScope.FULL_MATCH,
        selection=selection,
        export=export,
        missing_data=resolved_missing_data,
        required=required,
        figures=figures,
        tables=tables,
    )


REPORT_SECTIONS: tuple[ReportSectionSpec, ...] = (
    section(
        "overview",
        "Overview and data coverage",
        1,
        export=PORTRAIT_PAGE,
        tables=(
            table(
                "match-overview",
                "Match overview",
                export=PORTRAIT_PAGE,
            ),
            table(
                "data-coverage",
                "Data coverage",
                export=PORTRAIT_PAGE,
            ),
        ),
    ),
    section(
        "formation-timeline",
        "Formation Timeline",
        2,
        figures=(
            figure(
                "formation-timeline-figure",
                "Formation Timeline",
            ),
        ),
        tables=(
            table(
                "formation-spells",
                "Formation spells",
            ),
        ),
    ),
    section(
        "mean-positions",
        "Mean Positions",
        3,
        figures=(
            figure(
                "mean-positions-figure",
                "Mean Positions",
            ),
        ),
        tables=(
            table(
                "mean-position-summary",
                "Mean position summary",
            ),
        ),
    ),
    section(
        "pass-network",
        "Pass Network",
        4,
        figures=(
            figure(
                "pass-network-figure",
                "Pass Network",
            ),
        ),
        tables=(
            table(
                "pass-network-leaders",
                "Pass network leaders",
            ),
        ),
    ),
    section(
        "progressive-passes",
        "Progressive Passes",
        5,
        selection=ALL_QUALIFIED,
        figures=(
            figure(
                "progressive-passes-figure",
                "Progressive Passes",
                ALL_QUALIFIED,
            ),
        ),
        tables=(
            table(
                "progressive-pass-leaders",
                "Progressive pass leaders",
                ALL_QUALIFIED,
            ),
        ),
    ),
    section(
        "final-third-entries",
        "Final Third Entries",
        6,
        selection=ALL_QUALIFIED,
        figures=(
            figure(
                "final-third-entries-figure",
                "Final Third Entries",
                ALL_QUALIFIED,
            ),
        ),
        tables=(
            table(
                "final-third-entry-breakdown",
                "Final third entry breakdown",
                ALL_QUALIFIED,
            ),
        ),
    ),
    section(
        "pass-locations",
        "Pass Locations",
        7,
        figures=(
            figure(
                "pass-locations-figure",
                "Pass Locations",
            ),
        ),
        tables=(
            table(
                "pass-location-breakdown",
                "Pass location breakdown",
            ),
        ),
    ),
    section(
        "cross-flow",
        "Cross Flow",
        8,
        figures=(
            figure(
                "cross-flow-figure",
                "Cross Flow",
                TOP_ROUTES,
            ),
            figure(
                "cross-origin-map",
                "Cross origin map",
            ),
            figure(
                "cross-destination-map",
                "Cross destination map",
            ),
        ),
        tables=(
            table(
                "cross-top-routes",
                "Top cross routes",
                TOP_ROUTES,
            ),
        ),
    ),
    section(
        "build-up",
        "Build-up",
        9,
        figures=(
            figure(
                "build-up-top-sequence",
                "Build-up top sequence",
                TOP_SEQUENCE,
            ),
        ),
        tables=(
            table(
                "build-up-summary",
                "Build-up summary",
            ),
            table(
                "build-up-sequences",
                "Build-up sequences",
                TOP_SEQUENCES,
            ),
        ),
    ),
    section(
        "defensive-shape",
        "Defensive Shape",
        10,
        figures=(
            figure(
                "defensive-shape-figure",
                "Defensive Shape",
            ),
        ),
        tables=(
            table(
                "defensive-shape-summary",
                "Defensive shape summary",
            ),
        ),
    ),
    section(
        "ppda",
        "PPDA",
        11,
        figures=(
            figure(
                "ppda-figure",
                "PPDA",
            ),
        ),
        tables=(
            table(
                "ppda-summary",
                "PPDA summary",
            ),
        ),
    ),
    section(
        "defensive-transitions",
        "Defensive Transitions",
        12,
        figures=(
            figure(
                "defensive-transitions-top-sequence",
                "Defensive transition top sequence",
                TOP_SEQUENCE,
            ),
        ),
        tables=(
            table(
                "defensive-transitions-summary",
                "Defensive transitions summary",
            ),
            table(
                "defensive-transitions-sequences",
                "Defensive transition sequences",
                TOP_SEQUENCES,
            ),
        ),
    ),
    section(
        "offensive-transitions",
        "Offensive Transitions",
        13,
        figures=(
            figure(
                "offensive-transitions-top-sequence",
                "Offensive transition top sequence",
                TOP_SEQUENCE,
            ),
        ),
        tables=(
            table(
                "offensive-transitions-summary",
                "Offensive transitions summary",
            ),
            table(
                "offensive-transitions-sequences",
                "Offensive transition sequences",
                TOP_SEQUENCES,
            ),
        ),
    ),
    section(
        "restarts",
        "Restarts",
        14,
        figures=(
            figure(
                "restart-map",
                "Restart map",
            ),
            figure(
                "restart-top-sequence",
                "Restart top sequence",
                TOP_SEQUENCE,
            ),
        ),
        tables=(
            table(
                "restart-summary",
                "Restart summary",
            ),
            table(
                "restart-takers",
                "Restart takers",
                TOP_TAKERS,
            ),
        ),
    ),
    section(
        "player-highlights",
        "Player highlights",
        15,
        required=False,
        figures=(
            figure(
                "player-highlight-passing",
                "Passing highlight",
                TOP_PLAYER,
                missing_data=MissingDataBehavior.OMIT,
            ),
            figure(
                "player-highlight-shooting",
                "Shooting highlight",
                TOP_PLAYER,
                missing_data=MissingDataBehavior.OMIT,
            ),
            figure(
                "player-highlight-defending",
                "Defending highlight",
                TOP_PLAYER,
                missing_data=MissingDataBehavior.OMIT,
            ),
        ),
        tables=(
            table(
                "player-highlights-table",
                "Player highlights",
                ReportSelectionSpec(
                    rule="top-players-by-metric-family",
                    limit=3,
                ),
                missing_data=MissingDataBehavior.OMIT,
            ),
        ),
    ),
    section(
        "methodology-appendix",
        "Methodology/appendix",
        16,
        export=PORTRAIT_PAGE,
        tables=(
            table(
                "methodology-notes",
                "Methodology notes",
                export=PORTRAIT_PAGE,
            ),
            table(
                "metric-definitions",
                "Metric definitions",
                export=PORTRAIT_PAGE,
            ),
            table(
                "data-quality-notes",
                "Data quality notes",
                export=PORTRAIT_PAGE,
            ),
        ),
    ),
)


REQUIRED_SECTION_IDS: tuple[str, ...] = tuple(
    section_spec.id
    for section_spec in REPORT_SECTIONS
    if section_spec.required
)


REPORT_MANIFEST = ReportManifest(
    id="match-analysis-report",
    schema_version="1.0",
    sections=REPORT_SECTIONS,
)

REPORT_MANIFEST.validate()
