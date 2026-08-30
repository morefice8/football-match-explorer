"""In-memory, brand-aware PDF renderer for Match Analysis reports.

REPORT-05 composes the neutral REPORT-02 bundle and the static REPORT-04
figure catalog with ReportLab/Platypus. It does not import Dash, app.py,
HTML renderers, browser automation or screenshot tooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.sax.saxutils import escape

import pandas as pd
import plotly.io as pio
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

from src.reporting.manifest import REPORT_MANIFEST
from src.reporting.models import ReportManifest


CORAL = HexColor("#E96A4A")
CYAN = HexColor("#63C9D2")
NAVY = HexColor("#17354D")
TEXT = HexColor("#29465B")
MUTED = HexColor("#6C8190")
LIGHT = HexColor("#F5F8FA")
BORDER = HexColor("#DCE7EC")
WHITE = colors.white

ANALYTIC_PAGE_SIZE = landscape(A4)


@dataclass(frozen=True)
class MatchReportPdfConfig:
    """Presentation-only options for REPORT-05."""

    brand_name: str = "IL LAB DELL'8"
    report_label: str = "MATCH ANALYSIS"
    image_scale: float = 1.0
    max_table_columns: int = 10
    brand_font_path: str | Path | None = None
    enable_brand_font: bool = True

    def __post_init__(self) -> None:
        if self.image_scale <= 0:
            raise ValueError("image_scale must be positive")
        if self.max_table_columns <= 0:
            raise ValueError("max_table_columns must be positive")


class _ReportDocTemplate(BaseDocTemplate):
    """Platypus document with TOC and PDF outline support."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._toc_counter = 0

    def afterFlowable(self, flowable) -> None:  # noqa: N802 - ReportLab API
        if not isinstance(flowable, Paragraph):
            return
        if flowable.style.name != "ReportSectionHeading":
            return

        text = flowable.getPlainText()
        key = getattr(flowable, "_report_bookmark", None)
        if not key:
            self._toc_counter += 1
            key = f"report-section-{self._toc_counter}"

        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=0, closed=False)
        self.notify("TOCEntry", (0, text, self.page, key))


@dataclass(frozen=True)
class _FigureCell:
    flowables: tuple[Any, ...]
    error: str | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _section(bundle, section_id: str):
    try:
        return bundle.section(section_id)
    except Exception:
        return None


def _section_status(section) -> str | None:
    if section is None:
        return None
    status = getattr(section, "status", None)
    return getattr(status, "value", status)


def _as_frame(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        return value.to_frame().T
    if isinstance(value, Mapping):
        return pd.DataFrame([dict(value)])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            return pd.DataFrame()
        if all(isinstance(item, Mapping) for item in value):
            return pd.DataFrame(list(value))
    return pd.DataFrame()


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(value)


def _format_scalar(value: Any) -> str:
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
    except Exception:
        pass

    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:.2f}"
    if isinstance(value, Mapping) or isinstance(value, (list, tuple, set)):
        return _json_text(value)
    return str(value)


def _mapping_rows(
    mapping: Mapping[str, Any],
    *,
    prefix: str = "",
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key, value in mapping.items():
        label = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            rows.extend(_mapping_rows(value, prefix=label))
        else:
            rows.append({"Metric": label, "Value": _format_scalar(value)})
    return rows


def _register_brand_font(config: MatchReportPdfConfig) -> str:
    if not config.enable_brand_font:
        return "Helvetica-Bold"

    candidate = (
        Path(config.brand_font_path)
        if config.brand_font_path is not None
        else _repo_root() / "assets" / "BebasNeue-Regular.ttf"
    )

    if not candidate.exists():
        return "Helvetica-Bold"

    font_name = "ReportBebasNeue"
    try:
        pdfmetrics.getFont(font_name)
    except KeyError:
        try:
            pdfmetrics.registerFont(TTFont(font_name, str(candidate)))
        except Exception:
            return "Helvetica-Bold"
    return font_name


def _styles(config: MatchReportPdfConfig) -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    brand_font = _register_brand_font(config)

    return {
        "brand": ParagraphStyle(
            "ReportBrand",
            parent=sample["Title"],
            fontName=brand_font,
            fontSize=27,
            leading=30,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=4,
        ),
        "cover_eyebrow": ParagraphStyle(
            "ReportCoverEyebrow",
            parent=sample["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=11,
            textColor=CORAL,
            tracking=1.2,
            spaceAfter=12,
        ),
        "cover_teams": ParagraphStyle(
            "ReportCoverTeams",
            parent=sample["Title"],
            fontName=brand_font,
            fontSize=34,
            leading=36,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "cover_score": ParagraphStyle(
            "ReportCoverScore",
            parent=sample["Title"],
            fontName=brand_font,
            fontSize=42,
            leading=44,
            textColor=CYAN,
            alignment=TA_LEFT,
            spaceAfter=12,
        ),
        "cover_meta": ParagraphStyle(
            "ReportCoverMeta",
            parent=sample["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            textColor=TEXT,
        ),
        "toc_title": ParagraphStyle(
            "ReportTocTitle",
            parent=sample["Heading1"],
            fontName=brand_font,
            fontSize=28,
            leading=30,
            textColor=NAVY,
            spaceAfter=14,
        ),
        "section": ParagraphStyle(
            "ReportSectionHeading",
            parent=sample["Heading1"],
            fontName=brand_font,
            fontSize=25,
            leading=28,
            textColor=NAVY,
            spaceBefore=0,
            spaceAfter=10,
            keepWithNext=True,
        ),
        "section_kicker": ParagraphStyle(
            "ReportSectionKicker",
            parent=sample["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=CORAL,
            spaceAfter=4,
        ),
        "subheading": ParagraphStyle(
            "ReportSubheading",
            parent=sample["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=TEXT,
            spaceBefore=7,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=13.2,
            textColor=TEXT,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "ReportSmall",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=7.7,
            leading=10.2,
            textColor=MUTED,
            spaceAfter=4,
        ),
        "figure_caption": ParagraphStyle(
            "ReportFigureCaption",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.2,
            leading=10,
            textColor=TEXT,
            alignment=TA_CENTER,
            spaceAfter=3,
        ),
        "reason": ParagraphStyle(
            "ReportSelectionReason",
            parent=sample["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=7,
            leading=9,
            textColor=MUTED,
            alignment=TA_LEFT,
            spaceBefore=3,
        ),
        "placeholder": ParagraphStyle(
            "ReportPlaceholder",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "table": ParagraphStyle(
            "ReportTableCell",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=6.8,
            leading=8.6,
            textColor=TEXT,
        ),
        "table_header": ParagraphStyle(
            "ReportTableHeader",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=6.8,
            leading=8.6,
            textColor=WHITE,
            alignment=TA_LEFT,
        ),
        "footer": ParagraphStyle(
            "ReportFooter",
            parent=sample["Normal"],
            fontName="Helvetica",
            fontSize=7,
            textColor=MUTED,
            alignment=TA_RIGHT,
        ),
    }


def _cover_context(bundle) -> dict[str, str]:
    info = dict(getattr(bundle, "match_info", {}) or {})
    teams = tuple(getattr(bundle, "teams", ()) or ())
    home = str(teams[0]) if len(teams) > 0 else str(
        _first_present(info, "hteamName", "home_team", "homeTeam") or "Home"
    )
    away = str(teams[1]) if len(teams) > 1 else str(
        _first_present(info, "ateamName", "away_team", "awayTeam") or "Away"
    )

    overview = _section(bundle, "overview")
    overview_data = getattr(overview, "data", {}) or {}
    result = overview_data.get("result", {}) if isinstance(overview_data, Mapping) else {}
    metadata = overview_data.get("metadata", {}) if isinstance(overview_data, Mapping) else {}
    merged = {**info, **(metadata if isinstance(metadata, Mapping) else {})}

    home_score = result.get("home_score") if isinstance(result, Mapping) else None
    away_score = result.get("away_score") if isinstance(result, Mapping) else None
    if home_score in (None, ""):
        home_score = _first_present(merged, "hteamScore", "home_score", "homeScore")
    if away_score in (None, ""):
        away_score = _first_present(merged, "ateamScore", "away_score", "awayScore")

    score = "-"
    if home_score not in (None, "") or away_score not in (None, ""):
        score = f"{home_score if home_score not in (None, '') else '-'} - {away_score if away_score not in (None, '') else '-'}"

    competition = _first_present(
        merged,
        "competitionName",
        "competition_name",
        "competition",
        "leagueName",
        "league_name",
        "tournamentName",
        "tournament_name",
    ) or "Competition not specified"

    date = _first_present(
        merged,
        "game_date",
        "match_date",
        "matchDate",
        "date",
        "startDate",
        "start_date",
    ) or "Date not specified"

    return {
        "home": home,
        "away": away,
        "score": str(score),
        "competition": str(competition),
        "date": str(date),
    }


def _draw_page_chrome(canvas, doc, config: MatchReportPdfConfig) -> None:
    page = canvas.getPageNumber()
    if page <= 1:
        return

    width, height = ANALYTIC_PAGE_SIZE
    canvas.saveState()
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(13 * mm, 10.5 * mm, width - 13 * mm, 10.5 * mm)

    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(13 * mm, 6.5 * mm, config.brand_name)
    canvas.drawRightString(width - 13 * mm, 6.5 * mm, f"Page {page}")
    canvas.restoreState()


def _doc(buffer: BytesIO, config: MatchReportPdfConfig) -> _ReportDocTemplate:
    width, height = ANALYTIC_PAGE_SIZE
    left = right = 13 * mm
    top = 14 * mm
    bottom = 15 * mm

    doc = _ReportDocTemplate(
        buffer,
        pagesize=ANALYTIC_PAGE_SIZE,
        leftMargin=left,
        rightMargin=right,
        topMargin=top,
        bottomMargin=bottom,
        title="Match Analysis Report",
        author=config.brand_name,
        subject="Football match analysis",
    )

    frame = Frame(
        left,
        bottom,
        width - left - right,
        height - top - bottom,
        id="report-main-frame",
        showBoundary=0,
    )
    doc.addPageTemplates(
        [
            PageTemplate(
                id="report-landscape",
                pagesize=ANALYTIC_PAGE_SIZE,
                frames=[frame],
                onPage=lambda canvas, document: _draw_page_chrome(
                    canvas,
                    document,
                    config,
                ),
            )
        ]
    )
    return doc


def _cover_story(bundle, config, styles) -> list[Any]:
    context = _cover_context(bundle)

    accent = Table(
        [["", ""]],
        colWidths=[90 * mm, 90 * mm],
        rowHeights=[3.5 * mm],
    )
    accent.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), CORAL),
                ("BACKGROUND", (1, 0), (1, 0), CYAN),
                ("LINEBELOW", (0, 0), (-1, 0), 0, WHITE),
            ]
        )
    )

    meta = Table(
        [
            [
                Paragraph("COMPETITION", styles["small"]),
                Paragraph("DATE", styles["small"]),
            ],
            [
                Paragraph(escape(context["competition"]), styles["body"]),
                Paragraph(escape(context["date"]), styles["body"]),
            ],
        ],
        colWidths=[88 * mm, 88 * mm],
        hAlign="LEFT",
    )
    meta.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, BORDER),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    return [
        Spacer(1, 12 * mm),
        accent,
        Spacer(1, 12 * mm),
        Paragraph(escape(config.brand_name), styles["brand"]),
        Paragraph(escape(config.report_label), styles["cover_eyebrow"]),
        Spacer(1, 7 * mm),
        Paragraph(
            f"{escape(context['home'])}<br/>{escape(context['away'])}",
            styles["cover_teams"],
        ),
        Paragraph(escape(context["score"]), styles["cover_score"]),
        Spacer(1, 4 * mm),
        meta,
        Spacer(1, 15 * mm),
        Paragraph(
            "Full Match tactical report generated from the validated Match Analysis data bundle and static figure catalog.",
            styles["cover_meta"],
        ),
    ]


def _toc_story(styles) -> list[Any]:
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            "ReportTOCLevel0",
            fontName="Helvetica",
            fontSize=8.8,
            leading=10.8,
            textColor=TEXT,
            leftIndent=0,
            firstLineIndent=0,
            spaceBefore=1,
        )
    ]
    return [
        Paragraph("Contents", styles["toc_title"]),
        Paragraph(
            "All analytical sections use the Full Match scope. Empty or failed sections remain visible as explicit report placeholders.",
            styles["body"],
        ),
        Spacer(1, 4 * mm),
        toc,
    ]


def _section_heading(section_spec, styles) -> list[Any]:
    heading = Paragraph(
        escape(section_spec.title),
        styles["section"],
    )
    setattr(heading, "_report_bookmark", f"section-{section_spec.order:02d}-{section_spec.id}")
    return [
        Paragraph(
            f"SECTION {section_spec.order:02d} / FULL MATCH",
            styles["section_kicker"],
        ),
        heading,
        Table(
            [["", ""]],
            colWidths=[32 * mm, 145 * mm],
            rowHeights=[2.1 * mm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, 0), CORAL),
                    ("BACKGROUND", (1, 0), (1, 0), CYAN),
                ]
            ),
        ),
        Spacer(1, 4 * mm),
    ]


def _placeholder_box(message: str, styles, *, height: float = 28 * mm) -> Table:
    content = Paragraph(escape(message), styles["placeholder"])
    table = Table([[content]], colWidths=[175 * mm], rowHeights=[height])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.7, BORDER),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return table


def _image_flowable(
    artifact,
    styles,
    config: MatchReportPdfConfig,
    *,
    max_width: float,
    max_height: float,
) -> _FigureCell:
    caption_parts = []
    if getattr(artifact, "team_name", None):
        caption_parts.append(str(artifact.team_name))
    caption_parts.append(str(getattr(artifact, "title", "Figure")))
    variant = str(getattr(artifact, "variant", "summary") or "summary")
    if variant != "summary":
        caption_parts.append(variant.title())
    caption = " - ".join(caption_parts)

    try:
        png = pio.to_image(
            artifact.figure,
            format="png",
            width=int(artifact.width_px),
            height=int(artifact.height_px),
            scale=config.image_scale,
        )
        image_buffer = BytesIO(png)
        image = Image(image_buffer)
        image._report_buffer = image_buffer  # keep it alive through multiBuild

        raw_width = float(image.imageWidth or artifact.width_px)
        raw_height = float(image.imageHeight or artifact.height_px)
        ratio = min(max_width / raw_width, max_height / raw_height)
        image.drawWidth = raw_width * ratio
        image.drawHeight = raw_height * ratio

        flowables: list[Any] = [
            Paragraph(escape(caption), styles["figure_caption"]),
            image,
        ]
    except Exception as exc:
        message = (
            f"Figure unavailable: {type(exc).__name__}: {exc}"
        )
        flowables = [
            Paragraph(escape(caption), styles["figure_caption"]),
            _placeholder_box(message, styles, height=50 * mm),
        ]
        error = message
    else:
        error = None

    reason = getattr(artifact, "selection_reason", None)
    if reason:
        flowables.append(
            Paragraph(
                "<b>Selection reason:</b> " + escape(str(reason)),
                styles["reason"],
            )
        )

    artifact_error = getattr(artifact, "error_message", None)
    if artifact_error and not reason:
        flowables.append(
            Paragraph(
                "<b>Figure status:</b> " + escape(str(artifact_error)),
                styles["reason"],
            )
        )

    return _FigureCell(tuple(flowables), error=error)


def _artifacts_for(catalog, figure_id: str) -> list[Any]:
    try:
        values = list(catalog.by_id(figure_id))
    except Exception:
        values = [
            item
            for item in getattr(catalog, "figures", ())
            if getattr(item, "id", None) == figure_id
        ]
    return values


def _figure_story_for_spec(
    catalog,
    figure_spec,
    styles,
    config,
) -> list[Any]:
    artifacts = _artifacts_for(catalog, figure_spec.id)
    if not artifacts:
        return [
            _placeholder_box(
                f"No catalog artifact was generated for {figure_spec.title}.",
                styles,
            )
        ]

    variants: list[str] = []
    for artifact in artifacts:
        variant = str(getattr(artifact, "variant", "summary") or "summary")
        if variant not in variants:
            variants.append(variant)

    story: list[Any] = []
    for variant in variants:
        group = [
            item
            for item in artifacts
            if str(getattr(item, "variant", "summary") or "summary") == variant
        ]

        home_away = [item for item in group if getattr(item, "team_name", None)]
        if len(home_away) == 2:
            max_cell_width = 84 * mm
            max_cell_height = 92 * mm
            left = _image_flowable(
                home_away[0],
                styles,
                config,
                max_width=max_cell_width,
                max_height=max_cell_height,
            )
            right = _image_flowable(
                home_away[1],
                styles,
                config,
                max_width=max_cell_width,
                max_height=max_cell_height,
            )
            pair = Table(
                [[list(left.flowables), list(right.flowables)]],
                colWidths=[88 * mm, 88 * mm],
                hAlign="LEFT",
            )
            pair.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 3),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.extend([pair, Spacer(1, 4 * mm)])
            continue

        for artifact in group:
            cell = _image_flowable(
                artifact,
                styles,
                config,
                max_width=176 * mm,
                max_height=118 * mm,
            )
            story.extend([KeepTogether(list(cell.flowables)), Spacer(1, 4 * mm)])

    return story


def _trim_columns(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if frame.shape[1] <= limit:
        return frame

    priority = [
        "team_name",
        "playerName",
        "player_name",
        "sequence_id",
        "loss_sequence_id",
        "restart_id",
        "terminal_outcome",
        "outcome",
        "minute",
        "duration_seconds",
        "event_count",
        "Offensive Pass Contributions",
        "Progressive Passes",
        "Passes into Box",
        "Key Passes",
        "Shot Sequence Involvements",
        "unique",
    ]
    selected = [column for column in priority if column in frame.columns]
    for column in frame.columns:
        if column not in selected:
            selected.append(column)
        if len(selected) >= limit:
            break
    return frame[selected]


_MAX_TABLE_CELL_CHARS = 420


def _format_table_cell(value: Any) -> str:
    """Return a bounded, human-readable representation for a PDF table cell."""

    if isinstance(value, pd.DataFrame):
        return (
            f"[table: {len(value):,} rows x "
            f"{len(value.columns):,} columns]"
        )

    if isinstance(value, pd.Series):
        if len(value) > 12:
            return f"[series: {len(value):,} values]"
        value = value.to_dict()

    if isinstance(value, Mapping):
        text = _json_text(value)
    elif isinstance(value, (list, tuple, set)):
        if len(value) > 12:
            return f"[collection: {len(value):,} items]"
        text = _json_text(value)
    elif (
        not isinstance(value, (str, bytes, bytearray))
        and getattr(value, "shape", None) not in (None, ())
    ):
        shape = getattr(value, "shape", None)
        return f"[array shape={shape}]"
    else:
        text = _format_scalar(value)

    text = " ".join(str(text).split())
    if len(text) > _MAX_TABLE_CELL_CHARS:
        text = text[: _MAX_TABLE_CELL_CHARS - 3].rstrip() + "..."
    return text


def _long_table(
    frame: pd.DataFrame,
    styles,
    *,
    max_columns: int,
    available_width: float = 176 * mm,
) -> LongTable | Table:
    frame = _trim_columns(frame.copy(), max_columns)
    if frame.empty:
        return _placeholder_box("No table rows available.", styles)

    headers = [str(column) for column in frame.columns]
    data: list[list[Any]] = [
        [Paragraph(escape(header), styles["table_header"]) for header in headers]
    ]

    for _, row in frame.iterrows():
        data.append(
            [
                Paragraph(
                    escape(_format_table_cell(row[column])),
                    styles["table"],
                )
                for column in frame.columns
            ]
        )

    count = len(headers)
    widths = [available_width / count] * count

    table = LongTable(
        data,
        colWidths=widths,
        repeatRows=1,
        splitByRow=1,
        splitInRow=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("GRID", (0, 0), (-1, -1), 0.25, BORDER),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _rows_from_team_mapping(data: Mapping[str, Any], teams: Iterable[str]) -> pd.DataFrame:
    rows = []
    for team in teams:
        payload = data.get(team, {}) if isinstance(data, Mapping) else {}
        if isinstance(payload, Mapping):
            simple = {
                key: value
                for key, value in payload.items()
                if not isinstance(value, (pd.DataFrame, list, tuple, Mapping))
            }
            simple = {"team_name": team, **simple}
            rows.append(simple)
    return pd.DataFrame(rows)


def _table_payloads(bundle, table_id: str) -> list[tuple[str, pd.DataFrame]]:
    teams = tuple(getattr(bundle, "teams", ()) or ())

    if table_id in {"match-overview", "data-coverage"}:
        section = _section(bundle, "overview")
        data = getattr(section, "data", {}) or {}
        if table_id == "match-overview":
            game_profile = data.get("game_profile", {}) if isinstance(data, Mapping) else {}
            frame = pd.DataFrame.from_dict(game_profile, orient="index") if game_profile else pd.DataFrame()
            if not frame.empty:
                frame.index.name = "team_name"
                frame = frame.reset_index()
            return [("Match overview", frame)]
        coverage = data.get("data_coverage", {}) if isinstance(data, Mapping) else {}
        return [("Data coverage", pd.DataFrame(_mapping_rows(coverage)))]

    if table_id == "formation-spells":
        section = _section(bundle, "formation-timeline")
        data = getattr(section, "data", {}) or {}
        moments = data.get("moments", []) if isinstance(data, Mapping) else []
        return [("Formation spells", _as_frame(moments))]

    if table_id == "mean-position-summary":
        section = _section(bundle, "mean-positions")
        data = getattr(section, "data", {}) or {}
        rows = []
        for team in teams:
            payload = data.get(team, {}) if isinstance(data, Mapping) else {}
            summary = payload.get("summary", {}) if isinstance(payload, Mapping) else {}
            if isinstance(summary, Mapping):
                rows.append({"team_name": team, **summary})
        return [("Mean position summary", pd.DataFrame(rows))]

    if table_id == "pass-network-leaders":
        section = _section(bundle, "pass-network")
        data = getattr(section, "data", {}) or {}
        payloads = []
        for team in teams:
            nodes = _as_frame((data.get("teams", {}).get(team, {}) or {}).get("nodes"))
            payloads.append((f"Pass network leaders - {team}", nodes))
        return payloads

    if table_id == "progressive-pass-leaders":
        section = _section(bundle, "progressive-passes")
        data = getattr(section, "data", {}) or {}
        return [
            (
                f"Progressive pass leaders - {team}",
                _as_frame((data.get("teams", {}).get(team, {}) or {}).get("player_ranking")),
            )
            for team in teams
        ]

    if table_id == "final-third-entry-breakdown":
        section = _section(bundle, "final-third-entries")
        data = getattr(section, "data", {}) or {}
        rows = []
        for team in teams:
            stats = (data.get("teams", {}).get(team, {}) or {}).get("stats", {})
            if isinstance(stats, Mapping):
                rows.extend(
                    {"team_name": team, **row}
                    for row in _mapping_rows(stats)
                )
        return [("Final third entry breakdown", pd.DataFrame(rows))]

    if table_id == "pass-location-breakdown":
        section = _section(bundle, "pass-locations")
        data = getattr(section, "data", {}) or {}
        rows = []
        for team in teams:
            passes = _as_frame((data.get("teams", {}) or {}).get(team))
            rows.append({"team_name": team, "pass_attempts": int(len(passes))})
        return [("Pass location breakdown", pd.DataFrame(rows))]

    if table_id == "cross-top-routes":
        section = _section(bundle, "cross-flow")
        data = getattr(section, "data", {}) or {}
        return [
            (
                f"Top cross routes - {team}",
                _as_frame((data.get(team, {}) or {}).get("routes")),
            )
            for team in teams
        ]

    if table_id in {"build-up-summary", "build-up-sequences"}:
        section = _section(bundle, "build-up")
        data = getattr(section, "data", {}) or {}
        if table_id == "build-up-summary":
            comparison = data.get("comparison", {}) if isinstance(data, Mapping) else {}
            if isinstance(comparison, Mapping):
                return [("Build-up summary", pd.DataFrame(_mapping_rows(comparison)))]
            return [("Build-up summary", _as_frame(comparison))]
        payloads = []
        for team in teams:
            sequences = _as_frame(
                (data.get("teams", {}).get(team, {}) or {}).get("sequences")
            )
            id_column = next(
                (
                    column
                    for column in (
                        "sequence_id",
                        "trigger_sequence_id",
                        "buildup_sequence_id",
                        "id",
                    )
                    if column in sequences.columns
                ),
                None,
            )
            if id_column is not None and not sequences.empty:
                sequences = sequences.drop_duplicates(
                    subset=[id_column],
                    keep="last",
                )
            payloads.append((f"Build-up sequences - {team}", sequences))
        return payloads

    if table_id == "defensive-shape-summary":
        section = _section(bundle, "defensive-shape")
        data = getattr(section, "data", {}) or {}
        rows = []
        for team in teams:
            profile = data.get(team, {}) if isinstance(data, Mapping) else {}
            if isinstance(profile, Mapping):
                simple = {
                    key: value
                    for key, value in profile.items()
                    if not isinstance(value, (pd.DataFrame, Mapping, list, tuple))
                }
                rows.append({"team_name": team, **simple})
        return [("Defensive shape summary", pd.DataFrame(rows))]

    if table_id == "ppda-summary":
        section = _section(bundle, "ppda")
        data = getattr(section, "data", {}) or {}
        rows = []
        for team in teams:
            overall = (data.get("teams", {}).get(team, {}) or {}).get("overall", {})
            if isinstance(overall, Mapping):
                rows.append({"team_name": team, **overall})
        return [("PPDA summary", pd.DataFrame(rows))]

    if table_id in {
        "defensive-transitions-summary",
        "offensive-transitions-summary",
    }:
        section_id = (
            "defensive-transitions"
            if table_id.startswith("defensive")
            else "offensive-transitions"
        )
        section = _section(bundle, section_id)
        data = getattr(section, "data", {}) or {}
        rows = []
        for team in teams:
            stats = (data.get(team, {}) or {}).get("stats", {})
            if isinstance(stats, Mapping):
                for row in _mapping_rows(stats):
                    rows.append({"team_name": team, **row})
        return [(table_id.replace("-", " ").title(), pd.DataFrame(rows))]

    if table_id in {
        "defensive-transitions-sequences",
        "offensive-transitions-sequences",
    }:
        section_id = (
            "defensive-transitions"
            if table_id.startswith("defensive")
            else "offensive-transitions"
        )
        section = _section(bundle, section_id)
        data = getattr(section, "data", {}) or {}
        payloads = []
        for team in teams:
            combined = _as_frame((data.get(team, {}) or {}).get("combined"))
            id_column = next(
                (
                    column
                    for column in ("loss_sequence_id", "sequence_id", "id")
                    if column in combined.columns
                ),
                None,
            )
            if id_column is not None and not combined.empty:
                combined = combined.drop_duplicates(subset=[id_column], keep="last")
            payloads.append((f"Transition sequences - {team}", combined))
        return payloads

    if table_id in {"restart-summary", "restart-takers"}:
        section = _section(bundle, "restarts")
        data = getattr(section, "data", {}) or {}
        if table_id == "restart-summary":
            rows = []
            for team in teams:
                summary = (data.get(team, {}) or {}).get("summary", {})
                if isinstance(summary, Mapping):
                    for row in _mapping_rows(summary):
                        rows.append({"team_name": team, **row})
            return [("Restart summary", pd.DataFrame(rows))]
        return [
            (
                f"Restart takers - {team}",
                _as_frame((data.get(team, {}) or {}).get("records")),
            )
            for team in teams
        ]

    if table_id == "player-highlights-table":
        section = _section(bundle, "player-highlights")
        data = getattr(section, "data", {}) or {}
        payloads = []
        for label, key in (
            ("Passing", "player_stats"),
            ("Shooting", "shot_sequence_ranking"),
            ("Defending", "defensive_ranking"),
        ):
            frame = _as_frame(data.get(key))
            if not frame.empty:
                payloads.append((f"Player highlights - {label}", frame.head(6)))
        return payloads

    if table_id == "methodology-notes":
        section = _section(bundle, "methodology-appendix")
        data = getattr(section, "data", {}) or {}
        return [("Methodology notes", pd.DataFrame(_mapping_rows(data)))]

    if table_id == "metric-definitions":
        definitions = [
            {
                "Term": "Full Match",
                "Definition": "All report sections use the canonical full-match scope; current UI filters do not alter the report.",
            },
            {
                "Term": "Progressive pass",
                "Definition": "Classified by the canonical project progressive-pass metric; REPORT-05 does not reimplement the formula.",
            },
            {
                "Term": "Final third entry",
                "Definition": "Derived by the canonical final-third entry analyzer from qualifying passes and carries.",
            },
            {
                "Term": "PPDA",
                "Definition": "Uses the canonical defensive-metrics PPDA profile and its Full/1H/2H and timeline outputs.",
            },
            {
                "Term": "Representative sequence",
                "Definition": "Chosen deterministically by REPORT-03 from canonical sequence outcomes and tie-break rules; no weighted score is used.",
            },
            {
                "Term": "Top player",
                "Definition": "Chosen independently per team and metric family by the deterministic REPORT-03 selectors.",
            },
        ]
        return [("Metric definitions", pd.DataFrame(definitions))]

    if table_id == "data-quality-notes":
        overview = _section(bundle, "overview")
        data = getattr(overview, "data", {}) or {}
        coverage = data.get("data_coverage", {}) if isinstance(data, Mapping) else {}
        return [("Data quality notes", pd.DataFrame(_mapping_rows(coverage)))]

    return []


def _table_story_for_spec(bundle, table_spec, styles, config) -> list[Any]:
    payloads = _table_payloads(bundle, table_spec.id)
    if not payloads:
        return [
            _placeholder_box(
                f"No neutral table payload is available for {table_spec.title}.",
                styles,
            ),
            Spacer(1, 3 * mm),
        ]

    story: list[Any] = []
    for title, frame in payloads:
        story.append(Paragraph(escape(title), styles["subheading"]))
        story.append(
            _long_table(
                frame,
                styles,
                max_columns=config.max_table_columns,
            )
        )
        story.append(Spacer(1, 4 * mm))
    return story


def _overview_notes(bundle, styles) -> list[Any]:
    section = _section(bundle, "overview")
    data = getattr(section, "data", {}) or {}
    if not isinstance(data, Mapping):
        return []

    scorers = data.get("scorers", []) or []
    if not scorers:
        return []

    rows = []
    for item in scorers:
        if not isinstance(item, Mapping):
            continue
        minute = item.get("minute")
        second = item.get("second")
        clock = ""
        if minute is not None:
            clock = f"{minute}'"
        if second not in (None, 0, "0"):
            clock += f"{second}s"
        rows.append(
            {
                "Team": item.get("team_name"),
                "Scorer": item.get("scorer"),
                "Time": clock,
                "Type": item.get("goal_type"),
            }
        )

    if not rows:
        return []

    return [
        Paragraph("Goals", styles["subheading"]),
        _long_table(pd.DataFrame(rows), styles, max_columns=4),
        Spacer(1, 4 * mm),
    ]


def _appendix_manifest_table(manifest: ReportManifest) -> pd.DataFrame:
    rows = []
    for section in manifest.sections:
        rows.append(
            {
                "Order": section.order,
                "Section ID": section.id,
                "Title": section.title,
                "Required": bool(section.required),
                "Figures": ", ".join(item.id for item in section.figures) or "-",
                "Tables": ", ".join(item.id for item in section.tables) or "-",
            }
        )
    return pd.DataFrame(rows)


def _generation_notes(bundle, catalog, manifest, styles, config) -> list[Any]:
    methodology = _section(bundle, "methodology-appendix")
    methodology_data = getattr(methodology, "data", {}) or {}

    lines = [
        "This PDF is composed entirely in memory with ReportLab/Platypus.",
        "Figures come from the REPORT-04 static Plotly catalog and are converted to PNG bytes in memory via Plotly/Kaleido.",
        "Tables are native ReportLab tables with repeated headers and row splitting; they are not screenshots.",
        "A failed or empty report section is rendered as an explicit placeholder and does not abort document generation.",
    ]

    if isinstance(methodology_data, Mapping):
        contract = methodology_data.get("contract")
        if contract:
            lines.append(str(contract))

    story: list[Any] = [
        Paragraph("Generation manifest", styles["subheading"]),
    ]
    for line in lines:
        story.append(Paragraph(escape(line), styles["body"]))

    story.extend(
        [
            Paragraph("Canonical manifest", styles["subheading"]),
            _long_table(
                _appendix_manifest_table(manifest),
                styles,
                max_columns=6,
            ),
            Spacer(1, 4 * mm),
            Paragraph("Generation metadata", styles["subheading"]),
            _long_table(
                pd.DataFrame(
                    [
                        {
                            "manifest_id": getattr(catalog, "manifest_id", manifest.id),
                            "manifest_version": getattr(catalog, "manifest_version", manifest.schema_version),
                            "source_signature": getattr(bundle, "source_signature", ""),
                            "scope": getattr(getattr(bundle, "scope", None), "value", getattr(bundle, "scope", "")),
                            "figure_count": len(getattr(catalog, "figures", ()) or ()),
                            "image_scale": config.image_scale,
                        }
                    ]
                ),
                styles,
                max_columns=6,
            ),
        ]
    )
    return story


def _section_story(
    bundle,
    catalog,
    section_spec,
    styles,
    config,
    manifest,
) -> list[Any]:
    section = _section(bundle, section_spec.id)
    status = _section_status(section)

    story: list[Any] = []
    story.extend(_section_heading(section_spec, styles))

    if status == "error":
        message = (
            "This section failed during neutral data generation. "
            f"{getattr(section, 'error_type', '')}: {getattr(section, 'error_message', '')}"
        )
        story.extend([_placeholder_box(message, styles), Spacer(1, 4 * mm)])
    elif status == "skipped":
        story.extend(
            [
                _placeholder_box(
                    "This section was intentionally skipped by the report data bundle.",
                    styles,
                ),
                Spacer(1, 4 * mm),
            ]
        )
    elif status == "empty":
        story.extend(
            [
                _placeholder_box(
                    "No qualifying sample is available for this section. The report preserves the section so the absence of data is explicit.",
                    styles,
                ),
                Spacer(1, 4 * mm),
            ]
        )

    if section_spec.id == "overview":
        story.extend(_overview_notes(bundle, styles))

    for figure_spec in section_spec.figures:
        story.append(Paragraph(escape(figure_spec.title), styles["subheading"]))
        story.extend(
            _figure_story_for_spec(
                catalog,
                figure_spec,
                styles,
                config,
            )
        )

    for table_spec in section_spec.tables:
        story.extend(
            _table_story_for_spec(
                bundle,
                table_spec,
                styles,
                config,
            )
        )

    if section_spec.id == "methodology-appendix":
        story.extend(
            _generation_notes(
                bundle,
                catalog,
                manifest,
                styles,
                config,
            )
        )

    if (
        not section_spec.figures
        and not section_spec.tables
        and len(story) <= 4
    ):
        story.append(
            _placeholder_box(
                "No report content is declared for this section.",
                styles,
            )
        )

    return story


def render_match_report_pdf(
    bundle,
    catalog,
    manifest: ReportManifest = REPORT_MANIFEST,
    *,
    config: MatchReportPdfConfig | None = None,
) -> bytes:
    """Render a complete Match Analysis PDF and return its bytes.

    The function writes no persistent files. Plotly figures are converted to
    image bytes in memory and all ReportLab output is written to a BytesIO.
    """

    config = config or MatchReportPdfConfig()
    styles = _styles(config)
    buffer = BytesIO()
    doc = _doc(buffer, config)

    story: list[Any] = []
    story.extend(_cover_story(bundle, config, styles))
    story.append(PageBreak())
    story.extend(_toc_story(styles))

    for section_spec in manifest.sections:
        story.append(PageBreak())
        try:
            story.extend(
                _section_story(
                    bundle,
                    catalog,
                    section_spec,
                    styles,
                    config,
                    manifest,
                )
            )
        except Exception as exc:
            # Last-resort section isolation: a ReportLab composition problem in
            # one section must not prevent the remaining document from building.
            story.extend(_section_heading(section_spec, styles))
            story.append(
                _placeholder_box(
                    "Section composition failed: "
                    f"{type(exc).__name__}: {exc}",
                    styles,
                )
            )

    doc.multiBuild(story)
    return buffer.getvalue()


def render_match_report_pdf_buffer(
    bundle,
    catalog,
    manifest: ReportManifest = REPORT_MANIFEST,
    *,
    config: MatchReportPdfConfig | None = None,
) -> BytesIO:
    """Return the same REPORT-05 PDF in a rewound BytesIO buffer."""

    buffer = BytesIO(
        render_match_report_pdf(
            bundle,
            catalog,
            manifest,
            config=config,
        )
    )
    buffer.seek(0)
    return buffer
