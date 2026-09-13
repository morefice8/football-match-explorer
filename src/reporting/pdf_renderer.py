"""In-memory, brand-aware PDF renderer for Match Analysis reports.

REPORT-05 composes the neutral REPORT-02 bundle and the static REPORT-04
figure catalog with ReportLab/Platypus. It does not import Dash, app.py,
HTML renderers, browser automation or screenshot tooling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import unicodedata
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
    PageBreakIfNotEmpty,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

from src.reporting.manifest import REPORT_MANIFEST
from src.reporting.models import ReportManifest
from src.reporting.selectors import (
    rank_players_by_metric_family,
    rank_sequences_by_outcome_priority,
)
from src.reporting.render_audit import artifact_key

logger = logging.getLogger(__name__)


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
    # REPORT-12: the PDF is an editorial reading layer, not a data dump.
    # Eight deliberately selected columns remain legible on A4 landscape;
    # complete analytical schemas continue to live in the CSV files.
    max_table_columns: int = 8
    brand_font_path: str | Path | None = None
    enable_brand_font: bool = True
    render_audit: dict | None = field(default=None, compare=False, repr=False)
    # Selection reasons remain in the machine-readable catalog/manifest.
    # The editorial PDF hides implementation-level ranking traces by default.
    include_selection_reasons: bool = False

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


def _with_named_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Materialise a meaningful player/sequence index for PDF display.

    Several canonical ranking frames use ``playerName`` as their index.  The
    previous renderer silently dropped that identity and printed only metric
    columns.  REPORT-12 keeps the identity while still leaving the underlying
    bundle/CSV payload untouched.
    """

    frame = frame.copy()
    if isinstance(frame.index, pd.RangeIndex) and frame.index.name is None:
        return frame
    index_name = str(frame.index.name or "item")
    if index_name in frame.columns:
        index_name = "item"
    return frame.reset_index(names=index_name)


def _column_lookup(frame: pd.DataFrame, *aliases: str) -> str | None:
    by_name = {
        str(column).strip().casefold(): column
        for column in frame.columns
    }
    for alias in aliases:
        found = by_name.get(str(alias).strip().casefold())
        if found is not None:
            return found
    return None


def _explicit_columns(
    frame: pd.DataFrame,
    columns: Sequence[str | Sequence[str]],
) -> pd.DataFrame:
    """Return only explicitly declared PDF columns that exist in ``frame``."""

    frame = _with_named_index(frame)
    selected: list[Any] = []
    for candidate in columns:
        aliases = (candidate,) if isinstance(candidate, str) else tuple(candidate)
        column = _column_lookup(frame, *aliases)
        if column is not None and column not in selected:
            selected.append(column)
    return frame.loc[:, selected].copy() if selected else pd.DataFrame()


def _is_nested_pdf_value(value: Any) -> bool:
    return isinstance(value, (pd.DataFrame, pd.Series, Mapping, list, tuple, set))


def _drop_nested_pdf_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Never render nested analytical payloads inside native PDF tables."""

    if frame.empty:
        return frame.copy()
    keep: list[Any] = []
    for column in frame.columns:
        series = frame[column]
        if any(_is_nested_pdf_value(value) for value in series if value is not None):
            continue
        keep.append(column)
    return frame.loc[:, keep].copy()


def _metric_rows(
    frame: pd.DataFrame,
    *,
    exact: Sequence[str] = (),
    prefixes: Sequence[str] = (),
    limit: int | None = None,
) -> pd.DataFrame:
    """Select a compact allow-listed subset from a Metric/Value table."""

    if frame.empty or "Metric" not in frame.columns:
        return frame.head(0).copy()
    exact_keys = {item.casefold() for item in exact}
    prefix_keys = tuple(item.casefold() for item in prefixes)
    mask = frame["Metric"].astype(str).map(
        lambda value: (
            value.casefold() in exact_keys
            or any(value.casefold().startswith(prefix) for prefix in prefix_keys)
        )
    )
    result = frame.loc[mask].copy()
    if limit is not None:
        result = result.head(limit)
    return result


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


def _pdf_safe_helvetica_text(value: Any) -> str:
    """Return text that core Helvetica can render without black-square glyphs.

    ReportLab's core Helvetica font uses WinAnsi/CP1252.  Football names can
    contain Latin characters outside that repertoire (for example č/ć).
    Preserve characters supported by CP1252 and transliterate only unsupported
    code points to an ASCII approximation.  The analytical source values remain
    untouched in CSV/JSON exports.
    """

    text = str(value)
    # Normalize common typographic punctuation to ASCII before the generic
    # CP1252/transliteration pass.  Formation timeline labels use prime and
    # double-prime symbols (e.g. 31′ 05″), which core Helvetica cannot render
    # and would otherwise become question marks in the PDF.
    punctuation = {
        "\u2032": "'",   # prime / minutes
        "\u2033": '"',   # double prime / seconds
        "\u2018": "'",   # left single quotation mark
        "\u2019": "'",   # right single quotation mark
        "\u201c": '"',   # left double quotation mark
        "\u201d": '"',   # right double quotation mark
        "\u2013": "-",   # en dash
        "\u2014": "-",   # em dash
    }
    text = "".join(punctuation.get(char, char) for char in text)

    output: list[str] = []
    for char in text:
        try:
            char.encode("cp1252")
        except UnicodeEncodeError:
            decomposed = unicodedata.normalize("NFKD", char)
            ascii_part = "".join(
                item
                for item in decomposed
                if ord(item) < 128 and not unicodedata.combining(item)
            )
            output.append(ascii_part or "?")
        else:
            output.append(char)
    return "".join(output)


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

"table_subheading": ParagraphStyle(
    "ReportTableSubheading",
    parent=sample["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=12,
    leading=15,
    textColor=TEXT,
    spaceBefore=7,
    spaceAfter=6,
    keepWithNext=False,
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
            fontSize=7.4,
            leading=9.2,
            textColor=TEXT,
        ),
        "table_header": ParagraphStyle(
            "ReportTableHeader",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.2,
            leading=9.0,
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
    result = (
        overview_data.get("result", {})
        if isinstance(overview_data, Mapping)
        else {}
    )
    metadata = (
        overview_data.get("metadata", {})
        if isinstance(overview_data, Mapping)
        else {}
    )
    merged = {
        **info,
        **(metadata if isinstance(metadata, Mapping) else {}),
    }

    home_score = (
        result.get("home_score")
        if isinstance(result, Mapping)
        else None
    )
    away_score = (
        result.get("away_score")
        if isinstance(result, Mapping)
        else None
    )
    if home_score in (None, ""):
        home_score = _first_present(
            merged,
            "hteamScore",
            "home_score",
            "homeScore",
        )
    if away_score in (None, ""):
        away_score = _first_present(
            merged,
            "ateamScore",
            "away_score",
            "awayScore",
        )

    score = "-"
    if home_score not in (None, "") or away_score not in (None, ""):
        score = (
            f"{home_score if home_score not in (None, '') else '-'}"
            " - "
            f"{away_score if away_score not in (None, '') else '-'}"
        )

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
        "date_formatted",
        "game_date",
        "match_date",
        "matchDate",
        "date",
        "startDate",
        "start_date",
        "date_iso",
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
    table = Table([[content]], colWidths=[268 * mm], rowHeights=[height])
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
    audit = config.render_audit
    record = audit.get(artifact_key(artifact)) if audit is not None else None

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
        logger.exception("PDF figure export failed: %s", artifact.id)
        if record is not None:
            record.update(render_status="error", error_type=type(exc).__name__,
                          error_message="PNG conversion failed.")
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
        if record is not None:
            status = getattr(artifact.status, "value", artifact.status)
            record.update(
                render_status=status,
                error_type=(
                    (getattr(artifact, "error_type", None) or "FigureConstructionError")
                    if status == "error" else None
                ),
                error_message="Figure construction failed." if status == "error" else None,
            )

    reason = getattr(artifact, "selection_reason", None)
    if reason and config.include_selection_reasons:
        flowables.append(
            Paragraph(
                "<b>Selection reason:</b> " + escape(_pdf_safe_helvetica_text(reason)),
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
            # Two-team figures are deliberately compact so editorial sections
            # can keep visual comparison + concise tables on one/two pages.
            max_cell_width = 92 * mm
            max_cell_height = 54 * mm
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
                colWidths=[133 * mm, 133 * mm],
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
                max_width=220 * mm,
                max_height=92 * mm,
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


_PDF_TABLE_COLUMNS: dict[str, tuple[str | tuple[str, ...], ...]] = {
    "match-overview": (
        "team_name",
        "passes",
        "pass_completion_pct",
        "shots",
        "shots_on_target",
        "progressive_passes",
        "final_third_entries",
        "crosses",
    ),
    "data-coverage": ("Metric", "Value"),
    "formation-spells": (
        ("time", "time_label", "minute", "timeMin", "start_minute"),
        ("score", "Score"),
        ("home_formation", "home_formation_name", "Home Formation", "homeFormation"),
        ("away_formation", "away_formation_name", "Away Formation", "awayFormation"),
        ("change", "event_summary", "Reason", "change_reason"),
    ),
    "mean-position-summary": (
        "team_name",
        "touches",
        "team_length_m",
        "team_width_m",
        "team_compactness_m",
        "average_height_m",
    ),
    "pass-network-leaders": (
        "team_name",
        "playerName",
        "jersey_number",
        "pass_sent",
        "pass_received",
        "pass_involvement",
        "minutes",
    ),
    "progressive-pass-leaders": (
        "team_name",
        "Player",
        "Successful",
        "Attempted",
        "Completion %",
        "Progression m",
    ),
    "final-third-entry-breakdown": ("team_name", "entries", "passes", "carries", "left", "central", "right", "zone14"),
    "pass-location-breakdown": ("team_name", "pass_attempts"),
    "cross-top-routes": (
        "Origin Zone",
        "Destination Zone",
        "Crosses",
        "Share %",
        "Completion %",
        "Retention %",
        "Shots",
        "Shot Rate %",
    ),
    "build-up-summary": (
        "team_name",
        "sequences",
        "avg_duration_s",
        "avg_completed_passes",
    ),
    "build-up-sequences": (
        "sequence_id",
        "start_zone",
        "buildup_type",
        "terminal_outcome",
        "termination_reason",
        "pass_count",
        "duration_seconds",
    ),
    "defensive-shape-summary": (
        "team_name",
        "block_height_m",
        "width_m",
        "compactness_m",
        "action_count",
        "snapshot_count",
    ),
    "ppda-summary": (
        "team_name",
        "ppda",
        "first_half",
        "second_half",
        "opponent_passes",
        "defensive_actions",
    ),
    "defensive-transitions-summary": (
        "team_name", "transitions", "goals", "shots", "consolidated", "regained", "dominant_channel",
    ),
    "offensive-transitions-summary": (
        "team_name", "transitions", "goals", "shots", "consolidated", "regained", "dominant_channel",
    ),
    "defensive-transitions-sequences": (
        "sequence_id",
        "minute",
        "player_name",
        "start_zone",
        "sequence_outcome",
        "terminal_outcome",
        "event_count",
        "duration_seconds",
    ),
    "offensive-transitions-sequences": (
        "sequence_id",
        "minute",
        "player_name",
        "start_zone",
        "sequence_outcome",
        "terminal_outcome",
        "event_count",
        "duration_seconds",
    ),
    "restart-summary": (
        "team_name",
        "restarts",
        "corners",
        "free_kicks",
        "throw_ins",
        "goal_kicks",
        "shots",
    ),
    "restart-takers": (
        "player_name",
        "restart_count",
        "primary_restart",
        "shots",
    ),
    "player-highlights-table": (
        ("playerName", "player_name", "Player", "item"),
        "team_name",
        "Offensive Pass Contributions",
        "Progressive Passes",
        "Passes into Box",
        "Key Passes",
        "Successful Passes",
        "Shot Sequence Involvements",
        "Shot Sequence Shots",
        "Shot Sequence Shot Assists",
        "Shot Sequence Pre-Assists",
        ("unique", "Unique Defensive Contributions"),
        ("tackles_won", "Tackles Won"),
        ("interceptions", "Interceptions"),
        ("recoveries", "Recoveries"),
        ("clearances", "Clearances"),
    ),
    "methodology-notes": ("Metric", "Value"),
    "metric-definitions": ("Term", "Definition"),
    "data-quality-notes": ("Metric", "Value"),
}


def _format_table_cell(value: Any) -> str:
    """Return a bounded, human-readable representation for a PDF table cell."""

    if isinstance(value, (pd.DataFrame, pd.Series, Mapping, list, tuple, set)):
        # Editorial PDFs never expose nested analytical structures.  Those
        # values remain available in report-data.json and the CSV tables.
        return "-"
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
    return _pdf_safe_helvetica_text(text)


def _long_table(
    frame: pd.DataFrame,
    styles,
    *,
    max_columns: int,
    available_width: float = 268 * mm,
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


def _sequence_summary_frame(
    frame: pd.DataFrame,
    *,
    id_aliases: Sequence[str],
    zone_aliases: Sequence[str] = (),
) -> pd.DataFrame:
    """Collapse event-level sequence rows into one editorial row per sequence."""

    frame = _as_frame(frame)
    if frame.empty:
        return pd.DataFrame()
    id_column = _column_lookup(frame, *id_aliases)
    if id_column is None:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for sequence_id, group in frame.groupby(id_column, dropna=True, sort=False):
        row: dict[str, Any] = {"sequence_id": sequence_id}

        minute_col = _column_lookup(group, "timeMin", "minute")
        if minute_col is not None:
            minutes = pd.to_numeric(group[minute_col], errors="coerce").dropna()
            if not minutes.empty:
                row["minute"] = int(minutes.min())

        seconds_col = _column_lookup(group, "total_seconds", "match_second")
        if seconds_col is not None:
            seconds = pd.to_numeric(group[seconds_col], errors="coerce").dropna()
            if not seconds.empty:
                row["duration_seconds"] = round(float(seconds.max() - seconds.min()), 1)

        row["event_count"] = int(len(group))

        for output, aliases in (
            ("player_name", ("playerName", "player_name", "Player")),
            ("start_zone", tuple(zone_aliases)),
            ("sequence_outcome", ("sequence_outcome_type", "final_outcome", "outcome")),
            ("terminal_outcome", ("terminal_outcome", "result")),
            ("termination_reason", ("termination_reason",)),
            ("pass_count", ("buildup_pass_count", "pass_count", "opponent_pass_count")),
        ):
            if not aliases:
                continue
            column = _column_lookup(group, *aliases)
            if column is None:
                continue
            values = group[column].dropna()
            if not values.empty:
                row[output] = values.iloc[-1]
        rows.append(row)
    return pd.DataFrame(rows)


def _top_sequence_rows(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Apply the canonical TOP_SEQUENCES selector ordering."""

    return rank_sequences_by_outcome_priority(frame, limit=limit)


def _build_up_summary_rows(bundle) -> pd.DataFrame:
    section = _section(bundle, "build-up")
    data = getattr(section, "data", {}) or {}
    comparison = data.get("comparison", {}) if isinstance(data, Mapping) else {}
    teams = tuple(getattr(bundle, "teams", ()) or ())
    if not isinstance(comparison, Mapping):
        return pd.DataFrame()

    profile = comparison.get("profile", {}) or {}
    rows = []
    for index, team in enumerate(teams[:2]):
        side = "home" if index == 0 else "away"
        side_profile = profile.get(side, {}) if isinstance(profile, Mapping) else {}
        rows.append(
            {
                "team_name": team,
                "sequences": comparison.get(f"{side}_total"),
                "avg_duration_s": (
                    side_profile.get("avg_duration_seconds")
                    if isinstance(side_profile, Mapping)
                    else None
                ),
                "avg_completed_passes": (
                    side_profile.get("avg_completed_passes")
                    if isinstance(side_profile, Mapping)
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def _final_third_summary_rows(bundle) -> pd.DataFrame:
    section = _section(bundle, "final-third-entries")
    data = getattr(section, "data", {}) or {}
    teams = tuple(getattr(bundle, "teams", ()) or ())
    rows = []
    for team in teams:
        stats = (data.get("teams", {}).get(team, {}) or {}).get("stats", {}) if isinstance(data, Mapping) else {}
        if not isinstance(stats, Mapping):
            continue
        rows.append(
            {
                "team_name": team,
                "entries": stats.get("total_final_third", 0),
                "passes": stats.get("pass_entries", 0),
                "carries": stats.get("carry_entries", 0),
                "left": stats.get("channel_left", 0),
                "central": stats.get("channel_central", 0),
                "right": stats.get("channel_right", 0),
                "zone14": stats.get("zone14", 0),
            }
        )
    return pd.DataFrame(rows)


def _transition_summary_rows(bundle, section_id: str) -> pd.DataFrame:
    section = _section(bundle, section_id)
    data = getattr(section, "data", {}) or {}
    teams = tuple(getattr(bundle, "teams", ()) or ())
    rows = []
    for team in teams:
        stats = (data.get(team, {}) or {}).get("stats", {}) if isinstance(data, Mapping) else {}
        if not isinstance(stats, Mapping):
            continue
        outcomes = stats.get("outcomes", {}) or {}
        flanks = stats.get("flanks", {}) or {}

        def summed(token: str) -> int:
            if not isinstance(outcomes, Mapping):
                return 0
            return int(sum(float(v or 0) for k, v in outcomes.items() if token in str(k).casefold()))

        dominant = "-"
        if isinstance(flanks, Mapping) and flanks:
            dominant = str(max(flanks.items(), key=lambda item: float(item[1] or 0))[0])
        rows.append(
            {
                "team_name": team,
                "transitions": stats.get("total", 0),
                "goals": summed("goal"),
                "shots": summed("shot"),
                "consolidated": summed("consolid"),
                "regained": summed("regain"),
                "dominant_channel": dominant,
            }
        )
    return pd.DataFrame(rows)


def _ppda_summary_rows(bundle) -> pd.DataFrame:
    section = _section(bundle, "ppda")
    data = getattr(section, "data", {}) or {}
    teams = tuple(getattr(bundle, "teams", ()) or ())
    rows = []
    for team in teams:
        payload = (data.get("teams", {}).get(team, {}) or {}) if isinstance(data, Mapping) else {}
        overall = payload.get("overall", {}) if isinstance(payload, Mapping) else {}
        first = payload.get("first_half", {}) if isinstance(payload, Mapping) else {}
        second = payload.get("second_half", {}) if isinstance(payload, Mapping) else {}
        rows.append(
            {
                "team_name": team,
                "ppda": overall.get("ppda") if isinstance(overall, Mapping) else None,
                "first_half": first.get("ppda") if isinstance(first, Mapping) else None,
                "second_half": second.get("ppda") if isinstance(second, Mapping) else None,
                "opponent_passes": overall.get("opponent_passes") if isinstance(overall, Mapping) else None,
                "defensive_actions": overall.get("defensive_actions") if isinstance(overall, Mapping) else None,
            }
        )
    return pd.DataFrame(rows)


def _formation_spells_rows(moments: Any) -> pd.DataFrame:
    """Flatten formation timeline moments into an editorial scalar table."""

    rows: list[dict[str, Any]] = []
    labels = {
        "starting_xi": "Starting XI",
        "goal": "Goal",
        "substitution": "Substitution",
        "formation_change": "Formation change",
        "red_card": "Red card",
    }
    for moment in moments or []:
        if not isinstance(moment, Mapping):
            continue
        events = moment.get("events") or []
        changes: list[str] = []
        if isinstance(events, Sequence) and not isinstance(events, (str, bytes)):
            for event in events:
                if not isinstance(event, Mapping):
                    continue
                kind = str(event.get("kind") or "").strip()
                label = labels.get(kind, kind.replace("_", " ").title() if kind else "")
                team = str(event.get("team") or "").strip()
                value = f"{team}: {label}" if team and team.casefold() != "both" else label
                if value and value not in changes:
                    changes.append(value)
        rows.append(
            {
                "time": moment.get("time_label") or moment.get("minute"),
                "score": moment.get("score"),
                "home_formation": moment.get("home_formation_name"),
                "away_formation": moment.get("away_formation_name"),
                "change": ", ".join(changes) if changes else "-",
            }
        )
    return pd.DataFrame(rows)


def _restart_summary_rows(bundle) -> pd.DataFrame:
    section = _section(bundle, "restarts")
    data = getattr(section, "data", {}) or {}
    teams = tuple(getattr(bundle, "teams", ()) or ())
    rows = []
    for team in teams:
        summary = (data.get(team, {}) or {}).get("summary", {}) if isinstance(data, Mapping) else {}
        if not isinstance(summary, Mapping):
            continue

        def count(mapping_name: str, token: str) -> int:
            mapping = summary.get(mapping_name, {}) or {}
            if not isinstance(mapping, Mapping):
                return 0
            return int(
                sum(
                    float(value or 0)
                    for key, value in mapping.items()
                    if token in str(key).casefold()
                )
            )

        rows.append(
            {
                "team_name": team,
                "restarts": summary.get("total", 0),
                "corners": count("action_types", "corner"),
                "free_kicks": count("action_types", "free kick"),
                "throw_ins": count("action_types", "throw"),
                "goal_kicks": count("action_types", "goal kick"),
                "shots": count("development_outcomes", "shot"),
            }
        )
    return pd.DataFrame(rows)


def _restart_takers(records: Any, limit: int) -> pd.DataFrame:
    frame = _as_frame(records)
    if frame.empty:
        return frame
    player = _column_lookup(frame, "player_name", "playerName", "Player")
    if player is None:
        return pd.DataFrame()
    jersey = _column_lookup(frame, "jersey_number", "jersey", "Mapped Jersey Number")
    outcome = _column_lookup(frame, "outcome", "Outcome")
    development = _column_lookup(frame, "development_outcome", "Development Outcome")
    restart_type = _column_lookup(frame, "restart_type", "Action Type")

    rows = []
    for player_name, group in frame.groupby(player, dropna=True, sort=False):
        row = {
            "player_name": player_name,
            "restart_count": int(len(group)),
        }
        if jersey is not None:
            values = group[jersey].dropna()
            row["jersey"] = values.iloc[0] if not values.empty else None
        if restart_type is not None:
            modes = group[restart_type].dropna().astype(str).value_counts()
            row["primary_restart"] = modes.index[0] if not modes.empty else None
        if outcome is not None:
            row["successful"] = int(
                group[outcome].astype(str).str.contains("success", case=False, na=False).sum()
            )
        if development is not None:
            row["shots"] = int(
                group[development].astype(str).str.contains("shot", case=False, na=False).sum()
            )
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["restart_count", "player_name"],
        ascending=[False, True],
        kind="stable",
    ).head(limit)


def _player_team_lookup(data: Mapping[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    options = data.get("player_options", {}) if isinstance(data, Mapping) else {}
    if not isinstance(options, Mapping):
        return lookup
    for team, families in options.items():
        if not isinstance(families, Mapping):
            continue
        for payload in families.values():
            if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
                continue
            for item in payload:
                if isinstance(item, Mapping):
                    name = item.get("player_name") or item.get("playerName")
                else:
                    name = item
                if name:
                    lookup[str(name)] = str(team)
    return lookup


def _player_highlight_payloads(bundle, limit: int) -> list[tuple[str, pd.DataFrame]]:
    section = _section(bundle, "player-highlights")
    data = getattr(section, "data", {}) or {}
    if not isinstance(data, Mapping):
        return []
    team_lookup = _player_team_lookup(data)
    teams = tuple(getattr(bundle, "teams", ()) or ())

    families = (
        (
            "Passing",
            "player_stats",
            (
                ("playerName", "player_name", "Player"),
                "team_name",
                "Offensive Pass Contributions",
                "Progressive Passes",
                "Passes into Box",
                "Key Passes",
                "Successful Passes",
            ),
            ("Offensive Pass Contributions", "Progressive Passes", "Successful Passes"),
        ),
        (
            "Shooting",
            "shot_sequence_ranking",
            (
                ("playerName", "player_name", "Player"),
                "team_name",
                "Shot Sequence Involvements",
                "Shot Sequence Shots",
                "Shot Sequence Shot Assists",
                "Shot Sequence Pre-Assists",
            ),
            ("Shot Sequence Involvements", "Shot Sequence Shots"),
        ),
        (
            "Defending",
            "defensive_ranking",
            (
                ("playerName", "player_name", "Player"),
                "team_name",
                ("unique", "Unique Defensive Contributions"),
                ("tackles_won", "Tackles Won"),
                ("interceptions", "Interceptions"),
                ("recoveries", "Recoveries"),
                ("clearances", "Clearances"),
            ),
            ("unique", "tackles_won", "interceptions"),
        ),
    )

    payloads: list[tuple[str, pd.DataFrame]] = []
    for label, key, columns, _sort_aliases in families:
        frame = _with_named_index(_as_frame(data.get(key)))
        if frame.empty:
            continue
        player_col = _column_lookup(frame, "playerName", "player_name", "Player", "item")
        team_col = _column_lookup(frame, "team_name", "teamName", "Team")
        if player_col is None:
            continue
        if team_col is None:
            frame["team_name"] = frame[player_col].astype(str).map(team_lookup)
            team_col = "team_name"

        selected_rows = []
        category = label.casefold()
        for team in teams:
            scoped = rank_players_by_metric_family(
                frame,
                str(team),
                category=category,
                limit=limit,
            )
            if not scoped.empty:
                selected_rows.append(scoped)
        if not selected_rows:
            continue
        selected = pd.concat(selected_rows, ignore_index=True, sort=False)
        selected = _explicit_columns(selected, columns)
        payloads.append((f"Player highlights - {label}", selected))
    return payloads


def _table_payloads(
    bundle,
    table_id: str,
    *,
    selection_limit: int | None = None,
) -> list[tuple[str, pd.DataFrame]]:
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
            frame = _explicit_columns(
                frame,
                (
                    "team_name",
                    "passes",
                    "pass_completion_pct",
                    "shots",
                    "shots_on_target",
                    "progressive_passes",
                    "final_third_entries",
                    "crosses",
                ),
            )
            return [("Match overview", frame)]
        coverage = data.get("data_coverage", {}) if isinstance(data, Mapping) else {}
        frame = pd.DataFrame(_mapping_rows(coverage))
        frame = _metric_rows(
            frame,
            exact=(
                "event_rows",
                "receiver.eligible",
                "receiver.resolved",
                "receiver.coverage_pct",
                "coordinates.coverage_pct",
                "outcome.known_pct",
            ),
            prefixes=("final_third_carries.",),
            limit=10,
        )
        return [("Data coverage", frame)]

    if table_id == "formation-spells":
        section = _section(bundle, "formation-timeline")
        data = getattr(section, "data", {}) or {}
        moments = data.get("moments", []) if isinstance(data, Mapping) else []
        return [("Formation spells", _formation_spells_rows(moments))]

    if table_id == "mean-position-summary":
        section = _section(bundle, "mean-positions")
        data = getattr(section, "data", {}) or {}
        rows = []
        for team in teams:
            payload = data.get(team, {}) if isinstance(data, Mapping) else {}
            summary = payload.get("summary", {}) if isinstance(payload, Mapping) else {}
            if isinstance(summary, Mapping):
                rows.append({"team_name": team, **summary})
        frame = _explicit_columns(
            pd.DataFrame(rows),
            (
                "team_name",
                "touches",
                "team_length_m",
                "team_width_m",
                "team_compactness_m",
                "average_height_m",
            ),
        )
        return [("Mean position summary", frame)]

    if table_id == "pass-network-leaders":
        section = _section(bundle, "pass-network")
        data = getattr(section, "data", {}) or {}
        frames = []
        for team in teams:
            nodes = _as_frame((data.get("teams", {}).get(team, {}) or {}).get("nodes"))
            involvement = _column_lookup(nodes, "pass_involvement")
            if involvement is not None:
                nodes = nodes.sort_values(involvement, ascending=False, kind="stable")
            nodes = _explicit_columns(
                nodes.head(5),
                (
                    "playerName",
                    "jersey_number",
                    "pass_sent",
                    "pass_received",
                    "pass_involvement",
                    "minutes",
                ),
            )
            if not nodes.empty:
                nodes.insert(0, "team_name", team)
                frames.append(nodes)
        combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        return [("Pass network leaders", combined)]

    if table_id == "progressive-pass-leaders":
        section = _section(bundle, "progressive-passes")
        data = getattr(section, "data", {}) or {}
        frames = []
        for team in teams:
            frame = _explicit_columns(
                _as_frame((data.get("teams", {}).get(team, {}) or {}).get("player_ranking")).head(5),
                ("Player", "Successful", "Attempted", "Completion %", "Progression m"),
            )
            if not frame.empty:
                frame.insert(0, "team_name", team)
                frames.append(frame)
        combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        return [("Progressive pass leaders", combined)]

    if table_id == "final-third-entry-breakdown":
        return [("Final third entry breakdown", _final_third_summary_rows(bundle))]

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
        payloads = []
        for team in teams:
            routes = _as_frame((data.get(team, {}) or {}).get("routes"))
            crosses = _column_lookup(routes, "Crosses", "crosses")
            if crosses is not None:
                routes = routes.sort_values(crosses, ascending=False, kind="stable")
            routes = routes.head(selection_limit or 8)
            routes = _explicit_columns(
                routes,
                (
                    "Origin Zone",
                    "Destination Zone",
                    "Crosses",
                    "Share %",
                    "Completion %",
                    "Retention %",
                    "Shots",
                    "Shot Rate %",
                ),
            )
            payloads.append((f"Top cross routes - {team}", routes))
        return payloads

    if table_id in {"build-up-summary", "build-up-sequences"}:
        section = _section(bundle, "build-up")
        data = getattr(section, "data", {}) or {}
        if table_id == "build-up-summary":
            return [("Build-up summary", _build_up_summary_rows(bundle))]
        payloads = []
        for team in teams:
            payload = data.get("teams", {}).get(team, {}) or {}
            sequences = _as_frame(payload.get("summary"))
            if sequences.empty:
                sequences = _sequence_summary_frame(
                    _as_frame(payload.get("sequences")),
                    id_aliases=("trigger_sequence_id", "sequence_id", "buildup_sequence_id", "id"),
                    zone_aliases=("trigger_zone", "start_zone"),
                )
            else:
                id_column = _column_lookup(
                    sequences,
                    "sequence_id",
                    "trigger_sequence_id",
                    "buildup_sequence_id",
                    "id",
                )
                if id_column is not None:
                    sequences = sequences.rename(columns={id_column: "sequence_id"})
            sequences = _top_sequence_rows(sequences, selection_limit or 10)
            sequences = _explicit_columns(
                sequences,
                (
                    "sequence_id",
                    ("start_zone", "trigger_zone"),
                    ("buildup_type", "type_of_initial_trigger"),
                    ("terminal_outcome", "final_outcome"),
                    "termination_reason",
                    ("pass_count", "buildup_pass_count"),
                    ("duration_seconds", "buildup_active_duration_seconds"),
                ),
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
        frame = _explicit_columns(
            pd.DataFrame(rows),
            (
                "team_name",
                "block_height_m",
                "width_m",
                "compactness_m",
                "action_count",
                "snapshot_count",
            ),
        )
        return [("Defensive shape summary", frame)]

    if table_id == "ppda-summary":
        return [("PPDA summary", _ppda_summary_rows(bundle))]

    if table_id in {
        "defensive-transitions-summary",
        "offensive-transitions-summary",
    }:
        section_id = (
            "defensive-transitions"
            if table_id.startswith("defensive")
            else "offensive-transitions"
        )
        return [
            (
                table_id.replace("-", " ").title(),
                _transition_summary_rows(bundle, section_id),
            )
        ]

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
            summary = _sequence_summary_frame(
                combined,
                id_aliases=("loss_sequence_id", "sequence_id", "id"),
                zone_aliases=("loss_zone", "recovery_zone"),
            )
            summary = _top_sequence_rows(summary, selection_limit or 10)
            summary = _explicit_columns(
                summary,
                (
                    "sequence_id",
                    "minute",
                    "player_name",
                    "start_zone",
                    "sequence_outcome",
                    "terminal_outcome",
                    "event_count",
                    "duration_seconds",
                ),
            )
            payloads.append((f"Transition sequences - {team}", summary))
        return payloads

    if table_id in {"restart-summary", "restart-takers"}:
        section = _section(bundle, "restarts")
        data = getattr(section, "data", {}) or {}
        if table_id == "restart-summary":
            return [("Restart summary", _restart_summary_rows(bundle))]
        return [
            (
                f"Restart takers - {team}",
                _restart_takers(
                    (data.get(team, {}) or {}).get("records"),
                    selection_limit or 6,
                ),
            )
            for team in teams
        ]

    if table_id == "player-highlights-table":
        return _player_highlight_payloads(bundle, selection_limit or 3)

    if table_id == "methodology-notes":
        section = _section(bundle, "methodology-appendix")
        data = getattr(section, "data", {}) or {}
        frame = pd.DataFrame(_mapping_rows(data))
        frame = _metric_rows(
            frame,
            exact=("scope",),
        )
        return [("Methodology notes", frame)]

    if table_id == "metric-definitions":
        definitions = [
            {
                "Term": "Full Match",
                "Definition": "All report sections use the canonical full-match scope; current UI filters do not alter the report.",
            },
            {
                "Term": "Progressive pass",
                "Definition": "Classified by the project's canonical progressive-pass metric.",
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
                "Definition": "Chosen deterministically from canonical sequence outcomes and declared tie-break rules; no weighted score is used.",
            },
            {
                "Term": "Top player",
                "Definition": "Chosen independently per team and metric family using deterministic ranking rules.",
            },
        ]
        return [("Metric definitions", pd.DataFrame(definitions))]

    if table_id == "data-quality-notes":
        overview = _section(bundle, "overview")
        data = getattr(overview, "data", {}) or {}
        coverage = data.get("data_coverage", {}) if isinstance(data, Mapping) else {}
        frame = pd.DataFrame(_mapping_rows(coverage))
        frame = _metric_rows(
            frame,
            exact=(
                "event_rows",
                "receiver.eligible",
                "receiver.resolved",
                "receiver.coverage_pct",
                "coordinates.coverage_pct",
                "outcome.known_pct",
            ),
            limit=6,
        )
        return [("Data quality notes", frame)]

    return []


def _prepare_pdf_table_frame(table_id: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Enforce REPORT-12's explicit, scalar-only PDF table contract."""

    prepared = _drop_nested_pdf_columns(_as_frame(frame))
    policy = _PDF_TABLE_COLUMNS.get(table_id)
    if policy is not None:
        prepared = _explicit_columns(prepared, policy)
    return prepared


def _paired_restart_takers_frame(
    prepared: Sequence[tuple[str, pd.DataFrame]],
) -> pd.DataFrame | None:
    """Place two team top-taker lists in one eight-column comparison table."""

    if len(prepared) != 2:
        return None
    (left_title, left), (right_title, right) = prepared
    if left.empty or right.empty:
        return None

    left_team = left_title.removeprefix("Restart takers - ").strip() or "Home"
    right_team = right_title.removeprefix("Restart takers - ").strip() or "Away"
    max_rows = max(len(left), len(right))
    left = left.reset_index(drop=True).reindex(range(max_rows))
    right = right.reset_index(drop=True).reindex(range(max_rows))

    def values(frame: pd.DataFrame, column: str) -> list[Any]:
        if column not in frame.columns:
            return [None] * max_rows
        return frame[column].tolist()

    return pd.DataFrame(
        {
            f"{left_team} player": values(left, "player_name"),
            f"{left_team} #": values(left, "restart_count"),
            f"{left_team} restart": values(left, "primary_restart"),
            f"{left_team} shots": values(left, "shots"),
            f"{right_team} player": values(right, "player_name"),
            f"{right_team} #": values(right, "restart_count"),
            f"{right_team} restart": values(right, "primary_restart"),
            f"{right_team} shots": values(right, "shots"),
        }
    )


def _table_story_for_spec(bundle, table_spec, styles, config) -> list[Any]:
    payloads = _table_payloads(
        bundle,
        table_spec.id,
        selection_limit=table_spec.selection.limit,
    )
    if not payloads:
        return [
            _placeholder_box(
                f"No neutral table payload is available for {table_spec.title}.",
                styles,
            ),
            Spacer(1, 3 * mm),
        ]

    prepared: list[tuple[str, pd.DataFrame]] = []
    for title, frame in payloads:
        frame = _prepare_pdf_table_frame(table_spec.id, frame)
        if not frame.empty:
            prepared.append((title, frame))

    if not prepared:
        return [
            _placeholder_box(
                f"No qualifying rows are available for {table_spec.title}.",
                styles,
            ),
            Spacer(1, 3 * mm),
        ]

    # REPORT-12: restart takers are a comparison, not two full-width tables.
    # Pair the two team top-six lists horizontally so the restart section stays
    # within its two-page editorial budget on real matches.
    if table_spec.id == "restart-takers":
        comparison = _paired_restart_takers_frame(prepared)
        if comparison is not None:
            return [
                Paragraph("Restart takers", styles["table_subheading"]),
                _long_table(comparison, styles, max_columns=8),
            ]

    story: list[Any] = []
    for index, (title, frame) in enumerate(prepared):
        story.append(Paragraph(escape(title), styles["table_subheading"]))
        story.append(
            _long_table(
                frame,
                styles,
                max_columns=config.max_table_columns,
            )
        )
        # Never leave a trailing Spacer after the final table. If a table ends
        # exactly at the bottom of a page, that spacer alone can spill onto a
        # new page and make the following section break create a blank page.
        if index < len(prepared) - 1:
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
        Paragraph("Goals", styles["table_subheading"]),
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
        "This PDF is the editorial reading layer of the Match Analysis Pack.",
        "Detailed analytical rows remain available in the CSV files and report-data.json.",
        "Figures come from the static Plotly catalog and tables contain only explicitly selected editorial fields.",
        "Empty or failed sections remain explicit so missing data is never hidden.",
    ]

    story: list[Any] = [
        Paragraph("Generation manifest", styles["subheading"]),
    ]
    for line in lines:
        story.append(Paragraph(escape(line), styles["body"]))

    story.extend(
        [
            Paragraph(
                "The full machine-readable manifest, section statuses and plot statuses are stored in report-manifest.json.",
                styles["small"],
            ),
            Paragraph("Generation metadata", styles["table_subheading"]),
            _long_table(
                pd.DataFrame(
                    [
                        {
                            "scope": getattr(getattr(bundle, "scope", None), "value", getattr(bundle, "scope", "")),
                            "figure_count": len(getattr(catalog, "figures", ()) or ()),
                            "manifest_version": getattr(catalog, "manifest_version", manifest.schema_version),
                        }
                    ]
                ),
                styles,
                max_columns=3,
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
        story.append(PageBreakIfNotEmpty())
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
            logger.exception("PDF section composition failed: %s", section_spec.id)
            for record in (config.render_audit or {}).values():
                if record["section_id"] == section_spec.id:
                    record.update(render_status="error", error_type=type(exc).__name__,
                                  error_message="Section composition failed.")
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
