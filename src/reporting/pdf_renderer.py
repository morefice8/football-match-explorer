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
    """Materialise named Index/MultiIndex levels without losing identity.

    REPORT-13 player-ranking inputs may carry ``playerName`` and/or
    ``team_name`` in an Index or MultiIndex.  Those levels must become columns
    *before* editorial column trimming and per-team selection.
    """

    frame = frame.copy()
    index = frame.index
    if isinstance(index, pd.RangeIndex) and index.name is None:
        return frame

    raw_names = list(index.names) if isinstance(index, pd.MultiIndex) else [index.name]
    names: list[str] = []
    used = {str(column) for column in frame.columns}
    for position, raw_name in enumerate(raw_names):
        base = str(raw_name).strip() if raw_name not in (None, "") else (
            "item" if len(raw_names) == 1 else f"index_level_{position}"
        )
        name = base
        suffix = 1
        while name in used or name in names:
            name = f"index_{base}" if suffix == 1 else f"index_{base}_{suffix}"
            suffix += 1
        names.append(name)
        used.add(name)

    return frame.reset_index(names=names)


def _coalesce_player_alias(
    frame: pd.DataFrame,
    canonical: str,
    aliases: Sequence[str],
) -> pd.DataFrame:
    """Create/fill one canonical identity column from any accepted aliases."""

    frame = frame.copy()
    candidates = [
        column
        for column in frame.columns
        if str(column).strip().casefold()
        in {str(alias).strip().casefold() for alias in aliases}
    ]
    if canonical in frame.columns:
        target = frame[canonical].copy()
    elif candidates:
        first = candidates.pop(0)
        target = frame[first].copy()
        if first != canonical:
            frame = frame.drop(columns=[first])
    else:
        return frame

    for column in candidates:
        missing = target.isna() | target.astype(str).str.strip().eq("")
        target = target.where(~missing, frame[column])
        if column != canonical:
            frame = frame.drop(columns=[column])
    frame[canonical] = target
    return frame


def _normalize_player_ranking_frame(
    value: Any,
    *,
    team_lookup: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Return a canonical player ranking frame for REPORT-13 PDF selection."""

    frame = _with_named_index(_as_frame(value))
    if frame.empty:
        return frame

    frame = _coalesce_player_alias(
        frame,
        "playerName",
        ("playerName", "player_name", "Player", "player", "name"),
    )
    frame = _coalesce_player_alias(
        frame,
        "team_name",
        ("team_name", "teamName", "Team", "team"),
    )

    if "playerName" not in frame.columns:
        return frame

    frame["playerName"] = frame["playerName"].map(
        lambda value: str(value).strip() if pd.notna(value) else value
    )

    lookup = {str(key): str(value) for key, value in (team_lookup or {}).items()}
    inferred = frame["playerName"].astype(str).map(lookup)
    if "team_name" not in frame.columns:
        frame["team_name"] = inferred
    else:
        team = frame["team_name"]
        missing = team.isna() | team.astype(str).str.strip().eq("")
        frame["team_name"] = team.where(~missing, inferred)
        frame["team_name"] = frame["team_name"].map(
            lambda value: str(value).strip() if pd.notna(value) else value
        )

    return frame


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
            "The report covers the full match. Selected sections may compare explicitly defined match periods such as 1H and 2H. Empty or failed sections remain visible as explicit report placeholders.",
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
    scope_label = (
        "1H vs 2H"
        if section_spec.id == "defensive-shape"
        else "FULL MATCH"
    )
    return [
        Paragraph(
            f"SECTION {section_spec.order:02d} / {scope_label}",
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
        if getattr(artifact, "id", None) == "defensive-shape-figure":
            variant_label = {
                "first_half": "1H",
                "second_half": "2H",
            }.get(variant, variant.replace("_", " ").title())
        else:
            variant_label = variant.replace("_", " ").title()
        if getattr(artifact, "id", None) == "formation-timeline-figure":
            selection = getattr(artifact, "selection", None) or {}
            time_label = _pdf_safe_helvetica_text(
                str(selection.get("time_label") or "").strip()
            )
            score = _pdf_safe_helvetica_text(
                str(selection.get("score") or "").strip().replace("–", "-")
            )
            if time_label:
                variant_label += f" @ {time_label}"
            if score:
                variant_label += f" / {score}"
        caption_parts.append(variant_label)
    caption = _pdf_safe_helvetica_text(" - ".join(caption_parts))
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
    *,
    variants_filter: Sequence[str] | None = None,
) -> list[Any]:
    artifacts = _artifacts_for(catalog, figure_spec.id)
    if variants_filter is not None:
        allowed = {str(item) for item in variants_filter}
        artifacts = [
            item for item in artifacts
            if str(getattr(item, "variant", "summary") or "summary") in allowed
        ]
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
    compact_formation = (
        figure_spec.id == "formation-timeline-figure"
        and len(variants) >= 3
    )
    large_defensive_density = (
        figure_spec.id == "defensive-shape-figure"
    )
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
            max_cell_width = (
                131 * mm
                if large_defensive_density
                else 92 * mm
            )
            max_cell_height = (
                86 * mm
                if large_defensive_density
                else (42 * mm if compact_formation else 54 * mm)
            )
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
                        (
                            "LEFTPADDING",
                            (0, 0),
                            (-1, -1),
                            1 if large_defensive_density else 3,
                        ),
                        (
                            "RIGHTPADDING",
                            (0, 0),
                            (-1, -1),
                            1 if large_defensive_density else 3,
                        ),
                        (
                            "TOPPADDING",
                            (0, 0),
                            (-1, -1),
                            1 if large_defensive_density else 3,
                        ),
                        (
                            "BOTTOMPADDING",
                            (0, 0),
                            (-1, -1),
                            2 if large_defensive_density else 5,
                        ),
                    ]
                )
            )
            story.extend([
                pair,
                Spacer(
                    1,
                    (
                        1 * mm
                        if large_defensive_density
                        else (2 * mm if compact_formation else 4 * mm)
                    ),
                ),
            ])
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
        "start",
        "end",
        "duration",
        "score",
        "home formation",
        "away formation",
        "change reason",
        "subs / dismissals",
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
        "Team",
        "Block height (m)",
        "Width (m)",
        "Compactness (m)",
        "Sample size",
        "Δ vs 1H",
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
        "Side", "Team", "Transitions", "Median duration (s)",
        "Final third %", "Penalty area %", "Shot %",
    ),
    "offensive-transitions-summary": (
        "Side", "Team", "Transitions", "Median duration (s)",
        "Final third %", "Penalty area %", "Shot %",
    ),
    "defensive-transitions-selection": (
        "Side", "Team", "Sequence", "Selection reason",
    ),
    "offensive-transitions-selection": (
        "Side", "Team", "Sequence", "Selection reason",
    ),
    # REPORT-16 keeps this legacy table id for manifest/backward compatibility,
    # but the PDF payload is now the canonical transition_profile_table rather
    # than event rows from combined.
    "defensive-transitions-sequences": (
        "Start zone", "Channel", "Home transitions", "Away transitions",
        "Home avg duration (s)", "Away avg duration (s)",
    ),
    "offensive-transitions-sequences": (
        "Start zone", "Channel", "Home transitions", "Away transitions",
        "Home avg duration (s)", "Away avg duration (s)",
    ),
    "defensive-transitions-taxonomy": (
        "Dimension", "Category", "Home", "Away",
    ),
    "offensive-transitions-taxonomy": (
        "Dimension", "Category", "Home", "Away",
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
    # Family-specific REPORT-13 columns are selected before this generic
    # scalar-safety policy is applied.  Keep every allowed family field here.
    "player-highlights-table": (
        "team_name",
        ("playerName", "player_name", "Player", "item"),
        "Offensive Pass Contributions",
        "Progressive Passes",
        "Passes into Box",
        "Key Passes",
        "Assists",
        "Shot Sequence Shots",
        ("Shot Sequence Shot Assists", "Shot Sequence Assists", "Shot Assists"),
        "Shot Sequence Pre-Assists",
        "Shot Sequence Involvements",
        ("unique", "Unique Defensive Contributions"),
        ("tackles_won", "Tackles Won"),
        ("interceptions", "Interceptions"),
        ("recoveries", "Recoveries"),
        ("clearances", "Clearances"),
        ("blocks", "Blocks"),
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
    column_widths: Sequence[float] | None = None,
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
    widths = (
        list(column_widths)
        if column_widths is not None and len(column_widths) == count
        else [available_width / count] * count
    )

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
    """Compact Home/Away transition KPIs prepared by the neutral bundle."""

    section = _section(bundle, section_id)
    data = getattr(section, "data", {}) or {}
    teams = tuple(getattr(bundle, "teams", ()) or ())
    rows = []

    for index, team in enumerate(teams[:2]):
        payload = data.get(team, {}) if isinstance(data, Mapping) else {}
        kpis = (payload or {}).get("kpis", {})
        if not isinstance(kpis, Mapping):
            kpis = {}

        def number(key: str, *, digits: int = 1):
            value = kpis.get(key)
            if value is None:
                return None
            try:
                return round(float(value), digits)
            except (TypeError, ValueError):
                return None

        rows.append(
            {
                "Side": "Home" if index == 0 else "Away",
                "Team": team,
                "Transitions": int(kpis.get("transition_count", 0) or 0),
                "Median duration (s)": number("median_duration_seconds"),
                "Final third %": number("final_third_pct"),
                "Penalty area %": number("penalty_area_pct"),
                "Shot %": number("shot_pct"),
            }
        )

    return pd.DataFrame(rows)

def _transition_zone(value: Any) -> str:
    text = " ".join(str(value or "").split()).strip()
    folded = text.casefold()
    if "defensive" in folded:
        return "Defensive Third"
    if "middle" in folded:
        return "Middle Third"
    if "attacking" in folded:
        return "Attacking Third"
    return "Unknown"


def _transition_channel(value: Any) -> str:
    text = " ".join(str(value or "").split()).strip()
    folded = text.casefold()
    if folded == "left":
        return "Left"
    if folded in {"center", "central", "centre"}:
        return "Center"
    if folded == "right":
        return "Right"
    return "Unknown"


_TRANSITION_ZONE_ORDER = {
    "Defensive Third": 0,
    "Middle Third": 1,
    "Attacking Third": 2,
    "Unknown": 3,
}
_TRANSITION_CHANNEL_ORDER = {
    "Left": 0,
    "Center": 1,
    "Right": 2,
    "Unknown": 3,
}


def _canonical_transition_profile_map(
    stats: Mapping[str, Any],
) -> dict[tuple[str, str], dict[str, float]]:
    """Normalize the canonical stats.transition_profile_table without combined."""

    frame = _as_frame(stats.get("transition_profile_table"))
    if frame.empty:
        return {}

    zone_col = _column_lookup(frame, "Loss Zone", "Recovery Zone", "Start zone")
    channel_col = _column_lookup(
        frame,
        "Counterattack Side",
        "Attack Side",
        "Channel",
    )
    count_col = _column_lookup(frame, "Num_Sequences", "Sequences", "transitions")
    duration_col = _column_lookup(frame, "Avg Duration (s)", "Avg duration (s)")
    if zone_col is None or channel_col is None:
        return {}

    buckets: dict[tuple[str, str], dict[str, float]] = {}
    for _, row in frame.iterrows():
        key = (
            _transition_zone(row.get(zone_col)),
            _transition_channel(row.get(channel_col)),
        )
        try:
            count = float(row.get(count_col, 0) or 0) if count_col else 0.0
        except (TypeError, ValueError):
            count = 0.0
        try:
            duration = (
                float(row.get(duration_col))
                if duration_col is not None and pd.notna(row.get(duration_col))
                else None
            )
        except (TypeError, ValueError):
            duration = None

        bucket = buckets.setdefault(
            key,
            {"count": 0.0, "duration_weighted": 0.0, "duration_n": 0.0},
        )
        bucket["count"] += count
        if duration is not None and count > 0:
            bucket["duration_weighted"] += duration * count
            bucket["duration_n"] += count

    return buckets


def _transition_profile_rows(
    bundle,
    section_id: str,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    """Home/Away comparison built from each team's canonical profile table."""

    section = _section(bundle, section_id)
    data = getattr(section, "data", {}) or {}
    teams = tuple(getattr(bundle, "teams", ()) or ())
    if len(teams) < 2:
        return pd.DataFrame()

    maps = []
    for team in teams[:2]:
        payload = data.get(team, {}) if isinstance(data, Mapping) else {}
        stats = (payload or {}).get("stats", {})
        maps.append(
            _canonical_transition_profile_map(stats)
            if isinstance(stats, Mapping)
            else {}
        )

    keys = sorted(
        set(maps[0]) | set(maps[1]),
        key=lambda item: (
            _TRANSITION_ZONE_ORDER.get(item[0], 99),
            _TRANSITION_CHANNEL_ORDER.get(item[1], 99),
            item[0],
            item[1],
        ),
    )
    if limit is not None:
        keys = keys[: int(limit)]

    def avg(bucket: Mapping[str, float]) -> float | None:
        denominator = float(bucket.get("duration_n", 0.0) or 0.0)
        if denominator <= 0:
            return None
        return round(float(bucket.get("duration_weighted", 0.0)) / denominator, 1)

    rows = []
    for key in keys:
        home = maps[0].get(key, {})
        away = maps[1].get(key, {})
        rows.append(
            {
                "Start zone": key[0],
                "Channel": key[1],
                "Home transitions": int(home.get("count", 0) or 0),
                "Away transitions": int(away.get("count", 0) or 0),
                "Home avg duration (s)": avg(home),
                "Away avg duration (s)": avg(away),
            }
        )
    return pd.DataFrame(rows)


def _taxonomy_priority(dimension: str, category: str) -> tuple[int, str]:
    folded = str(category or "").casefold()
    if dimension == "Channel":
        return (_TRANSITION_CHANNEL_ORDER.get(_transition_channel(category), 99), folded)

    tokens = (
        ("goal", 0),
        ("shot", 1),
        ("consolid", 2),
        ("regain", 3),
        ("recover", 3),
        ("turnover", 4),
        ("lost", 4),
        ("unsuccess", 5),
        ("out", 6),
        ("foul", 7),
        ("period", 8),
        ("unknown", 99),
    )
    for token, order in tokens:
        if token in folded:
            return (order, folded)
    return (50, folded)


def _transition_taxonomy_rows(bundle, section_id: str) -> pd.DataFrame:
    """Align outcome, terminal-outcome and channel taxonomies Home vs Away."""

    section = _section(bundle, section_id)
    data = getattr(section, "data", {}) or {}
    teams = tuple(getattr(bundle, "teams", ()) or ())
    if len(teams) < 2:
        return pd.DataFrame()

    stats_by_team: list[Mapping[str, Any]] = []
    for team in teams[:2]:
        payload = data.get(team, {}) if isinstance(data, Mapping) else {}
        stats = (payload or {}).get("stats", {})
        stats_by_team.append(stats if isinstance(stats, Mapping) else {})

    rows: list[dict[str, Any]] = []
    for dimension, key in (
        ("Outcome", "outcomes"),
        ("Terminal outcome", "terminal_outcomes"),
        ("Channel", "flanks"),
    ):
        raw_maps = []
        for stats in stats_by_team:
            raw = stats.get(key, {}) or {}
            normalized: dict[str, float] = {}
            if isinstance(raw, Mapping):
                for category, value in raw.items():
                    label = (
                        _transition_channel(category)
                        if dimension == "Channel"
                        else " ".join(str(category).split()) or "Unknown"
                    )
                    try:
                        numeric = float(value or 0)
                    except (TypeError, ValueError):
                        numeric = 0.0
                    normalized[label] = normalized.get(label, 0.0) + numeric
            raw_maps.append(normalized)

        categories = sorted(
            set(raw_maps[0]) | set(raw_maps[1]),
            key=lambda category: _taxonomy_priority(dimension, category),
        )
        totals = [sum(mapping.values()) for mapping in raw_maps]

        for category in categories:
            values = []
            for index, mapping in enumerate(raw_maps):
                count = int(mapping.get(category, 0) or 0)
                total = totals[index]
                share = count / total * 100.0 if total else 0.0
                values.append(f"{count} ({share:.0f}%)")
            rows.append(
                {
                    "Dimension": dimension,
                    "Category": category,
                    "Home": values[0],
                    "Away": values[1],
                }
            )

    return pd.DataFrame(rows)


def _transition_taxonomy_compact_table(frame: pd.DataFrame, styles) -> Table:
    """Render REPORT-16 taxonomy densely enough to stay on profile page.

    The taxonomy is intentionally complete and aligned Home/Away; the compact
    padding only removes vertical whitespace. It does not drop categories,
    reorder rows, or abbreviate scalar values.
    """

    frame = _trim_columns(frame.copy(), 4)
    if frame.empty:
        return _placeholder_box("No taxonomy rows available.", styles)

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

    table = Table(
        data,
        colWidths=[44 * mm, 96 * mm, 64 * mm, 64 * mm],
        repeatRows=1,
        splitByRow=1,
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
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                # REPORT-16 real-match profile tables can contain 15 taxonomy
                # rows. Two-point vertical padding keeps the full taxonomy on
                # page 2 while preserving the normal 7.4pt table type size.
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def _transition_selection_reason(artifact: Any) -> str:
    """Turn selector metadata into a short editorial explanation."""

    selection = getattr(artifact, "selection", None) or {}
    criteria_rows = selection.get("criteria", []) if isinstance(selection, Mapping) else []
    criteria = {
        str(item.get("criterion")): item.get("value")
        for item in criteria_rows
        if isinstance(item, Mapping) and item.get("criterion")
    }
    milestone = criteria.get("milestone")
    progression = criteria.get("territorial_progression")
    action_count = criteria.get("action_count")
    duration = criteria.get("duration")

    pieces = []
    if milestone not in (None, ""):
        pieces.append(f"highest-priority outcome: {milestone}")
    tie_breaks = []
    if progression not in (None, ""):
        tie_breaks.append(f"progression {progression}")
    if action_count not in (None, ""):
        tie_breaks.append(f"{action_count} actions")
    if duration not in (None, ""):
        tie_breaks.append(f"{duration}s duration")
    if tie_breaks:
        pieces.append("tie-breaks: " + ", ".join(tie_breaks))

    if pieces:
        return "; ".join(pieces) + "."

    reason = getattr(artifact, "selection_reason", None)
    return str(reason or "Deterministic representative-sequence selector.")


def _transition_selection_rows(
    catalog,
    figure_id: str,
    teams: Sequence[str],
) -> pd.DataFrame:
    artifacts = _artifacts_for(catalog, figure_id)
    by_team = {
        str(getattr(artifact, "team_name", "")): artifact
        for artifact in artifacts
        if getattr(artifact, "team_name", None)
    }
    rows = []
    for index, team in enumerate(tuple(teams)[:2]):
        artifact = by_team.get(str(team))
        if artifact is None:
            continue
        selection = getattr(artifact, "selection", None) or {}
        selected_id = (
            selection.get("selected_id")
            if isinstance(selection, Mapping)
            else None
        )
        rows.append(
            {
                "Side": "Home" if index == 0 else "Away",
                "Team": team,
                "Sequence": selected_id if selected_id is not None else "-",
                "Selection reason": _transition_selection_reason(artifact),
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


def _formation_clock_label(seconds: Any) -> str:
    try:
        total = max(int(float(seconds)), 0)
    except (TypeError, ValueError):
        return "-"
    minute, second = divmod(total, 60)
    return f"{minute}'" if second == 0 else f"{minute}' {second:02d}\""


def _formation_duration_label(seconds: Any) -> str:
    try:
        total = max(int(float(seconds)), 0)
    except (TypeError, ValueError):
        return "-"
    minute, second = divmod(total, 60)
    if minute and second:
        return f"{minute}m {second:02d}s"
    if minute:
        return f"{minute}m"
    return f"{second}s"


def _formation_spells_rows(
    moments: Any,
    *,
    match_end_seconds: Any = None,
) -> pd.DataFrame:
    """Build REPORT-14's compact scalar Formation Timeline table.

    The analytical model keeps rich nested home/away state snapshots and event
    payloads.  The PDF deliberately exposes only editorial spell metadata;
    complete nested structures remain in report-data.json.
    """

    labels = {
        "starting_xi": "Starting XI",
        "goal": "Goal",
        "substitution": "Substitution",
        "formation_change": "Tactical change",
        "dismissal": "Dismissal",
        "red_card": "Dismissal",
    }

    ordered: list[tuple[int, int, Mapping[str, Any]]] = []
    for index, moment in enumerate(moments or []):
        if not isinstance(moment, Mapping):
            continue
        raw_seconds = moment.get("time_seconds")
        if raw_seconds is None:
            raw_seconds = (moment.get("minute") or 0) * 60
        try:
            seconds = max(int(float(raw_seconds)), 0)
        except (TypeError, ValueError):
            seconds = 0
        ordered.append((seconds, index, moment))
    ordered.sort(key=lambda item: (item[0], item[1]))

    if not ordered:
        return pd.DataFrame(
            columns=[
                "start",
                "end",
                "duration",
                "score",
                "home formation",
                "away formation",
                "change reason",
                "subs / dismissals",
            ]
        )

    try:
        resolved_match_end = max(
            int(float(match_end_seconds)),
            ordered[-1][0],
        )
    except (TypeError, ValueError):
        resolved_match_end = ordered[-1][0]

    rows: list[dict[str, Any]] = []
    for position, (start_seconds, _, moment) in enumerate(ordered):
        end_seconds = (
            ordered[position + 1][0]
            if position + 1 < len(ordered)
            else resolved_match_end
        )
        end_seconds = max(end_seconds, start_seconds)

        reasons: list[str] = []
        personnel: list[str] = []
        events = moment.get("events") or []
        if isinstance(events, Sequence) and not isinstance(events, (str, bytes)):
            for event in events:
                if not isinstance(event, Mapping):
                    continue
                kind = str(event.get("kind") or "").strip()
                label = labels.get(
                    kind,
                    kind.replace("_", " ").title() if kind else "",
                )
                if label and label not in reasons:
                    reasons.append(label)

                if kind in {"substitution", "dismissal", "red_card"}:
                    description = str(event.get("description") or "").strip()
                    if description and description not in personnel:
                        personnel.append(description)

        rows.append(
            {
                "start": _formation_clock_label(start_seconds),
                "end": _formation_clock_label(end_seconds),
                "duration": _formation_duration_label(end_seconds - start_seconds),
                "score": str(moment.get("score") or "-").replace("–", "-"),
                "home formation": moment.get("home_formation_name") or "-",
                "away formation": moment.get("away_formation_name") or "-",
                "change reason": " + ".join(reasons) if reasons else "-",
                "subs / dismissals": "; ".join(personnel) if personnel else "-",
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
                "team_name",
                "playerName",
                "Offensive Pass Contributions",
                "Progressive Passes",
                "Passes into Box",
                "Key Passes",
                "Assists",
            ),
        ),
        (
            "Shooting",
            "shot_sequence_ranking",
            (
                "team_name",
                "playerName",
                "Shot Sequence Shots",
                ("Shot Sequence Shot Assists", "Shot Sequence Assists", "Shot Assists"),
                "Shot Sequence Pre-Assists",
                "Shot Sequence Involvements",
            ),
        ),
        (
            "Defending",
            "defensive_ranking",
            (
                "team_name",
                "playerName",
                ("unique", "Unique Defensive Contributions"),
                ("tackles_won", "Tackles Won"),
                ("interceptions", "Interceptions"),
                ("recoveries", "Recoveries"),
                ("clearances", "Clearances"),
                ("blocks", "Blocks"),
            ),
        ),
    )

    payloads: list[tuple[str, pd.DataFrame]] = []
    for label, key, columns in families:
        # REPORT-13: identity may live in Index/MultiIndex.  Canonicalise it
        # before any selector or PDF column trimming is allowed to run.
        frame = _normalize_player_ranking_frame(
            data.get(key),
            team_lookup=team_lookup,
        )
        if frame.empty or "playerName" not in frame.columns:
            continue

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


def _defensive_shape_delta_text(first: Mapping[str, Any], second: Mapping[str, Any]) -> str:
    """Compact 2H-v-1H change for defensive-action density metrics."""
    parts = []
    for key, label in (
        ("block_height_m", "BH"),
        ("width_m", "W"),
        ("compactness_m", "C"),
    ):
        before = first.get(key) if isinstance(first, Mapping) else None
        after = second.get(key) if isinstance(second, Mapping) else None
        try:
            if before is None or after is None or pd.isna(before) or pd.isna(after):
                continue
            delta = float(after) - float(before)
        except Exception:
            continue
        parts.append(f"{label} {delta:+.1f}m")
    return " · ".join(parts) if parts else "-"


def _defensive_shape_period_frame(bundle, period_key: str) -> pd.DataFrame:
    section = _section(bundle, "defensive-shape")
    data = getattr(section, "data", {}) or {}
    teams = tuple(getattr(bundle, "teams", ()) or ())
    rows = []

    for team in teams:
        payload = data.get(team, {}) if isinstance(data, Mapping) else {}
        profile = (
            payload.get(period_key, {})
            if isinstance(payload, Mapping)
            else {}
        )
        first = (
            payload.get("first_half", {})
            if isinstance(payload, Mapping)
            else {}
        )
        action_count = profile.get("action_count", 0) if isinstance(profile, Mapping) else 0
        sample_size = f"{int(action_count or 0)} actions"

        row = {
            "Team": team,
            "Block height (m)": profile.get("block_height_m") if isinstance(profile, Mapping) else None,
            "Width (m)": profile.get("width_m") if isinstance(profile, Mapping) else None,
            "Compactness (m)": profile.get("compactness_m") if isinstance(profile, Mapping) else None,
            "Sample size": sample_size,
        }
        if period_key == "second_half":
            row["Δ vs 1H"] = _defensive_shape_delta_text(first, profile)
        rows.append(row)

    columns = [
        "Team",
        "Block height (m)",
        "Width (m)",
        "Compactness (m)",
        "Sample size",
    ]
    if period_key == "second_half":
        columns.append("Δ vs 1H")
    return pd.DataFrame(rows, columns=columns)


def _defensive_shape_table_payloads(bundle) -> list[tuple[str, pd.DataFrame]]:
    return [
        ("1H metrics", _defensive_shape_period_frame(bundle, "first_half")),
        ("2H metrics and change vs 1H", _defensive_shape_period_frame(bundle, "second_half")),
    ]


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
        match_end_seconds = (
            data.get("match_end_seconds")
            if isinstance(data, Mapping)
            else None
        )
        return [(
            "Formation spells",
            _formation_spells_rows(
                moments,
                match_end_seconds=match_end_seconds,
            ),
        )]

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
        return _defensive_shape_table_payloads(bundle)

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
                "Transition overview",
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
        # REPORT-16: this legacy id now represents the canonical profile table.
        # Never use combined event rows as the editorial PDF table.
        return [
            (
                "Transition profile · Home vs Away",
                _transition_profile_rows(
                    bundle,
                    section_id,
                    limit=selection_limit or 10,
                ),
            )
        ]

    if table_id in {
        "defensive-transitions-taxonomy",
        "offensive-transitions-taxonomy",
    }:
        section_id = (
            "defensive-transitions"
            if table_id.startswith("defensive")
            else "offensive-transitions"
        )
        return [
            (
                "Outcome, terminal outcome and channel",
                _transition_taxonomy_rows(bundle, section_id),
            )
        ]

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
                "Definition": "Default report scope. Individual sections may use explicitly declared sub-periods when the analysis requires temporal comparison; current UI filters do not alter the report.",
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
                "Term": "Defensive density",
                "Definition": "Spatial distribution of all qualifying defensive actions in each half; it shows where defending occurs, not simultaneous player positions.",
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

    # REPORT-14: the eight-column timeline is intentionally asymmetric.
    # Clock/score columns stay narrow while change/personnel descriptions get
    # enough width to remain readable without inflating the section beyond two
    # pages on A4 landscape.
    if table_spec.id == "formation-spells" and len(prepared) == 1:
        title, frame = prepared[0]
        return [
            Paragraph(escape(title), styles["table_subheading"]),
            _long_table(
                frame,
                styles,
                max_columns=8,
                column_widths=[
                    15 * mm,
                    15 * mm,
                    17 * mm,
                    18 * mm,
                    30 * mm,
                    30 * mm,
                    42 * mm,
                    101 * mm,
                ],
            ),
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


def _defensive_shape_section_story(
    bundle,
    catalog,
    section_spec,
    styles,
    config,
) -> list[Any]:
    """Two-page 1H/2H defensive-action density comparison."""

    story: list[Any] = []
    figure_spec = section_spec.figures[0] if section_spec.figures else None

    for index, (period_key, title) in enumerate((
        ("first_half", "First half (1H)"),
        ("second_half", "Second half (2H)"),
    )):
        if index:
            story.append(PageBreak())
        story.append(Paragraph(title, styles["subheading"]))
        story.append(
            Paragraph(
                "Same density rendering as the interactive app. "
                "Colour intensity is scaled within each panel; light points "
                "show the underlying defensive actions.",
                styles["small"],
            )
        )

        if figure_spec is not None:
            story.extend(
                _figure_story_for_spec(
                    catalog,
                    figure_spec,
                    styles,
                    config,
                    variants_filter=(period_key,),
                )
            )

        frame = _defensive_shape_period_frame(bundle, period_key)
        table_title = (
            "1H defensive-density metrics"
            if period_key == "first_half"
            else "2H defensive-density metrics · change vs 1H"
        )
        story.append(Paragraph(table_title, styles["table_subheading"]))
        story.append(
            _long_table(
                frame,
                styles,
                max_columns=6,
            )
        )

    return story


def _transition_section_story(
    bundle,
    catalog,
    section_spec,
    styles,
    config,
) -> list[Any]:
    """Two-page editorial transition profile for REPORT-16."""

    section_id = section_spec.id
    defensive = section_id == "defensive-transitions"
    prefix = "defensive" if defensive else "offensive"
    figure_id = f"{prefix}-transitions-top-sequence"
    teams = tuple(getattr(bundle, "teams", ()) or ())

    story: list[Any] = []
    story.append(Paragraph("Representative sequence", styles["subheading"]))
    figure_spec = next(
        (item for item in section_spec.figures if item.id == figure_id),
        None,
    )
    if figure_spec is not None:
        story.extend(
            _figure_story_for_spec(
                catalog,
                figure_spec,
                styles,
                config,
            )
        )

    selection = _transition_selection_rows(catalog, figure_id, teams)
    story.append(Paragraph("Selection reason", styles["table_subheading"]))
    story.append(
        _long_table(
            selection,
            styles,
            max_columns=4,
            column_widths=[18 * mm, 34 * mm, 24 * mm, 192 * mm],
        )
    )
    story.append(Spacer(1, 2 * mm))

    story.append(Paragraph("Home–Away overview", styles["table_subheading"]))
    overview = _transition_summary_rows(bundle, section_id)
    story.append(
        _long_table(
            overview,
            styles,
            max_columns=7,
        )
    )

    # Page 2 is intentionally profile-first.  The full event rows stay in the
    # Match Analysis Pack CSVs; the PDF shows only aligned, scalar summaries.
    story.append(PageBreak())
    matchup = (
        f"Home: {teams[0]} · Away: {teams[1]}"
        if len(teams) >= 2
        else "Home–Away comparison"
    )
    story.append(Paragraph(escape(matchup), styles["small"]))
    story.append(Spacer(1, 1.5 * mm))

    story.append(Paragraph("Transition profile", styles["subheading"]))
    profile_spec = next(
        (
            item
            for item in section_spec.tables
            if item.id == f"{prefix}-transitions-sequences"
        ),
        None,
    )
    profile_limit = (
        getattr(getattr(profile_spec, "selection", None), "limit", None)
        if profile_spec is not None
        else None
    )
    profile = _transition_profile_rows(
        bundle,
        section_id,
        limit=profile_limit or 10,
    )
    story.append(
        _long_table(
            profile,
            styles,
            max_columns=6,
            column_widths=[46 * mm, 30 * mm, 42 * mm, 42 * mm, 54 * mm, 54 * mm],
        )
    )
    story.append(Spacer(1, 3 * mm))

    story.append(
        Paragraph(
            "Outcome, terminal outcome and channel",
            styles["table_subheading"],
        )
    )
    taxonomy = _transition_taxonomy_rows(bundle, section_id)
    story.append(
        _transition_taxonomy_compact_table(
            taxonomy,
            styles,
        )
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

    if section_spec.id == "defensive-shape":
        story.extend(
            _defensive_shape_section_story(
                bundle,
                catalog,
                section_spec,
                styles,
                config,
            )
        )
        return story

    if section_spec.id in {
        "defensive-transitions",
        "offensive-transitions",
    }:
        story.extend(
            _transition_section_story(
                bundle,
                catalog,
                section_spec,
                styles,
                config,
            )
        )
        return story

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
