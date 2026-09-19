"""Poppler-based visual regression checks for generated Match Analysis PDFs.

REPORT-21 deliberately validates the final PDF independently of ReportLab and
Plotly. All pages are rendered through Poppler in a temporary workspace; only
five diagnostic snapshots are persisted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from typing import Any, Mapping, Sequence

from PIL import Image


DEFAULT_MIN_PAGES = 24
DEFAULT_MAX_PAGES = 32
DEFAULT_DPI = 96

FORBIDDEN_TECHNICAL_TEXT = (
    "Figure unavailable",
    "Section composition failed",
    "No static renderer registered for this manifest figure.",
    "Corresponding report-data section is missing.",
)

_TABLE_MARKERS = (
    "formation spells",
    "home-away summary",
    "top cross routes",
    "build-up summary",
    "transition profile",
    "outcome, terminal outcome and channel",
    "ppda summary",
    "restart summary",
    "restart takers",
    "player highlights - passing",
    "player highlights - shooting",
    "player highlights - defending",
    "methodology notes",
    "metric definitions",
    "data quality notes",
)

_SNAPSHOT_SELECTORS = (
    ("cover", ("__cover__",)),
    ("overview", ("SECTION 01", "Overview and data coverage")),
    ("pitch", ("SECTION 03", "Mean Positions")),
    ("transitions", ("SECTION 12", "Defensive Transitions")),
    ("player-highlights", ("SECTION 15", "Player highlights")),
)


@dataclass(frozen=True)
class PdfVisualAuditConfig:
    dpi: int = DEFAULT_DPI
    min_pages: int = DEFAULT_MIN_PAGES
    max_pages: int = DEFAULT_MAX_PAGES
    blank_ink_ratio: float = 0.001
    nearly_empty_table_ink_ratio: float = 0.018
    header_text: str = "IL LAB DELL'8"
    require_headers: bool = True
    save_snapshots: bool = True


@dataclass(frozen=True)
class PdfVisualIssue:
    severity: str
    code: str
    message: str
    page_number: int | None = None


@dataclass(frozen=True)
class PdfPageVisual:
    page_number: int
    text: str
    ink_ratio: float
    body_ink_ratio: float
    width_px: int
    height_px: int


@dataclass
class PdfVisualAuditResult:
    page_count: int = 0
    pages: list[PdfPageVisual] = field(default_factory=list)
    issues: list[PdfVisualIssue] = field(default_factory=list)
    snapshots: dict[str, str] = field(default_factory=dict)
    poppler_renderer: str | None = None
    poppler_text: str | None = None

    @property
    def failed(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)


def _candidate_executable_paths(name: str) -> list[Path]:
    suffix = ".exe" if os.name == "nt" else ""
    filename = name + suffix

    roots = []
    for raw in (
        os.environ.get("CONDA_PREFIX"),
        sys.prefix,
    ):
        if raw:
            roots.append(Path(raw))

    candidates = []
    for root in roots:
        candidates.extend(
            (
                root / "Library" / "bin" / filename,
                root / "bin" / filename,
            )
        )
    return candidates


def find_poppler_executable(name: str) -> str | None:
    resolved = shutil.which(name)
    if resolved:
        return resolved

    for candidate in _candidate_executable_paths(name):
        if candidate.is_file():
            return str(candidate)

    return None


def poppler_available() -> bool:
    return bool(
        find_poppler_executable("pdftoppm")
        and find_poppler_executable("pdftotext")
    )


def _run(
    args: Sequence[str],
    *,
    text: bool = False,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(args),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        encoding="utf-8" if text else None,
        errors="replace" if text else None,
    )


def _normalized_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _contains_text(haystack: str, needle: str) -> bool:
    return _normalized_text(needle) in _normalized_text(haystack)


def _ink_ratio(image: Image.Image) -> tuple[float, float]:
    gray = image.convert("L")
    histogram = gray.histogram()
    ink = sum(histogram[:245])
    total = max(gray.width * gray.height, 1)
    full = ink / total

    left = int(gray.width * 0.03)
    right = int(gray.width * 0.97)
    top = int(gray.height * 0.10)
    bottom = int(gray.height * 0.92)
    body = gray.crop((left, top, right, bottom))
    body_hist = body.histogram()
    body_ink = sum(body_hist[:245])
    body_total = max(body.width * body.height, 1)
    return full, body_ink / body_total


def _render_pdf(
    pdf_path: Path,
    render_dir: Path,
    *,
    dpi: int,
    executable: str,
) -> list[Path]:
    render_dir.mkdir(parents=True, exist_ok=True)
    prefix = render_dir / "page"

    _run(
        (
            executable,
            "-png",
            "-r",
            str(int(dpi)),
            str(pdf_path),
            str(prefix),
        )
    )

    pages = sorted(render_dir.glob("page-*.png"))
    return pages


def _extract_page_text(
    pdf_path: Path,
    *,
    executable: str,
) -> list[str]:
    completed = _run(
        (
            executable,
            "-layout",
            "-enc",
            "UTF-8",
            str(pdf_path),
            "-",
        ),
        text=True,
    )

    pages = completed.stdout.split("\f")
    while pages and not pages[-1].strip():
        pages.pop()
    return pages


def _top_players_from_summary(
    analysis_summary: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    if not isinstance(analysis_summary, Mapping):
        return ()

    top_players = analysis_summary.get("top_players", {})
    if not isinstance(top_players, Mapping):
        return ()

    selected: list[str] = []
    for family in ("passing", "shooting", "defending"):
        teams = top_players.get(family, [])
        if not isinstance(teams, Sequence) or isinstance(
            teams,
            (str, bytes, bytearray),
        ):
            continue

        for team_payload in teams:
            if not isinstance(team_payload, Mapping):
                continue
            ranking = team_payload.get("ranking", [])
            if not ranking or not isinstance(ranking, Sequence):
                continue
            first = ranking[0]
            if not isinstance(first, Mapping):
                continue
            player = str(first.get("player") or "").strip()
            if player and player not in selected:
                selected.append(player)

    return tuple(selected)


def expectations_from_analysis_summary(
    analysis_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    match = (
        analysis_summary.get("match", {})
        if isinstance(analysis_summary, Mapping)
        else {}
    )
    match = match if isinstance(match, Mapping) else {}

    return {
        "teams": tuple(
            value
            for value in (
                match.get("home_team"),
                match.get("away_team"),
            )
            if value not in (None, "")
        ),
        "top_players": _top_players_from_summary(
            analysis_summary
        ),
    }


def _table_page(text: str) -> bool:
    normalized = _normalized_text(text)
    return any(marker in normalized for marker in _TABLE_MARKERS)


def _select_snapshot_pages(
    pages: Sequence[PdfPageVisual],
) -> dict[str, int]:
    selected: dict[str, int] = {}

    for label, terms in _SNAPSHOT_SELECTORS:
        if label == "cover":
            if pages:
                selected[label] = 1
            continue

        for page in pages:
            if page.page_number <= 2:
                continue
            if all(_contains_text(page.text, term) for term in terms):
                selected[label] = page.page_number
                break

    return selected


def _persist_snapshots(
    rendered_pages: Sequence[Path],
    pages: Sequence[PdfPageVisual],
    output_dir: Path,
) -> tuple[dict[str, str], list[PdfVisualIssue]]:
    issues: list[PdfVisualIssue] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for existing in output_dir.glob("*.png"):
        existing.unlink()

    selected = _select_snapshot_pages(pages)
    snapshots: dict[str, str] = {}

    for label, _ in _SNAPSHOT_SELECTORS:
        page_number = selected.get(label)
        if page_number is None:
            issues.append(
                PdfVisualIssue(
                    "warning",
                    "snapshot-target-missing",
                    (
                        f"Could not locate the {label!r} snapshot page; "
                        "the corresponding optional section may be empty."
                    ),
                )
            )
            continue

        source = rendered_pages[page_number - 1]
        destination = output_dir / (
            f"{label}-page-{page_number:02d}.png"
        )
        shutil.copy2(source, destination)
        snapshots[label] = str(destination)

    return snapshots, issues


def _content_issues(
    pages: Sequence[PdfPageVisual],
    *,
    config: PdfVisualAuditConfig,
    expectations: Mapping[str, Any] | None,
) -> list[PdfVisualIssue]:
    issues: list[PdfVisualIssue] = []

    if not (
        config.min_pages
        <= len(pages)
        <= config.max_pages
    ):
        issues.append(
            PdfVisualIssue(
                "error",
                "pdf-page-budget",
                (
                    f"PDF has {len(pages)} pages; expected "
                    f"{config.min_pages}-{config.max_pages}."
                ),
            )
        )

    for page in pages:
        if page.ink_ratio < config.blank_ink_ratio:
            issues.append(
                PdfVisualIssue(
                    "error",
                    "blank-pdf-page",
                    (
                        f"Page {page.page_number} is visually blank "
                        f"(ink ratio {page.ink_ratio:.4f})."
                    ),
                    page.page_number,
                )
            )

        if (
            _table_page(page.text)
            and page.body_ink_ratio
            < config.nearly_empty_table_ink_ratio
        ):
            issues.append(
                PdfVisualIssue(
                    "error",
                    "nearly-empty-table-page",
                    (
                        f"Page {page.page_number} looks like a table page "
                        "but contains almost no rendered body content "
                        f"(body ink ratio {page.body_ink_ratio:.4f})."
                    ),
                    page.page_number,
                )
            )

        for phrase in FORBIDDEN_TECHNICAL_TEXT:
            if _contains_text(page.text, phrase):
                issues.append(
                    PdfVisualIssue(
                        "error",
                        "technical-placeholder-visible",
                        (
                            f"Page {page.page_number} contains technical "
                            f"fallback text: {phrase!r}."
                        ),
                        page.page_number,
                    )
                )

        if config.require_headers and page.page_number >= 2:
            if not _contains_text(page.text, config.header_text):
                issues.append(
                    PdfVisualIssue(
                        "error",
                        "pdf-header-missing",
                        (
                            f"Page {page.page_number} is missing the "
                            f"extractable report header."
                        ),
                        page.page_number,
                    )
                )

            if not re.search(
                rf"\bPage\s+{page.page_number}\b",
                page.text,
                flags=re.IGNORECASE,
            ):
                issues.append(
                    PdfVisualIssue(
                        "error",
                        "pdf-page-number-missing",
                        (
                            f"Page {page.page_number} is missing its "
                            "extractable page number."
                        ),
                        page.page_number,
                    )
                )

    combined_text = "\n".join(page.text for page in pages)

    expectations = expectations or {}
    for team in expectations.get("teams", ()) or ():
        if team and not _contains_text(combined_text, str(team)):
            issues.append(
                PdfVisualIssue(
                    "error",
                    "team-name-not-extractable",
                    f"Team name {team!r} is not extractable from the PDF.",
                )
            )

    top_players = tuple(
        expectations.get("top_players", ()) or ()
    )

    for player in top_players:
        if player and not _contains_text(
            combined_text,
            str(player),
        ):
            issues.append(
                PdfVisualIssue(
                    "error",
                    "top-player-not-extractable",
                    (
                        f"Top player {player!r} is not extractable "
                        "from the PDF."
                    ),
                )
            )

    return issues


def run_pdf_visual_regression(
    pdf_bytes: bytes,
    *,
    analysis_summary: Mapping[str, Any] | None = None,
    snapshot_dir: str | Path | None = None,
    config: PdfVisualAuditConfig | None = None,
) -> PdfVisualAuditResult:
    """Render every PDF page with Poppler and validate the final artifact."""

    config = config or PdfVisualAuditConfig()
    result = PdfVisualAuditResult()

    renderer = find_poppler_executable("pdftoppm")
    text_extractor = find_poppler_executable("pdftotext")
    result.poppler_renderer = renderer
    result.poppler_text = text_extractor

    if not renderer or not text_extractor:
        missing = []
        if not renderer:
            missing.append("pdftoppm")
        if not text_extractor:
            missing.append("pdftotext")
        result.issues.append(
            PdfVisualIssue(
                "error",
                "poppler-unavailable",
                (
                    "REPORT-21 requires Poppler executables: "
                    + ", ".join(missing)
                ),
            )
        )
        return result

    if not pdf_bytes.startswith(b"%PDF"):
        result.issues.append(
            PdfVisualIssue(
                "error",
                "visual-pdf-invalid",
                "Visual regression input does not start with %PDF.",
            )
        )
        return result

    with tempfile.TemporaryDirectory(
        prefix="report21-pdf-visual-"
    ) as temporary:
        workspace = Path(temporary)
        pdf_path = workspace / "report.pdf"
        render_dir = workspace / "rendered"
        pdf_path.write_bytes(bytes(pdf_bytes))

        try:
            rendered = _render_pdf(
                pdf_path,
                render_dir,
                dpi=config.dpi,
                executable=renderer,
            )
            page_text = _extract_page_text(
                pdf_path,
                executable=text_extractor,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (
                exc.stderr.decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes)
                else str(exc.stderr or "")
            )
            result.issues.append(
                PdfVisualIssue(
                    "error",
                    "poppler-render-failed",
                    stderr.strip() or str(exc),
                )
            )
            return result

        result.page_count = len(rendered)

        if len(rendered) != len(page_text):
            result.issues.append(
                PdfVisualIssue(
                    "error",
                    "pdf-render-text-page-mismatch",
                    (
                        f"Poppler rendered {len(rendered)} pages but "
                        f"extracted text for {len(page_text)} pages."
                    ),
                )
            )

        page_total = min(len(rendered), len(page_text))
        for index in range(page_total):
            path = rendered[index]
            with Image.open(path) as image:
                full_ink, body_ink = _ink_ratio(image)
                width, height = image.size

            result.pages.append(
                PdfPageVisual(
                    page_number=index + 1,
                    text=page_text[index],
                    ink_ratio=full_ink,
                    body_ink_ratio=body_ink,
                    width_px=width,
                    height_px=height,
                )
            )

        expectations = expectations_from_analysis_summary(
            analysis_summary
        )
        result.issues.extend(
            _content_issues(
                result.pages,
                config=config,
                expectations=expectations,
            )
        )

        if config.save_snapshots and snapshot_dir is not None:
            snapshots, snapshot_issues = _persist_snapshots(
                rendered,
                result.pages,
                Path(snapshot_dir),
            )
            result.snapshots = snapshots
            result.issues.extend(snapshot_issues)

    return result
