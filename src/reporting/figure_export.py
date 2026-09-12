"""Kaleido-backed static export for Match Report Plotly figures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import shutil
import tempfile

import plotly.graph_objects as go
import plotly.io as pio

from src.reporting.figure_catalog import MatchReportFigureCatalog
from src.reporting.pdf_renderer import MatchReportPdfConfig


logger = logging.getLogger(__name__)


class FigureExportStatus(str, Enum):
    EXPORTED = "exported"
    ERROR = "error"


class StaticExportPreflightStatus(str, Enum):
    """UI-facing severity of the static export preflight."""

    SUCCESS = "success"
    WARNING = "warning"
    FAILURE = "failure"


class StaticExportPreflightCode(str, Enum):
    READY = "ready"
    KALEIDO_MISSING = "kaleido-missing"
    CHROME_MISSING = "chrome-missing"
    RENDERER_ERROR = "renderer-error"


@dataclass(frozen=True)
class StaticExportPreflightResult:
    status: StaticExportPreflightStatus
    code: StaticExportPreflightCode
    user_message: str
    technical_detail: str | None = None

    @property
    def passed(self) -> bool:
        return self.status is StaticExportPreflightStatus.SUCCESS


@dataclass(frozen=True)
class ExportedFigure:
    id: str
    filename: str
    output_path: str | None
    status: FigureExportStatus
    error_message: str | None = None


@dataclass
class FigureExportBundle:
    output_dir: str
    items: tuple[ExportedFigure, ...]
    temporary: bool = False

    def cleanup(self) -> None:
        if self.temporary and self.output_dir:
            shutil.rmtree(
                self.output_dir,
                ignore_errors=True,
            )


def _static_export_diagnostic() -> tuple[bool, str | None]:
    try:
        import kaleido  # noqa: F401
    except Exception as exc:
        return (
            False,
            "Static report export requires Kaleido. "
            f"Kaleido import failed: {type(exc).__name__}: {exc}",
        )
    return True, None


def _friendly_export_error(exc: Exception) -> str:
    text = str(exc)
    lowered = text.casefold()

    if (
        "chrome" in lowered
        or "chromium" in lowered
        or "browser" in lowered
    ):
        return (
            "Kaleido is installed but Chrome/Chromium is not available "
            "to the static renderer. Install a supported Chrome/Chromium "
            f"runtime and retry. Original error: {text}"
        )

    return (
        "Static Plotly export failed. Verify the Kaleido and "
        f"Chrome/Chromium installation. {type(exc).__name__}: {text}"
    )


def _is_missing_chrome_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold()
    return any(
        marker in text
        for marker in (
            "chrome",
            "chromium",
            "browser not found",
            "browsernotfound",
        )
    )


def preflight_match_report_export(
    config: MatchReportPdfConfig | None = None,
) -> StaticExportPreflightResult:
    """Verify Plotly/Kaleido before building the expensive report bundle.

    The probe deliberately uses ``pio.to_image`` with PNG and the same
    ``image_scale`` consumed by :func:`render_match_report_pdf`.  Only its
    dimensions are small so the check remains cheap.
    """

    config = config or MatchReportPdfConfig()
    ready, diagnostic = _static_export_diagnostic()
    if not ready:
        technical_detail = diagnostic or "Kaleido import failed."
        logger.warning(
            "Match report export preflight: Kaleido unavailable. %s",
            technical_detail,
        )
        return StaticExportPreflightResult(
            status=StaticExportPreflightStatus.WARNING,
            code=StaticExportPreflightCode.KALEIDO_MISSING,
            user_message=(
                "Report generation cannot start because Kaleido is not "
                "installed. Run `python -m pip install --upgrade kaleido`, "
                "restart the app, and try again."
            ),
            technical_detail=technical_detail,
        )

    figure = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines")]
    )
    try:
        png = pio.to_image(
            figure,
            format="png",
            width=96,
            height=64,
            scale=config.image_scale,
        )
        if (
            not isinstance(png, (bytes, bytearray))
            or not png
            or not bytes(png).startswith(b"\x89PNG")
        ):
            raise RuntimeError(
                "Plotly/Kaleido returned an invalid PNG payload."
            )
    except Exception as exc:
        technical_detail = f"{type(exc).__name__}: {exc}"
        if _is_missing_chrome_error(exc):
            logger.warning(
                "Match report export preflight: Chrome unavailable. %s",
                technical_detail,
            )
            return StaticExportPreflightResult(
                status=StaticExportPreflightStatus.WARNING,
                code=StaticExportPreflightCode.CHROME_MISSING,
                user_message=(
                    "Report generation cannot start because Chrome is not "
                    "available to Kaleido. Run `plotly_get_chrome`, restart "
                    "the app, and try again."
                ),
                technical_detail=technical_detail,
            )

        logger.exception(
            "Unexpected Match Report Plotly/Kaleido preflight failure."
        )
        return StaticExportPreflightResult(
            status=StaticExportPreflightStatus.FAILURE,
            code=StaticExportPreflightCode.RENDERER_ERROR,
            user_message=(
                "The report export engine failed its startup check. "
                "Review the application logs, then try again."
            ),
            technical_detail=technical_detail,
        )

    return StaticExportPreflightResult(
        status=StaticExportPreflightStatus.SUCCESS,
        code=StaticExportPreflightCode.READY,
        user_message="The report export engine is ready.",
    )


def export_report_figure_catalog(
    catalog: MatchReportFigureCatalog,
    output_dir: str | Path | None = None,
    *,
    format: str = "png",
    scale: float = 2.0,
    batch: bool = True,
) -> FigureExportBundle:
    """Export all catalog figures with deterministic names.

    When output_dir is omitted a safe OS temporary directory is created outside
    the repository. Call FigureExportBundle.cleanup() after consuming it.
    """

    format = format.casefold()
    if format not in {"png", "svg", "pdf"}:
        raise ValueError("format must be one of: png, svg, pdf")
    if scale <= 0:
        raise ValueError("scale must be positive")

    ready, diagnostic = _static_export_diagnostic()

    temporary = output_dir is None
    destination = (
        Path(tempfile.mkdtemp(prefix="match-report-figures-"))
        if temporary
        else Path(output_dir)
    )
    destination.mkdir(parents=True, exist_ok=True)

    if not ready:
        return FigureExportBundle(
            output_dir=str(destination),
            items=tuple(
                ExportedFigure(
                    id=item.id,
                    filename=Path(item.filename).with_suffix(
                        "." + format
                    ).name,
                    output_path=None,
                    status=FigureExportStatus.ERROR,
                    error_message=diagnostic,
                )
                for item in catalog.figures
            ),
            temporary=temporary,
        )

    artifacts = [
        (
            item,
            Path(item.filename).with_suffix("." + format).name,
        )
        for item in catalog.figures
    ]

    if (
        batch
        and artifacts
        and hasattr(pio, "write_images")
    ):
        figures = [item.figure for item, _ in artifacts]
        paths = [
            str(destination / filename)
            for _, filename in artifacts
        ]
        try:
            pio.write_images(
                figures,
                paths,
                format=format,
                scale=scale,
            )
        except Exception:
            # Fall back to isolated writes so one problematic figure does not
            # prevent the rest of the report from being exported.
            pass
        else:
            return FigureExportBundle(
                output_dir=str(destination),
                items=tuple(
                    ExportedFigure(
                        id=item.id,
                        filename=filename,
                        output_path=str(destination / filename),
                        status=FigureExportStatus.EXPORTED,
                    )
                    for item, filename in artifacts
                ),
                temporary=temporary,
            )

    results: list[ExportedFigure] = []
    for artifact, filename in artifacts:
        target = destination / filename
        try:
            pio.write_image(
                artifact.figure,
                target,
                format=format,
                scale=scale,
            )
        except Exception as exc:
            results.append(
                ExportedFigure(
                    id=artifact.id,
                    filename=filename,
                    output_path=None,
                    status=FigureExportStatus.ERROR,
                    error_message=_friendly_export_error(exc),
                )
            )
        else:
            results.append(
                ExportedFigure(
                    id=artifact.id,
                    filename=filename,
                    output_path=str(target),
                    status=FigureExportStatus.EXPORTED,
                )
            )

    return FigureExportBundle(
        output_dir=str(destination),
        items=tuple(results),
        temporary=temporary,
    )


def kaleido_smoke_test() -> tuple[bool, str]:
    """Backward-compatible wrapper around the report preflight."""

    result = preflight_match_report_export()
    return result.passed, result.user_message
