"""Kaleido-backed static export for Match Report Plotly figures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import shutil
import tempfile

import plotly.graph_objects as go
import plotly.io as pio

from src.reporting.figure_catalog import MatchReportFigureCatalog


class FigureExportStatus(str, Enum):
    EXPORTED = "exported"
    ERROR = "error"


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
    """Attempt a tiny static export in an OS temp directory."""

    ready, diagnostic = _static_export_diagnostic()
    if not ready:
        return False, diagnostic or "Kaleido unavailable."

    directory = Path(
        tempfile.mkdtemp(prefix="match-report-kaleido-smoke-")
    )
    target = directory / "smoke.png"

    try:
        figure = go.Figure()
        figure.add_scatter(x=[0, 1], y=[0, 1])
        pio.write_image(
            figure,
            target,
            format="png",
            scale=1,
            width=320,
            height=180,
        )
        if not target.exists() or target.stat().st_size == 0:
            return False, "Kaleido produced no smoke-test image."
        return True, "Kaleido static export smoke test passed."
    except Exception as exc:
        return False, _friendly_export_error(exc)
    finally:
        shutil.rmtree(directory, ignore_errors=True)
