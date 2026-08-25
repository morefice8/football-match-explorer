from __future__ import annotations

from dash import html


def _metric(label, value, caption):
    return html.Div(
        [
            html.Span(
                label,
                className="transition-heatmap-kpi-label",
            ),
            html.Strong(
                value,
                className="transition-heatmap-kpi-value",
            ),
            html.Small(
                caption,
                className="transition-heatmap-kpi-caption",
            ),
        ],
        className="transition-heatmap-kpi",
    )


def kpi_strip(metrics):
    metrics = metrics or {}
    duration = metrics.get("median_duration_seconds")
    duration_text = f"{duration:.1f} s" if duration is not None else "—"
    reach_pct = float(metrics.get("final_third_or_shot_pct", 0.0) or 0.0)

    return html.Div(
        [
            _metric(
                "TRANSITIONS",
                str(int(metrics.get("transition_count", 0) or 0)),
                "Current filtered sample",
            ),
            _metric(
                "MEDIAN DURATION",
                duration_text,
                "Sequence start to terminal event",
            ),
            _metric(
                "FINAL THIRD / SHOT",
                f"{reach_pct:.0f}%",
                "Reached x ≥ 66.7 or ended in a shot",
            ),
        ],
        className="transition-heatmap-kpis",
    )
