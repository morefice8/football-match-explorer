from __future__ import annotations

from dash import dcc, html

def _metric(label, value, caption):
    return html.Div(
        [
            html.Span(label, className="player-pass-kpi-label"),
            html.Strong(value, className="player-pass-kpi-value"),
            html.Small(caption, className="player-pass-kpi-caption"),
        ],
        className="player-pass-kpi",
    )

def _count_label(
    count,
    singular,
    plural=None,
):
    count = int(count or 0)
    label = (
        singular
        if count == 1
        else (
            plural
            if plural is not None
            else singular + "s"
        )
    )
    return f"{count} {label}"


def kpi_strip(profile):
    return html.Div(
        [
            _metric(
                "PASS VOLUME",
                str(int(profile.get("volume", 0) or 0)),
                "All pass attempts",
            ),
            _metric(
                "COMPLETION",
                f"{float(profile.get('completion_pct', 0.0) or 0.0):.0f}%",
                f"{int(profile.get('completed', 0) or 0)} completed",
            ),
            _metric(
                "PROGRESSIVE",
                str(int(profile.get("progressive_completions", 0) or 0)),
                (
                    "Completed progressive passes · "
                    "includes progressive key passes / assists"
                ),
            ),
            _metric(
                "CHANCE CREATION",
                str(int(profile.get("chance_creation", 0) or 0)),
                (
                    _count_label(
                        profile.get("key_passes", 0),
                        "key pass",
                    )
                    + " · "
                    + _count_label(
                        profile.get("assists", 0),
                        "assist",
                    )
                ),
            ),
        ],
        className="player-pass-kpi-grid",
    )

def panel(figure, profile):
    return html.Div(
        [
            kpi_strip(profile),
            dcc.Graph(
                figure=figure,
                config={"displayModeBar": False, "responsive": True},
                className="player-pass-map-graph",
            ),
        ],
        className="player-pass-map-view",
    )
