"""Shared Plotly design system for Match Analysis.

FOUND-02 establishes presentation primitives without forcing every existing
figure to migrate at once. Older helpers remain available for backward
compatibility; new plots should prefer the ``match_*`` helpers below.
"""

from __future__ import annotations

import plotly.graph_objects as go

from src.visualization.coordinate_contract import ATTACKING_DIRECTION_LABEL


# ---------------------------------------------------------------------------
# Core visual tokens
# ---------------------------------------------------------------------------

MATCH_CARD_BG = "#ffffff"
MATCH_PITCH_BG = "#29343d"
MATCH_PITCH_LINE = "rgba(255,255,255,0.72)"

MATCH_INK = "#17354d"
MATCH_MUTED = "#647c8e"
MATCH_PITCH_TEXT = "#ffffff"
MATCH_PITCH_MUTED = "#b9c8d2"
MATCH_HOVER_BG = "#102f45"

MATCH_HOME_CORAL = "#e96a4a"
MATCH_AWAY_CYAN = "#1597c2"
MATCH_WARNING = "#d79b35"

MATCH_TEXT_FONT = "Inter, Arial, sans-serif"
MATCH_DISPLAY_FONT = "IBM Plex Sans, Inter, Arial, sans-serif"

MATCH_COMPARE_HEIGHT = 560
MATCH_LEGEND_RESERVED_PX = 52

# Backward-compatible aliases used by already-migrated figures.
DARK_PLOT_BG = MATCH_PITCH_BG
DARK_TEXT = MATCH_PITCH_TEXT
DARK_MUTED_TEXT = MATCH_PITCH_MUTED
DARK_HOVER_BG = MATCH_HOVER_BG


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

def get_team_palette(*, is_away=False):
    """Return the canonical Match Analysis team palette."""
    primary = (
        MATCH_AWAY_CYAN
        if is_away
        else MATCH_HOME_CORAL
    )

    soft = (
        "rgba(21,151,194,0.16)"
        if is_away
        else "rgba(233,106,74,0.16)"
    )

    return {
        "primary": primary,
        "soft": soft,
        "text": MATCH_INK,
        "muted": MATCH_MUTED,
    }


# ---------------------------------------------------------------------------
# Margins / typography
# ---------------------------------------------------------------------------

def match_plot_margins(
    *,
    legend=False,
    header=False,
    left=18,
    right=18,
    bottom=18,
):
    """Return standard margins for card-embedded Match Analysis figures."""
    top = 22

    if legend:
        top += MATCH_LEGEND_RESERVED_PX

    if header:
        top += 62

    return {
        "l": left,
        "r": right,
        "t": top,
        "b": bottom,
    }


def match_font(*, display=False, color=None, size=None):
    """Return a Plotly font dictionary using the shared brand typography."""
    font = {
        "family": (
            MATCH_DISPLAY_FONT
            if display
            else MATCH_TEXT_FONT
        ),
        "color": color or MATCH_INK,
    }

    if size is not None:
        font["size"] = size

    return font


# ---------------------------------------------------------------------------
# Header / subtitle
# ---------------------------------------------------------------------------

def add_plot_header(
    fig,
    title,
    subtitle=None,
    *,
    dark=True,
    y=1.155,
):
    """
    Backward-compatible shared figure header.

    New card-embedded plots should normally leave the figure title empty when
    the surrounding Dash card already owns the title.
    """
    title_color = (
        MATCH_PITCH_TEXT
        if dark
        else MATCH_INK
    )
    subtitle_color = (
        MATCH_PITCH_MUTED
        if dark
        else MATCH_MUTED
    )

    fig.add_annotation(
        x=0.5,
        y=y,
        xref="paper",
        yref="paper",
        text=f"<b>{title}</b>",
        showarrow=False,
        xanchor="center",
        yanchor="middle",
        font=match_font(
            display=True,
            color=title_color,
            size=18,
        ),
    )

    if subtitle:
        add_plot_subtitle(
            fig,
            subtitle,
            dark=dark,
            y=y - 0.05,
        )

    return fig


def add_plot_subtitle(
    fig,
    subtitle,
    *,
    dark=True,
    y=1.105,
):
    """Add a consistent standalone subtitle annotation."""
    color = (
        MATCH_PITCH_MUTED
        if dark
        else MATCH_MUTED
    )

    fig.add_annotation(
        x=0.5,
        y=y,
        xref="paper",
        yref="paper",
        text=subtitle,
        showarrow=False,
        xanchor="center",
        yanchor="middle",
        font=match_font(
            color=color,
            size=11,
        ),
    )

    return fig


# ---------------------------------------------------------------------------
# Legend / tooltip
# ---------------------------------------------------------------------------

def apply_match_legend(
    fig,
    *,
    show=True,
    y=1.025,
):
    """Place the legend in a fixed horizontal band above the pitch."""
    fig.update_layout(
        showlegend=show,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=y,
            yanchor="bottom",
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=match_font(
                color=MATCH_INK,
                size=13,
            ),
            traceorder="normal",
        ),
    )
    return fig


def apply_match_tooltip(fig):
    """Apply the canonical Match Analysis hover-label styling."""
    fig.update_layout(
        hoverlabel=dict(
            bgcolor=MATCH_HOVER_BG,
            font=dict(
                family=MATCH_TEXT_FONT,
                color="#ffffff",
                size=13,
            ),
            bordercolor="rgba(255,255,255,0.18)",
            align="left",
        )
    )
    return fig


# ---------------------------------------------------------------------------
# Pitch / direction / zero state
# ---------------------------------------------------------------------------

def apply_match_pitch_axes(
    fig,
    *,
    x_range=(-2, 102),
    y_range=(-5, 105),
):
    """Apply the canonical fixed Opta pitch axes."""
    fig.update_layout(
        xaxis=dict(
            range=list(x_range),
            visible=False,
            fixedrange=True,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            range=list(y_range),
            visible=False,
            fixedrange=True,
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=0.68,
        ),
    )
    return fig


def add_attacking_direction(
    fig,
    *,
    dark=True,
    x=0.985,
    y=0.985,
    xanchor="right",
    yanchor="top",
):
    """Add the canonical left-to-right attacking-direction label."""
    text_color = (
        MATCH_PITCH_MUTED
        if dark
        else MATCH_MUTED
    )
    bg_color = (
        "rgba(16,47,69,0.72)"
        if dark
        else "rgba(255,255,255,0.88)"
    )

    fig.add_annotation(
        x=x,
        y=y,
        xref="paper",
        yref="paper",
        text=f"<b>{ATTACKING_DIRECTION_LABEL}</b>",
        showarrow=False,
        xanchor=xanchor,
        yanchor=yanchor,
        font=match_font(
            color=text_color,
            size=11,
        ),
        bgcolor=bg_color,
        borderpad=3,
    )
    return fig


def add_zero_state(
    fig,
    text,
    *,
    dark_pitch=True,
    x=50,
    y=50,
):
    """Render a shared in-pitch zero state without inventing fake data."""
    color = (
        MATCH_PITCH_MUTED
        if dark_pitch
        else MATCH_MUTED
    )

    fig.add_annotation(
        x=x,
        y=y,
        text=f"<b>{text}</b>",
        showarrow=False,
        xanchor="center",
        yanchor="middle",
        font=match_font(
            color=color,
            size=14,
        ),
    )
    return fig


def apply_match_pitch_layout(
    fig,
    *,
    pitch_shapes=None,
    height=MATCH_COMPARE_HEIGHT,
    showlegend=True,
    header=False,
    x_range=(-2, 102),
    y_range=(-5, 105),
):
    """
    Apply the new card-embedded Match Plot Design System.

    The figure paper stays white so the surrounding card remains visually
    light. Only the axes/pitch area is dark.
    """
    fig.update_layout(
        title=None,
        paper_bgcolor=MATCH_CARD_BG,
        plot_bgcolor=MATCH_PITCH_BG,
        height=height,
        margin=match_plot_margins(
            legend=showlegend,
            header=header,
        ),
        font=match_font(
            color=MATCH_INK,
        ),
        shapes=(
            pitch_shapes
            if pitch_shapes is not None
            else fig.layout.shapes
        ),
    )

    apply_match_pitch_axes(
        fig,
        x_range=x_range,
        y_range=y_range,
    )
    apply_match_legend(
        fig,
        show=showlegend,
    )
    apply_match_tooltip(fig)

    return fig


# ---------------------------------------------------------------------------
# Legacy dark-paper helper — unchanged visual contract for existing plots.
# ---------------------------------------------------------------------------

def apply_dark_pitch_layout(
    fig,
    *,
    height=660,
    top_margin=125,
    showlegend=True,
):
    """
    Legacy shared dark Match Analysis layout.

    Kept unchanged so FOUND-02 only migrates the selected demonstration plot.
    """
    fig.update_layout(
        title=None,
        plot_bgcolor=MATCH_PITCH_BG,
        paper_bgcolor=MATCH_PITCH_BG,
        height=height,
        margin=dict(
            l=20,
            r=20,
            t=top_margin,
            b=20,
        ),
        showlegend=showlegend,
        hoverlabel=dict(
            bgcolor=MATCH_HOVER_BG,
            font_color="white",
            bordercolor="rgba(255,255,255,0.18)",
        ),
        font=dict(
            family=MATCH_TEXT_FONT,
            color=MATCH_PITCH_TEXT,
        ),
    )

    return fig
