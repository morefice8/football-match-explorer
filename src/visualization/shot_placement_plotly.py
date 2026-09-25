"""Goal-mouth shot placement chart: view from directly behind the goal.

Uses Opta qualifiers 102/103 ("Goal mouth y/z co-ordinate" in the processed
DataFrame) — the point where a shot's trajectory crossed the goal-line
plane. These live on Opta's own goal-frame scale, independent of the
general 0-100 pitch x/y scale used everywhere else in this app:

- y: 45.2 = left post, 54.8 = right post (post-to-post = 7.32m, the real
  FIFA goal width).
- z: 0 = ground, 100 = crossbar (2.44m, the real FIFA goal height); a value
  above 100 means the shot flew over the bar.

Both axes are converted to real metres here so the drawn frame has the
correct 7.32:2.44 aspect ratio instead of looking stretched.

Each team gets its own goal-frame panel, side by side, rather than sharing
one frame — with two teams' worth of shots this keeps markers from piling
up on top of each other and reads as a direct "who placed shots where"
comparison.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.visualization.plotly_branding import (
    MATCH_PITCH_BG,
    add_plot_subtitle,
    apply_match_tooltip,
    match_font,
)
from src.visualization.shot_map_plotly import OUTCOME_ORDER, OUTCOME_STYLE, UNKNOWN_COLOR

GOAL_MOUTH_Y_COLUMN = "Goal mouth y co-ordinate"
GOAL_MOUTH_Z_COLUMN = "Goal mouth z co-ordinate"

# Opta qualifier 74 ("High" — hit crossbar or missed over). When this is set
# on an off-target/woodwork shot, Opta's own categorical data says the ball
# went over the bar, even if q102/103 (the tracked crossing point) still
# lands inside the frame — a known inconsistency between Opta's outcome
# tagging and its geometric tracking, most common on headers and rising
# shots. We surface that mismatch rather than plotting it as if the numeric
# position were fully trustworthy.
HIGH_QUALIFIER_COLUMN = "High"

GOAL_Y_LEFT_POST = 45.2
GOAL_Y_RIGHT_POST = 54.8
GOAL_WIDTH_M = 7.32
GOAL_HEIGHT_M = 2.44  # z=100 is the crossbar

_Y_UNITS_PER_M = (GOAL_Y_RIGHT_POST - GOAL_Y_LEFT_POST) / GOAL_WIDTH_M
_Z_UNITS_PER_M = 100.0 / GOAL_HEIGHT_M

FRAME_LINE_COLOR = "#e8eef2"
NET_LINE_COLOR = "rgba(232,238,242,0.22)"
GROUND_COLOR = "rgba(232,238,242,0.45)"

# Comfortable viewport around the frame: shows close misses without letting
# a stray "miles wide" value blow out the scale.
X_RANGE_M = (-4.6, 4.6)
Y_RANGE_M = (-0.6, 3.6)

# Neutral swatch for the shared shape-only legend (team is conveyed by which
# panel a marker is in, not by legend colour, once shots are split by team).
_LEGEND_NEUTRAL_COLOR = "#b9c8d2"


def _y_to_metres(goal_mouth_y):
    """Horizontal offset from goal centre, metres. Negative = left post side."""
    return (goal_mouth_y - 50.0) / _Y_UNITS_PER_M


def _z_to_metres(goal_mouth_z):
    """Height above ground, metres."""
    return goal_mouth_z / _Z_UNITS_PER_M


def _team_shots_with_placement(shots_df, team_name):
    """Counted shots for one team that carry a goal-mouth crossing point."""
    if shots_df is None or shots_df.empty or "team_name" not in shots_df.columns:
        return pd.DataFrame()

    if (
        GOAL_MOUTH_Y_COLUMN not in shots_df.columns
        or GOAL_MOUTH_Z_COLUMN not in shots_df.columns
    ):
        return pd.DataFrame()

    team_shots = shots_df[
        (shots_df["team_name"] == team_name)
        & shots_df["shot_counts_as_shot"].fillna(False)
    ].copy()

    team_shots["goal_mouth_y"] = pd.to_numeric(
        team_shots[GOAL_MOUTH_Y_COLUMN], errors="coerce"
    )
    team_shots["goal_mouth_z"] = pd.to_numeric(
        team_shots[GOAL_MOUTH_Z_COLUMN], errors="coerce"
    )

    return team_shots.dropna(subset=["goal_mouth_y", "goal_mouth_z"])


def _qualifier_present(row, column):
    """True if a value-less Opta qualifier flag is set on this row."""
    value = row.get(column)
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return bool(str(value).strip())


def _goal_mouth_mismatch(row, z_m):
    """Opta tagged this shot as missing high, but the tracked point is inside the frame."""
    return (
        row.get("shot_outcome") in ("off_target", "post")
        and _qualifier_present(row, HIGH_QUALIFIER_COLUMN)
        and z_m <= GOAL_HEIGHT_M
    )


def _placement_label(x_m, z_m):
    horizontal = "center" if abs(x_m) < 0.6 else ("left post" if x_m < 0 else "right post")
    if z_m > GOAL_HEIGHT_M:
        return "over the bar"
    vertical = "low" if z_m < GOAL_HEIGHT_M * 0.35 else (
        "high" if z_m > GOAL_HEIGHT_M * 0.75 else "mid-height"
    )
    return f"{vertical}, {horizontal}"


def _hover_text(row, team_name, outcome_label, x_m, z_m, *, mismatch):
    minute = row.get("timeMin")
    minute_text = f"{int(minute)}'" if pd.notna(minute) else "?"
    player = row.get("playerName") or "Unknown player"
    text = (
        f"<b>{player}</b>"
        f"<br>{team_name} · {outcome_label}"
        f"<br>Minute {minute_text}"
        f"<br>{_placement_label(x_m, z_m)}"
    )
    if mismatch:
        text += "<br><i>Opta recorded this as missed high — plotted height may understate the miss</i>"
    return text


def _goal_frame_shapes(*, xref, yref):
    half_width = GOAL_WIDTH_M / 2.0
    base = [
        dict(
            type="rect",
            x0=-half_width,
            y0=0,
            x1=half_width,
            y1=GOAL_HEIGHT_M,
            line=dict(color=FRAME_LINE_COLOR, width=4),
            fillcolor="rgba(255,255,255,0.03)",
            layer="below",
        ),
        dict(
            type="line",
            x0=X_RANGE_M[0],
            y0=0,
            x1=X_RANGE_M[1],
            y1=0,
            line=dict(color=GROUND_COLOR, width=2),
            layer="below",
        ),
    ] + [
        dict(
            type="line",
            x0=-half_width + step * GOAL_WIDTH_M / 6,
            y0=0,
            x1=-half_width + step * GOAL_WIDTH_M / 6,
            y1=GOAL_HEIGHT_M,
            line=dict(color=NET_LINE_COLOR, width=1),
            layer="below",
        )
        for step in range(1, 6)
    ] + [
        dict(
            type="line",
            x0=-half_width,
            y0=level * GOAL_HEIGHT_M / 3,
            x1=half_width,
            y1=level * GOAL_HEIGHT_M / 3,
            line=dict(color=NET_LINE_COLOR, width=1),
            layer="below",
        )
        for level in range(1, 3)
    ]
    for shape in base:
        shape["xref"] = xref
        shape["yref"] = yref
    return base


def _add_team_panel(fig, team_shots, team_name, color, *, col):
    if team_shots.empty:
        return

    positions = team_shots.assign(
        x_m=team_shots["goal_mouth_y"].apply(_y_to_metres),
        z_m=team_shots["goal_mouth_z"].apply(_z_to_metres),
    )

    for outcome in OUTCOME_ORDER:
        rows = positions[positions["shot_outcome"] == outcome]
        if rows.empty:
            continue

        style = OUTCOME_STYLE[outcome]
        marker_color = UNKNOWN_COLOR if outcome == "unknown" else color

        fig.add_trace(
            go.Scatter(
                x=rows["x_m"],
                y=rows["z_m"],
                mode="markers",
                name=f"{team_name} · {style['label']}",
                showlegend=False,
                marker=dict(
                    symbol=style["symbol"],
                    size=style["size"],
                    color=marker_color,
                    line=dict(
                        color="#ffffff",
                        width=1.2 if outcome == "goal" else 0.8,
                    ),
                ),
                opacity=1.0 if outcome == "goal" else 0.85,
                text=[
                    _hover_text(
                        row,
                        team_name,
                        style["label"],
                        row["x_m"],
                        row["z_m"],
                        mismatch=_goal_mouth_mismatch(row, row["z_m"]),
                    )
                    for _, row in rows.iterrows()
                ],
                hovertemplate="%{text}<extra></extra>",
            ),
            row=1,
            col=col,
        )


def _add_outcome_legend(fig, combined_shots):
    """One shape-only legend entry per outcome present, shared across panels.

    Team is already conveyed by which panel a marker sits in, so the legend
    only needs to explain what each marker *shape* means, not repeat colour
    per team.
    """
    if combined_shots.empty:
        return

    outcomes_present = set(combined_shots["shot_outcome"].dropna())
    for outcome in OUTCOME_ORDER:
        if outcome not in outcomes_present:
            continue
        style = OUTCOME_STYLE[outcome]
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=style["label"],
                marker=dict(
                    symbol=style["symbol"],
                    size=style["size"],
                    color=_LEGEND_NEUTRAL_COLOR,
                ),
            ),
            row=1,
            col=1,
        )


def plot_shot_placement(shots_df, *, home_team, away_team, hcol, acol):
    """Return a Plotly figure with each team's goal frame side by side.

    ``shots_df`` must be the output of
    ``src.metrics.shot_classification.classify_shots`` run on a DataFrame
    that still has the raw Opta qualifier columns (i.e. before any column
    pruning). Only shots that actually carry a goal-mouth y/z value are
    plotted — this naturally excludes blocked shots (the ball never reached
    the goal line) rather than guessing a placement for them. Off-target
    shots that still crossed the goal-line plane (wide or over the bar) are
    included when Opta recorded a crossing point, so a near-miss shows up
    just outside the frame instead of being dropped.
    """
    home_shots = _team_shots_with_placement(shots_df, home_team)
    away_shots = _team_shots_with_placement(shots_df, away_team)

    if home_shots.empty and away_shots.empty:
        fig = go.Figure()
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="No shot placement data available for this match",
            showarrow=False,
            font=dict(color="#b9c8d2", size=14),
        )
        fig.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor=MATCH_PITCH_BG,
            height=420,
            margin=dict(l=18, r=18, t=40, b=18),
            xaxis=dict(visible=False, showgrid=False, zeroline=False),
            yaxis=dict(visible=False, showgrid=False, zeroline=False),
        )
        return fig

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
    )

    _add_team_panel(fig, home_shots, home_team, hcol, col=1)
    _add_team_panel(fig, away_shots, away_team, acol, col=2)
    _add_outcome_legend(fig, pd.concat([home_shots, away_shots]))

    panel_shapes = _goal_frame_shapes(xref="x", yref="y") + _goal_frame_shapes(
        xref="x2", yref="y2"
    )

    fig.update_xaxes(
        range=list(X_RANGE_M), visible=False, fixedrange=True,
        showgrid=False, zeroline=False, row=1, col=1,
    )
    fig.update_xaxes(
        range=list(X_RANGE_M), visible=False, fixedrange=True,
        showgrid=False, zeroline=False, row=1, col=2,
    )
    fig.update_yaxes(
        range=list(Y_RANGE_M), visible=False, fixedrange=True,
        showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1,
        row=1, col=1,
    )
    fig.update_yaxes(
        range=list(Y_RANGE_M), visible=False, fixedrange=True,
        showgrid=False, zeroline=False, scaleanchor="x2", scaleratio=1,
        row=1, col=2,
    )

    for team_name, x_domain_center in ((home_team, 0.20), (away_team, 0.80)):
        fig.add_annotation(
            x=x_domain_center,
            y=1.0,
            xref="paper",
            yref="paper",
            text=f"<b>{team_name}</b>",
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=match_font(color="#17354d", size=13),
        )

    add_plot_subtitle(
        fig,
        "Marker shape = shot outcome · view from behind each team's own goal",
        dark=False,
        y=1.155,
    )

    fig.update_layout(
        title=None,
        paper_bgcolor="#ffffff",
        plot_bgcolor=MATCH_PITCH_BG,
        height=560,
        margin=dict(l=18, r=18, t=110, b=18),
        font=match_font(color="#17354d"),
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.06,
            yanchor="bottom",
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=match_font(color="#17354d", size=13),
        ),
        shapes=panel_shapes,
    )
    apply_match_tooltip(fig)
    return fig
