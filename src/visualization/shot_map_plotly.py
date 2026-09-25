"""Plotly shot map: every shot from both teams on one attacking end.

Coordinates follow the Match Analysis coordinate contract
(``coordinate_contract.py``): each event's x/y is already team-relative, so
both teams' shots naturally cluster near x=100 (the opponent's goal) without
any home/away mirroring.
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go

from src.visualization import pitch_plots
from src.visualization.plotly_branding import (
    add_attacking_direction,
    add_plot_subtitle,
    apply_match_pitch_layout,
)

GOAL_X, GOAL_Y = 100.0, 50.0
PITCH_LENGTH_M = 105.0

UNKNOWN_COLOR = "#9fb3bf"

# Marker shape encodes the shot outcome; marker color encodes the team.
# Shared with shot_placement_plotly.py so both shot visualizations agree on
# what a "goal" or "saved" marker looks like.
OUTCOME_STYLE = {
    "goal": dict(label="Goal", symbol="star", size=17),
    "saved": dict(label="Saved", symbol="circle", size=10),
    "off_target": dict(label="Off Target", symbol="circle-open", size=9),
    "blocked": dict(label="Blocked", symbol="x", size=9),
    "post": dict(label="Woodwork", symbol="diamond", size=10),
    "unknown": dict(label="Unclear", symbol="circle-open-dot", size=9),
}
OUTCOME_ORDER = ("goal", "saved", "off_target", "blocked", "post", "unknown")


def _shot_distance_m(x, y):
    dx = GOAL_X - x
    dy = GOAL_Y - y
    return math.sqrt(dx * dx + dy * dy) * (PITCH_LENGTH_M / 100.0)


def _team_shots(shots_df, team_name):
    """Counted shots for one team, coordinates coerced and NaNs dropped."""
    if shots_df is None or shots_df.empty or "team_name" not in shots_df.columns:
        return pd.DataFrame()

    team_shots = shots_df[
        (shots_df["team_name"] == team_name)
        & shots_df["shot_counts_as_shot"].fillna(False)
    ].copy()

    for column in ("x", "y"):
        team_shots[column] = pd.to_numeric(team_shots[column], errors="coerce")

    return team_shots.dropna(subset=["x", "y"])


def _hover_text(row, team_name, outcome_label):
    minute = row.get("timeMin")
    minute_text = f"{int(minute)}'" if pd.notna(minute) else "?"
    player = row.get("playerName") or "Unknown player"
    distance = _shot_distance_m(row["x"], row["y"])
    return (
        f"<b>{player}</b>"
        f"<br>{team_name} · {outcome_label}"
        f"<br>Minute {minute_text}"
        f"<br>Distance: {distance:.1f}m"
    )


def _add_team_traces(fig, team_shots, team_name, color, *, show_team_legend):
    """Add one trace per outcome present for this team; legend stays team-only."""
    if team_shots.empty:
        return

    if show_team_legend:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=team_name,
                legendgroup=team_name,
                marker=dict(symbol="circle", size=12, color=color),
            )
        )

    for outcome in OUTCOME_ORDER:
        rows = team_shots[team_shots["shot_outcome"] == outcome]
        if rows.empty:
            continue

        style = OUTCOME_STYLE[outcome]
        marker_color = UNKNOWN_COLOR if outcome == "unknown" else color

        fig.add_trace(
            go.Scatter(
                x=rows["x"],
                y=rows["y"],
                mode="markers",
                name=f"{team_name} · {style['label']}",
                legendgroup=team_name,
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
                    _hover_text(row, team_name, style["label"])
                    for _, row in rows.iterrows()
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        )


def plot_shot_map(shots_df, *, home_team, away_team, hcol, acol):
    """Return a Plotly figure with every counted shot from both teams.

    ``shots_df`` must be the output of
    ``src.metrics.shot_classification.classify_shots``. Own goals and
    non-shot rows are excluded by ``shot_counts_as_shot``. Shots with an
    unresolved outcome are kept and drawn in a muted grey rather than hidden,
    matching how the rest of the app surfaces data-quality issues instead of
    silently dropping them.
    """
    fig = go.Figure()
    fig.update_layout(
        shapes=pitch_plots.get_plotly_pitch_shapes(
            "rgba(255,255,255,0.28)", "#d7e4eb"
        )
    )
    add_attacking_direction(fig, dark=True)

    home_shots = _team_shots(shots_df, home_team)
    away_shots = _team_shots(shots_df, away_team)

    if home_shots.empty and away_shots.empty:
        fig.add_annotation(
            x=75,
            y=50,
            text="No shots recorded for either team",
            showarrow=False,
            font=dict(color="#b9c8d2", size=14),
        )
    else:
        _add_team_traces(fig, home_shots, home_team, hcol, show_team_legend=True)
        _add_team_traces(fig, away_shots, away_team, acol, show_team_legend=True)

        has_unknown = bool(
            pd.concat([home_shots, away_shots])["shot_outcome"].eq("unknown").any()
        )
        subtitle = "Marker shape = shot outcome · marker color = team"
        if has_unknown:
            subtitle += " · grey = unresolved classification"
        # dark=False: the subtitle sits in the white paper margin above the
        # pitch (like the legend), not on the dark pitch itself, so it needs
        # dark-on-light contrast, not light-on-dark.
        add_plot_subtitle(fig, subtitle, dark=False, y=1.20)

    apply_match_pitch_layout(
        fig,
        # Cropped to roughly the shooting third rather than the full half
        # pitch: the old (45, 102) crop left a lot of empty, markless grass
        # between the centre circle and the box.
        x_range=(58, 102),
        y_range=(-5, 105),
        height=620,
        header=True,  # borrow the extra top margin reserved for a header
                      # so the legend and subtitle don't crowd each other.
    )
    return fig
