"""Window analysis: phases, window profile, pressure, positions, droughts and insight candidates.

Orchestrates the pure metrics (``match_clock``, ``match_phases``,
``window_profile``, ``insight_candidates``) for one match given its full event
frame and the two team names. It knows nothing about the report bundle, so it
can be used from the Match Analysis Pack, from scripts and from the app.

Scopes: ``phase_id = 0`` is the whole match, ``1..n`` are the match phases.
Lanes in the pressure tables are in the *touching team's own frame*; positions
carry both the own frame and a shared frame (``x_shared`` / ``y_shared``) with
the reference team (the home team) unchanged and the other team rotated 180
degrees, ready to draw both teams on one pitch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.metrics import match_clock as mc
from src.metrics import match_phases as mph
from src.metrics import window_profile as wp
from src.metrics.insight_candidates import build_insight_candidates

DROUGHT_MIN_MINUTES = 15.0

WINDOW_PROFILE_COLUMNS = (
    "report_team", "team_name", "phase_id", "phase_label", "start_clock", "end_clock", "play_minutes", "touches",
    "touches_per_min", "touch_share_pct", "own_half_touch_pct", "final_third_touches", "final_third_per_10min",
    "final_third_entries", "box_touches", "box_touches_per_10min", "crosses", "crosses_completed", "shots",
    "final_third_loss_pct", "interventions", "block_height_m", "team_length_m", "pressed_touches",
    "pressed_touch_pct", "low_sample",
)
PRESSURE_PROFILE_COLUMNS = (
    "report_team", "team_name", "phase_id", "phase_label", "group_type", "group", "touches", "pressed",
    "pressed_pct", "ci_low_pct", "ci_high_pct", "low_sample", "vs_rest_p",
)
PRESSURE_TOUCH_COLUMNS = (
    "report_team", "team_name", "player", "jersey", "play_seconds", "clock", "period", "x", "y", "lane", "third",
    "pressed", "presser", "intervention", "distance_m", "event_x", "event_y",
)
WINDOW_POSITION_COLUMNS = (
    "report_team", "team_name", "phase_id", "phase_label", "player", "jersey", "role", "x", "y", "x_shared",
    "y_shared", "actions", "minutes_present", "low_sample",
)
SHOT_DROUGHT_COLUMNS = (
    "report_team", "team_name", "kind", "start_clock", "end_clock", "play_minutes", "box_touches",
    "final_third_touches", "crosses", "opponent_shots", "ends_with_event",
)


def _empty(columns) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


@dataclass
class WindowAnalysis:
    phases: pd.DataFrame = field(default_factory=lambda: _empty(mph.PHASE_COLUMNS))
    window_profile: pd.DataFrame = field(default_factory=lambda: _empty(WINDOW_PROFILE_COLUMNS))
    pressure_profile: pd.DataFrame = field(default_factory=lambda: _empty(PRESSURE_PROFILE_COLUMNS))
    pressure_touches: pd.DataFrame = field(default_factory=lambda: _empty(PRESSURE_TOUCH_COLUMNS))
    positions: pd.DataFrame = field(default_factory=lambda: _empty(WINDOW_POSITION_COLUMNS))
    droughts: pd.DataFrame = field(default_factory=lambda: _empty(SHOT_DROUGHT_COLUMNS))
    insights: pd.DataFrame = field(default_factory=lambda: _empty(("id", "report_team", "team_name", "kind", "scope", "strength", "statement", "numbers", "caveat")))


def compute_window_analysis(
    source: pd.DataFrame,
    teams,
    shots_df: pd.DataFrame | None = None,
    *,
    reference_team: str | None = None,
) -> WindowAnalysis:
    """Run every window analysis for a match; degrades to empty tables on thin input."""
    teams = tuple(teams or ())
    if source is None or source.empty or len(teams) < 2:
        return WindowAnalysis()
    home, away = teams[0], teams[1]
    reference = reference_team or home
    clock = mc.PlayClock.from_events(source)
    view = wp.prepare_events(source, clock)
    if view.empty or clock.total_seconds <= 0:
        return WindowAnalysis()
    total = clock.total_seconds
    goals = mph.goal_events(source, home, away, clock, shots_df=shots_df)
    phases = mph.build_match_phases(source, home, away, goals=goals, clock=clock)
    scopes = [(0, "match", 0.0, total + 1.0)] + [(int(p.phase_id), p.label, float(p.start_play_s), float(p.end_play_s)) for p in phases.itertuples()]
    intervals = wp.on_pitch_intervals(view, source, clock)

    profile_rows, pressure_rows, touch_frames, position_frames, drought_frames = [], [], [], [], []
    for team, opponent in ((home, away), (away, home)):
        touches = wp.pressure_table(view, team, opponent, 0.0, total + 1.0)
        for phase_id, label, start, end in scopes:
            minutes = (min(end, total) - start) / 60.0
            terr = wp.territory_profile(view, team, opponent, start, end)
            positions = wp.window_positions(view, team, start, end, intervals, reference)
            in_scope = touches[(touches["play_seconds"] >= start) & (touches["play_seconds"] < end)] if len(touches) else touches
            overall = wp.summarize_pressure(in_scope, "all")
            pressed_n = int(overall["pressed"].iloc[0]) if len(overall) else 0
            pressed_pct = float(overall["pressed_pct"].iloc[0]) if len(overall) else None
            profile_rows.append({
                "report_team": team, "team_name": team, "phase_id": phase_id, "phase_label": label,
                "start_clock": clock.label(start), "end_clock": clock.label(min(end, total)),
                **terr, "play_minutes": round(minutes, 1),
                "team_length_m": wp.team_length_m(positions), "pressed_touches": pressed_n, "pressed_touch_pct": pressed_pct,
                "low_sample": bool(minutes < mph.SHORT_PHASE_MINUTES or terr["touches"] < 30),
            })
            for group_type in ("all", "lane", "third", "player"):
                summary = wp.summarize_pressure(in_scope, group_type if group_type != "all" else "all")
                for r in summary.to_dict("records"):
                    pressure_rows.append({"report_team": team, "team_name": team, "phase_id": phase_id, "phase_label": label,
                                          "group_type": group_type, **r})
            if len(positions):
                position_frames.append(positions.assign(report_team=team, team_name=team, phase_id=phase_id, phase_label=label).drop(columns="team"))
        if len(touches):
            shown = touches.assign(report_team=team, team_name=team, clock=[clock.label(t) for t in touches["play_seconds"]])
            touch_frames.append(shown)
        for kind in ("shots", "box_touches"):
            d = wp.droughts(view, clock, team, opponent, kind=kind, min_minutes=DROUGHT_MIN_MINUTES)
            if len(d):
                drought_frames.append(d.assign(report_team=team, team_name=team).drop(columns="team"))

    window_profile = pd.DataFrame(profile_rows, columns=list(WINDOW_PROFILE_COLUMNS))
    pressure_profile = pd.DataFrame(pressure_rows, columns=list(PRESSURE_PROFILE_COLUMNS))
    positions_all = pd.concat(position_frames, ignore_index=True)[list(WINDOW_POSITION_COLUMNS)] if position_frames else _empty(WINDOW_POSITION_COLUMNS)
    touches_all = pd.concat(touch_frames, ignore_index=True)[list(PRESSURE_TOUCH_COLUMNS)] if touch_frames else _empty(PRESSURE_TOUCH_COLUMNS)
    droughts_all = pd.concat(drought_frames, ignore_index=True)[list(SHOT_DROUGHT_COLUMNS)] if drought_frames else _empty(SHOT_DROUGHT_COLUMNS)
    insights = build_insight_candidates(window_profile, pressure_profile, droughts_all)
    return WindowAnalysis(phases, window_profile, pressure_profile, touches_all, positions_all, droughts_all, insights)
