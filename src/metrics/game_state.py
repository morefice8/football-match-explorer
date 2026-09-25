"""Lightweight running-scoreline classifier for the live app.

Computes each team's game state (leading/drawing/trailing) at any point in
the match, using only the goal/own_goal rows already produced by
``shot_classification.classify_shots``. Deliberately independent of the
heavier ``src/reporting`` pack bundle (which has its own, richer
``game_state`` computation over ``goal_origin_metrics`` scorer data) --
this exists so app.py tab renders don't need to build a full report bundle
just to filter a chart by game state.
"""

from __future__ import annotations

import pandas as pd

GAME_STATE_LEADING = "leading"
GAME_STATE_DRAWING = "drawing"
GAME_STATE_TRAILING = "trailing"
GAME_STATES = (GAME_STATE_LEADING, GAME_STATE_DRAWING, GAME_STATE_TRAILING)


def _as_int_or_none(value):
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _goal_timeline(shots_df, home_team, away_team):
    """Chronological ``(minute, second, beneficiary_team)`` goal events.

    Own goals are attributed to the opponent of the shooting team -- the
    shooting team's own net conceding benefits the other side's score.
    """
    if shots_df is None or shots_df.empty:
        return []

    events = []
    for _, row in shots_df.iterrows():
        if row.get("shot_outcome") not in ("goal", "own_goal"):
            continue
        minute = _as_int_or_none(row.get("timeMin"))
        if minute is None:
            continue
        second = _as_int_or_none(row.get("timeSec")) or 0
        team = row.get("team_name")
        if row.get("shot_outcome") == "own_goal":
            team = away_team if team == home_team else home_team
        if team not in (home_team, away_team):
            continue
        events.append((minute, second, team))

    events.sort(key=lambda item: (item[0], item[1]))
    return events


def _state_from_scores(team_score, opponent_score):
    if team_score > opponent_score:
        return GAME_STATE_LEADING
    if team_score < opponent_score:
        return GAME_STATE_TRAILING
    return GAME_STATE_DRAWING


def add_game_state_column(shots_df, home_team, away_team):
    """Return a copy of ``shots_df`` with a ``game_state`` column added.

    Game state is for the shooting team, evaluated the instant *before*
    the shot: a scoring shot reflects the state it was taken in, not the
    state its own goal just created.
    """
    if shots_df is None or shots_df.empty:
        result = shots_df.copy() if shots_df is not None else pd.DataFrame()
        result["game_state"] = pd.Series(dtype="object")
        return result

    goal_events = _goal_timeline(shots_df, home_team, away_team)
    result = shots_df.copy()

    def _state_for_row(row):
        team = row.get("team_name")
        minute = _as_int_or_none(row.get("timeMin"))
        if team not in (home_team, away_team) or minute is None:
            return None
        second = _as_int_or_none(row.get("timeSec")) or 0
        clock = (minute, second)

        home_score = away_score = 0
        for g_minute, g_second, g_team in goal_events:
            if (g_minute, g_second) >= clock:
                break
            if g_team == home_team:
                home_score += 1
            elif g_team == away_team:
                away_score += 1

        team_score, opponent_score = (
            (home_score, away_score) if team == home_team else (away_score, home_score)
        )
        return _state_from_scores(team_score, opponent_score)

    result["game_state"] = result.apply(_state_for_row, axis=1)
    return result
