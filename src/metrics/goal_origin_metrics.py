from __future__ import annotations

"""Goal-origin classification for the Match Overview.

GOAL-01 v1 reconstructs the attacking possession that leads to each goal and
keeps three ideas separate:

- possession_origin: how the possession started;
- attack_type: how the possession developed;
- decisive_mechanism: the event/pattern that most directly created the goal.

The classifier is deliberately conservative. It prefers explicit Opta restart,
recovery and error signals and falls back to a generic positional-attack label
when the raw event stream does not support a stronger claim.
"""

from dataclasses import dataclass
import math

import pandas as pd

from .restart_metrics import classify_restart_event


SHOT_TYPES = frozenset({"Goal", "Miss", "Attempt Saved", "Post"})

CONTROL_TYPES = frozenset({
    "Pass",
    "Take On",
    "Ball recovery",
    "Interception",
    "Tackle",
    "Keeper pick-up",
    "Claim",
    "Ball touch",
    "Aerial",
})

POSSESSION_START_TYPES = {
    "Ball recovery": "Ball recovery",
    "Interception": "Interception",
    "Keeper pick-up": "Goalkeeper possession",
    "Claim": "Goalkeeper possession",
}

TURNOVER_TYPES = frozenset({
    "Pass",
    "Take On",
    "Ball touch",
    "Dispossessed",
    "Aerial",
    "Challenge",
    "Out",
    "Offside Pass",
    "Error",
})

TURNOVER_DETAIL = {
    "Pass": "Unsuccessful pass",
    "Take On": "Failed dribble",
    "Ball touch": "Failed control",
    "Dispossessed": "Dispossessed",
    "Aerial": "Lost aerial duel",
    "Challenge": "Failed challenge",
    "Out": "Ball out of play",
    "Offside Pass": "Offside pass",
    "Error": "Error",
}

ADMIN_TYPES = frozenset({
    "Card",
    "Foul",
    "Unknown",
    "Unknown Type",
})

RECENT_CONTROL_LOOKBACK_SECONDS = 8.0
TRANSITION_MAX_SECONDS = 15.0
TRANSITION_MAX_PASSES = 5
DIRECT_RESTART_MAX_SECONDS = 12.0
DIRECT_RESTART_MAX_PASSES = 3
REBOUND_MAX_SECONDS = 6.0
ERROR_DECISIVE_MAX_SECONDS = 8.0


@dataclass(frozen=True)
class _TraceResult:
    rows: tuple[int, ...]
    origin: str
    origin_detail: str | None
    origin_index: int | None


def _flag(value) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _is_successful(row) -> bool:
    outcome = row.get("outcome")
    if isinstance(outcome, str):
        return outcome.strip().lower() == "successful"

    numeric = pd.to_numeric(
        pd.Series([outcome]),
        errors="coerce",
    ).iloc[0]
    return pd.notna(numeric) and float(numeric) == 1.0


def _event_second(row) -> float | None:
    minute = pd.to_numeric(
        pd.Series([row.get("timeMin")]),
        errors="coerce",
    ).iloc[0]
    second = pd.to_numeric(
        pd.Series([row.get("timeSec")]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(minute):
        return None

    return float(minute) * 60.0 + (
        0.0 if pd.isna(second) else float(second)
    )


def _event_id(row):
    value = row.get("id")
    if value is not None and not pd.isna(value):
        return value
    value = row.get("eventId")
    if value is not None and not pd.isna(value):
        return value
    return None


def _prepare_events(df_processed: pd.DataFrame) -> pd.DataFrame:
    df = df_processed.copy()
    df["_source_order"] = range(len(df))

    for column in ("periodId", "timeMin", "timeSec"):
        if column not in df.columns:
            df[column] = pd.NA
        df[f"_sort_{column}"] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return (
        df.sort_values(
            [
                "_sort_periodId",
                "_sort_timeMin",
                "_sort_timeSec",
                "_source_order",
            ],
            kind="stable",
            na_position="last",
        )
        .reset_index(drop=True)
    )


def _same_period(left, right) -> bool:
    left_period = left.get("periodId")
    right_period = right.get("periodId")
    if pd.isna(left_period) or pd.isna(right_period):
        return True
    return left_period == right_period


def _is_control_event(row) -> bool:
    event_type = str(row.get("type_name", "") or "")

    # A save can leave a rebound live; goalkeeper control is represented by
    # Keeper pick-up / Claim and those are included in CONTROL_TYPES.
    if event_type == "Save":
        return False

    if event_type in SHOT_TYPES:
        return False

    return event_type in CONTROL_TYPES and _is_successful(row)


def _team_had_recent_control(
    df: pd.DataFrame,
    event_index: int,
    team_name: str,
    *,
    max_seconds: float = RECENT_CONTROL_LOOKBACK_SECONDS,
) -> bool:
    """Check whether ``team_name`` demonstrably controlled the ball before idx."""
    event = df.iloc[event_index]
    event_time = _event_second(event)

    for idx in range(event_index - 1, -1, -1):
        candidate = df.iloc[idx]
        if not _same_period(candidate, event):
            break

        candidate_time = _event_second(candidate)
        if event_time is not None and candidate_time is not None:
            elapsed = event_time - candidate_time
            if elapsed < 0:
                continue
            if elapsed > max_seconds:
                break

        event_type = str(candidate.get("type_name", "") or "")
        if event_type in ADMIN_TYPES:
            continue

        candidate_team = candidate.get("team_name")

        if candidate_team == team_name:
            if classify_restart_event(candidate) is not None:
                return True
            if _is_control_event(candidate):
                return True

        elif _is_control_event(candidate):
            return False

    return False


def _turnover_detail(row) -> str:
    return TURNOVER_DETAIL.get(
        str(row.get("type_name", "") or ""),
        "Opponent turnover",
    )



def _preceding_opponent_turnover(
    df: pd.DataFrame,
    event_index: int,
    scoring_team: str,
    *,
    max_seconds: float = 3.0,
):
    """Return a nearby opponent loss that explains an explicit recovery."""
    recovery = df.iloc[event_index]
    recovery_time = _event_second(recovery)

    for idx in range(event_index - 1, -1, -1):
        candidate = df.iloc[idx]
        if not _same_period(candidate, recovery):
            break

        candidate_time = _event_second(candidate)
        if recovery_time is not None and candidate_time is not None:
            elapsed = recovery_time - candidate_time
            if elapsed < 0:
                continue
            if elapsed > max_seconds:
                break

        event_type = str(candidate.get("type_name", "") or "")
        if event_type in ADMIN_TYPES:
            continue

        candidate_team = candidate.get("team_name")
        if candidate_team == scoring_team:
            # Opta can emit an explicit Ball recovery marker immediately after
            # the first attacking action following a turnover. Keep looking
            # through that very short gap for the opponent loss that explains
            # how possession actually changed hands.
            if classify_restart_event(candidate) is not None:
                return None
            continue

        if event_type in TURNOVER_TYPES and not _is_successful(candidate):
            if _team_had_recent_control(df, idx, candidate_team):
                return idx, _turnover_detail(candidate)

        if _is_control_event(candidate):
            return None

    return None

def _trace_goal_possession(
    df: pd.DataFrame,
    goal_index: int,
    scoring_team: str,
) -> _TraceResult:
    goal = df.iloc[goal_index]

    if _flag(goal.get("Penalty")):
        return _TraceResult(
            rows=(goal_index,),
            origin="Penalty",
            origin_detail=None,
            origin_index=goal_index,
        )

    direct_restart = classify_restart_event(goal)
    if direct_restart is not None:
        return _TraceResult(
            rows=(goal_index,),
            origin=direct_restart,
            origin_detail="Direct shot",
            origin_index=goal_index,
        )

    selected = [goal_index]
    origin = "Established possession"
    origin_detail = None
    origin_index = None

    for idx in range(goal_index - 1, -1, -1):
        event = df.iloc[idx]

        if not _same_period(event, goal):
            break

        event_type = str(event.get("type_name", "") or "")
        event_team = event.get("team_name")

        if event_type in ADMIN_TYPES:
            continue

        if event_team == scoring_team:
            restart_type = classify_restart_event(event)
            if restart_type is not None:
                selected.append(idx)
                origin = restart_type
                origin_detail = None
                origin_index = idx
                break

            if event_type in POSSESSION_START_TYPES and _is_successful(event):
                selected.append(idx)

                if event_type == "Ball recovery":
                    turnover = _preceding_opponent_turnover(
                        df,
                        idx,
                        scoring_team,
                    )
                    if turnover is not None:
                        turnover_idx, turnover_detail = turnover
                        selected.append(turnover_idx)
                        origin = "Opponent turnover"
                        origin_detail = turnover_detail
                        origin_index = turnover_idx
                        break

                origin = POSSESSION_START_TYPES[event_type]
                origin_detail = None
                origin_index = idx
                break

            if event_type == "Tackle" and _is_successful(event):
                selected.append(idx)
                origin = "Tackle won"
                origin_detail = None
                origin_index = idx
                break

            # Previous shots do not close the possession here. This is the
            # key difference from shot-sequence contribution logic and keeps
            # saved-shot / post rebounds attached to the same goal origin.
            if event_type in SHOT_TYPES:
                selected.append(idx)
                continue

            # A clearly failed own-team action ends the possession unless the
            # raw feed subsequently records an immediate recovery before it;
            # such a recovery would have been encountered first above.
            if (
                event_type in TURNOVER_TYPES
                and event_type != "Error"
                and not _is_successful(event)
            ):
                break

            selected.append(idx)
            continue

        # Opponent events -------------------------------------------------
        if event_type == "Save":
            # Save alone does not prove controlled goalkeeper possession.
            selected.append(idx)
            continue

        if event_type == "Error":
            if _team_had_recent_control(df, idx, event_team):
                selected.append(idx)
                origin = "Opponent turnover"
                origin_detail = "Error"
                origin_index = idx
                break

            # Defensive error during the scoring team's established attack:
            # keep it in the chain as a decisive-context candidate.
            selected.append(idx)
            continue

        if event_type in TURNOVER_TYPES and not _is_successful(event):
            if _team_had_recent_control(df, idx, event_team):
                selected.append(idx)
                origin = "Opponent turnover"
                origin_detail = _turnover_detail(event)
                origin_index = idx
                break

            # Failed defensive duel / deflection during an existing attack.
            continue

        if _is_control_event(event):
            # We have crossed into prior opponent possession without an
            # explicit loss marker. Keep the origin generic but honest.
            origin = "Opponent turnover"
            origin_detail = "Untracked turnover"
            origin_index = idx
            break

        # Blocks, clearances, failed challenges and other non-control opponent
        # contacts do not by themselves terminate the scoring possession.

    selected = tuple(sorted(set(selected)))
    return _TraceResult(
        rows=selected,
        origin=origin,
        origin_detail=origin_detail,
        origin_index=origin_index,
    )


def _elapsed_seconds(start_row, goal_row) -> float | None:
    start = _event_second(start_row)
    end = _event_second(goal_row)
    if start is None or end is None:
        return None
    return max(0.0, float(end) - float(start))


def _is_completed_pass(row) -> bool:
    return str(row.get("type_name", "") or "") == "Pass" and _is_successful(row)


def _is_shot_creating_pass(row) -> bool:
    return _is_completed_pass(row) and (
        _flag(row.get("is_key_pass"))
        or _flag(row.get("is_assist"))
    )


def _is_official_assist(row) -> bool:
    return _is_completed_pass(row) and _flag(row.get("is_assist"))


def _attack_type(origin: str, duration: float | None, pass_count: int) -> str:
    duration = math.inf if duration is None else float(duration)

    if origin == "Penalty":
        return "Penalty"

    if origin in {"Corner", "Free Kick"}:
        if duration <= DIRECT_RESTART_MAX_SECONDS:
            return "Set Piece"
        return "Positional Attack"

    if origin == "Throw-in":
        if (
            duration <= DIRECT_RESTART_MAX_SECONDS
            and pass_count <= DIRECT_RESTART_MAX_PASSES
        ):
            return "Set Piece"
        return "Positional Attack"

    if origin in {
        "Opponent turnover",
        "Ball recovery",
        "Interception",
        "Tackle won",
    }:
        if (
            duration <= TRANSITION_MAX_SECONDS
            and pass_count <= TRANSITION_MAX_PASSES
        ):
            return "Offensive Transition"

    return "Positional Attack"


def _latest_row(rows: pd.DataFrame, mask) -> pd.Series | None:
    candidates = rows[mask]
    if candidates.empty:
        return None
    return candidates.iloc[-1]


def _decisive_mechanism(
    chain: pd.DataFrame,
    goal_row,
    scoring_team: str,
    attack_type: str,
    shot_creating_pass,
):
    goal_time = _event_second(goal_row)

    opponent_errors = chain[
        (chain.get("team_name") != scoring_team)
        & chain.get("type_name").eq("Error")
    ] if "team_name" in chain.columns and "type_name" in chain.columns else pd.DataFrame()

    if not opponent_errors.empty:
        for _, error in opponent_errors.iloc[::-1].iterrows():
            error_time = _event_second(error)
            if goal_time is None or error_time is None or (
                goal_time - error_time <= ERROR_DECISIVE_MAX_SECONDS
            ):
                player = error.get("playerName")
                return (
                    "Defensive error",
                    None if pd.isna(player) else str(player),
                )

    prior_shots = chain[
        (chain.get("team_name") == scoring_team)
        & chain.get("type_name").isin({"Attempt Saved", "Post"})
    ] if "team_name" in chain.columns and "type_name" in chain.columns else pd.DataFrame()

    if not prior_shots.empty:
        previous_shot = prior_shots.iloc[-1]
        shot_time = _event_second(previous_shot)
        if goal_time is None or shot_time is None or (
            goal_time - shot_time <= REBOUND_MAX_SECONDS
        ):
            return "Rebound", None

    if shot_creating_pass is not None:
        if _flag(shot_creating_pass.get("cross")):
            return "Cross", None

        for column in ("Through ball", "Through Ball", "through_ball"):
            if column in shot_creating_pass.index and _flag(
                shot_creating_pass.get(column)
            ):
                return "Through ball", None

    if attack_type == "Offensive Transition":
        return "Fast progression", None

    if attack_type == "Set Piece":
        return "Set-piece delivery", None

    if shot_creating_pass is not None:
        return "Shot-creating pass", None

    return "Combination play", None


def _analysis_route(origin: str, attack_type: str) -> tuple[str, str | None]:
    if attack_type == "Offensive Transition":
        return "Offensive Transition", "offensive-transition"

    if attack_type in {"Set Piece", "Penalty"}:
        return "Set Pieces", "set-piece"

    if origin in {"Goal Kick", "Goalkeeper possession"}:
        return "Build-up", "buildup"

    # The current application has no dedicated full-possession module for a
    # long positional attack that started, for example, from a throw-in. Keep
    # that gap explicit instead of linking the reader to the wrong chart.
    return "Possession development", None


def classify_goal_origins(
    df_processed: pd.DataFrame,
    *,
    home_team: str | None = None,
    away_team: str | None = None,
) -> list[dict]:
    """Return one article-friendly origin classification per goal event."""
    if df_processed is None or df_processed.empty:
        return []

    required = {"team_name", "type_name"}
    if not required.issubset(df_processed.columns):
        return []

    df = _prepare_events(df_processed)

    teams = [
        team
        for team in (home_team, away_team)
        if team not in (None, "")
    ]
    if len(teams) < 2:
        teams = [
            str(value)
            for value in df["team_name"].dropna().unique().tolist()
        ][:2]

    results: list[dict] = []

    goal_indices = df.index[
        df["type_name"].eq("Goal")
    ].tolist()

    for goal_index in goal_indices:
        goal = df.iloc[goal_index]
        event_team = goal.get("team_name")
        own_goal = _flag(goal.get("Own goal"))

        scoring_team = event_team
        if own_goal and len(teams) >= 2 and event_team in teams:
            scoring_team = teams[1] if event_team == teams[0] else teams[0]

        if scoring_team is None or pd.isna(scoring_team):
            continue

        trace = _trace_goal_possession(
            df,
            goal_index,
            str(scoring_team),
        )

        chain = df.iloc[list(trace.rows)].copy()
        chain = chain.sort_index(kind="stable")

        if trace.origin_index is not None:
            start_row = df.iloc[trace.origin_index]
        elif not chain.empty:
            start_row = chain.iloc[0]
        else:
            start_row = goal

        duration = _elapsed_seconds(start_row, goal)

        attacking_chain = chain[
            chain["team_name"].eq(scoring_team)
        ]

        completed_passes = attacking_chain[
            attacking_chain.apply(_is_completed_pass, axis=1)
        ]
        pass_count = int(len(completed_passes))

        assist_row = _latest_row(
            attacking_chain,
            attacking_chain.apply(_is_official_assist, axis=1),
        ) if not attacking_chain.empty else None

        shot_creating_row = _latest_row(
            attacking_chain,
            attacking_chain.apply(_is_shot_creating_pass, axis=1),
        ) if not attacking_chain.empty else None

        attack_type = _attack_type(
            trace.origin,
            duration,
            pass_count,
        )

        decisive_mechanism, decisive_player = _decisive_mechanism(
            chain,
            goal,
            str(scoring_team),
            attack_type,
            shot_creating_row,
        )

        analysis_module, analysis_tab = _analysis_route(
            trace.origin,
            attack_type,
        )

        scorer = goal.get("playerName")
        if scorer is None or pd.isna(scorer):
            scorer = "Unknown"

        if own_goal:
            scorer = f"{scorer} (OG)"

        results.append({
            "goal_event_id": _event_id(goal),
            "team_name": str(scoring_team),
            "scorer": str(scorer),
            "period_id": goal.get("periodId"),
            "minute": goal.get("timeMin"),
            "second": goal.get("timeSec"),
            "possession_origin": trace.origin,
            "origin_detail": trace.origin_detail,
            "attack_type": attack_type,
            "decisive_mechanism": decisive_mechanism,
            "decisive_player": decisive_player,
            "possession_duration_seconds": duration,
            "pass_count": pass_count,
            "action_count": int(len(chain)),
            "official_assist": (
                None
                if assist_row is None
                else str(assist_row.get("playerName"))
            ),
            "shot_creating_pass": (
                None
                if shot_creating_row is None
                else str(shot_creating_row.get("playerName"))
            ),
            "analysis_module": analysis_module,
            "analysis_tab": analysis_tab,
            "sequence_event_ids": [
                _event_id(row)
                for _, row in chain.iterrows()
                if _event_id(row) is not None
            ],
        })

    return results
