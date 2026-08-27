"""Canonical restart execution semantics for REL-09B.

Restart Analysis keeps one canonical question at sequence level:
    How was play restarted?

Each restart sequence therefore still stops at the delivery itself. A separate
short development annotation records whether that delivery immediately produced
a shot/goal or lost possession, without appending later events to the canonical
restart sequence. Longer possession development still belongs to Buildup / goal
origin analysis.
"""

from __future__ import annotations

import math

import pandas as pd

from src.utils.derived_cache import cache_derived_result
from src.utils.sequence_outcomes import apply_sequence_outcome_contract


RESTART_FLAG_ALIASES = {
    "Corner": (
        "Corner taken",
        "Corner",
    ),
    "Free Kick": (
        "Free kick taken",
        "Freekick taken",
        "Free Kick taken",
        "Free kick",
        "Freekick",
    ),
    "Throw-in": (
        "ThrowIn",
        "Throw in",
        "Throw-in",
        "Throw-in taken",
    ),
    "Goal Kick": (
        "Goal kick",
        "Goal kick taken",
        "Goal Kick",
        "Goal Kick taken",
    ),
}

SHOT_TYPES = frozenset({
    "Goal",
    "Miss",
    "Attempt Saved",
    "Post",
})

# Restart Analysis remains execution-first: every canonical restart sequence
# still contains only the delivery event. This short secondary window annotates
# what the attack produced immediately afterwards, so a corner -> saved shot ->
# rebound goal can be recognised without turning the restart sequence itself
# into a full possession chain.
RESTART_DEVELOPMENT_WINDOW_SECONDS = 12.0

CONTROL_GAIN_TYPES = frozenset({
    "Pass",
    "Ball recovery",
    "Interception",
    "Tackle",
    "Keeper pick-up",
    "Claim",
})

POSSESSION_LOSS_TYPES = frozenset({
    "Pass",
    "Take On",
    "Dispossessed",
    "Offside Pass",
    "Out",
})

DEVELOPMENT_STOP_TYPES = frozenset({
    "Foul",
    "Offside Pass",
    "Out",
    "Corner Awarded",
})


def _flag_is_true(value):
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _row_has_any_flag(row, aliases):
    return any(
        alias in row.index and _flag_is_true(row.get(alias))
        for alias in aliases
    )


def classify_restart_event(row):
    """Return the canonical restart type for the ACTUAL delivery event."""
    event_type = str(row.get("type_name", "") or "")

    if event_type == "Pass":
        for restart_type in ("Corner", "Throw-in", "Goal Kick", "Free Kick"):
            if _row_has_any_flag(row, RESTART_FLAG_ALIASES[restart_type]):
                return restart_type

    # Direct free-kick shots are not Pass events.
    if event_type in SHOT_TYPES and _row_has_any_flag(
        row,
        RESTART_FLAG_ALIASES["Free Kick"],
    ):
        return "Free Kick"

    return None


def _event_length_m(row):
    """Return delivery length in metres, preferring Opta Length when present."""
    length = pd.to_numeric(
        pd.Series([row.get("Length")]),
        errors="coerce",
    ).iloc[0]

    if pd.notna(length):
        return float(length)

    x = pd.to_numeric(pd.Series([row.get("x")]), errors="coerce").iloc[0]
    y = pd.to_numeric(pd.Series([row.get("y")]), errors="coerce").iloc[0]
    end_x = pd.to_numeric(
        pd.Series([row.get("end_x")]),
        errors="coerce",
    ).iloc[0]
    end_y = pd.to_numeric(
        pd.Series([row.get("end_y")]),
        errors="coerce",
    ).iloc[0]

    if any(pd.isna(v) for v in (x, y, end_x, end_y)):
        return float("nan")

    dx_m = (float(end_x) - float(x)) * 1.05
    dy_m = (float(end_y) - float(y)) * 0.68
    return math.hypot(dx_m, dy_m)


def _distance_bucket(length_m):
    if pd.isna(length_m):
        return "Unknown"
    if float(length_m) < 15.0:
        return "Short"
    if float(length_m) < 30.0:
        return "Medium"
    return "Long"


def classify_restart_delivery(row, restart_type=None):
    """Describe HOW the restart was executed, not WHAT restart it was."""
    restart_type = restart_type or classify_restart_event(row)
    if restart_type is None:
        return None

    event_type = str(row.get("type_name", "") or "")

    if event_type in SHOT_TYPES:
        return "Direct Shot"

    is_cross = _flag_is_true(row.get("cross"))

    if restart_type == "Corner":
        return "Direct Cross" if is_cross else "Short Corner"

    if restart_type == "Free Kick":
        if is_cross:
            return "Direct Cross"

        bucket = _distance_bucket(_event_length_m(row))
        if bucket == "Long":
            return "Long Pass"
        if bucket in {"Short", "Medium"}:
            return "Short Pass"
        return "Pass"

    if restart_type == "Throw-in":
        bucket = _distance_bucket(_event_length_m(row))
        return f"{bucket} Throw" if bucket != "Unknown" else "Unknown Throw"

    if restart_type == "Goal Kick":
        bucket = _distance_bucket(_event_length_m(row))
        return f"{bucket} Goal Kick" if bucket != "Unknown" else "Unknown Goal Kick"

    return restart_type


def _execution_outcome(row):
    event_type = str(row.get("type_name", "") or "")

    if event_type == "Goal":
        return "Goal"

    if event_type in SHOT_TYPES:
        return "Shot"

    if event_type == "Pass":
        if str(row.get("outcome", "")).lower() == "successful":
            return "Successful Delivery"
        return "Unsuccessful Delivery"

    return "Unknown"


def _legacy_sequence_outcome(row):
    """Compatibility outcome used by shared sequence infrastructure."""
    event_type = str(row.get("type_name", "") or "")

    if event_type == "Goal":
        return "Goals"

    if event_type in SHOT_TYPES:
        return "Shots"

    if event_type == "Pass":
        if str(row.get("outcome", "")).lower() == "successful":
            return "Possession Retained"
        return "Lost Possessions"

    return "Unknown"


def _event_match_second(row):
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


def _development_goal_for_team(row, team_name):
    if str(row.get("type_name", "") or "") != "Goal":
        return False

    event_team = row.get("team_name")
    own_goal = _flag_is_true(row.get("Own goal"))

    if own_goal:
        return event_team != team_name

    return event_team == team_name


def classify_restart_development(
    df,
    restart_index,
    team_name,
    *,
    max_seconds=RESTART_DEVELOPMENT_WINDOW_SECONDS,
):
    """Classify the immediate attacking development after a restart.

    The canonical restart sequence remains delivery-only. This helper scans a
    short, same-period window solely to annotate whether that restart quickly
    produced a shot/goal or whether possession was lost. A goalkeeper Save is
    deliberately not treated as controlled possession because rebounds remain
    live; Keeper pick-up / Claim and successful controlled opponent actions do
    end the development.
    """
    restart = df.iloc[restart_index]
    execution = _execution_outcome(restart)

    if execution == "Goal":
        return {
            "outcome": "Goal",
            "event_id": restart.get("eventId", restart.get("id")),
            "elapsed_seconds": 0.0,
        }

    if execution == "Shot":
        return {
            "outcome": "Shot",
            "event_id": restart.get("eventId", restart.get("id")),
            "elapsed_seconds": 0.0,
        }

    if execution == "Unsuccessful Delivery":
        return {
            "outcome": "Possession Lost",
            "event_id": restart.get("eventId", restart.get("id")),
            "elapsed_seconds": 0.0,
        }

    start_second = _event_match_second(restart)
    start_period = restart.get("periodId")

    best_outcome = "Possession Retained"
    best_event_id = None
    best_elapsed = None

    for future_index in range(restart_index + 1, len(df)):
        event = df.iloc[future_index]

        event_period = event.get("periodId")
        if (
            pd.notna(start_period)
            and pd.notna(event_period)
            and event_period != start_period
        ):
            break

        event_second = _event_match_second(event)
        elapsed = None

        if start_second is not None and event_second is not None:
            elapsed = event_second - start_second
            if elapsed < 0:
                continue
            if elapsed > max_seconds:
                break

        event_type = str(event.get("type_name", "") or "")
        event_team = event.get("team_name")
        successful = str(event.get("outcome", "")).lower() == "successful"

        if classify_restart_event(event) is not None:
            break

        if _development_goal_for_team(event, team_name):
            return {
                "outcome": "Goal",
                "event_id": event.get("eventId", event.get("id")),
                "elapsed_seconds": elapsed,
            }

        if event_team == team_name and event_type in SHOT_TYPES:
            best_outcome = "Shot"
            best_event_id = event.get("eventId", event.get("id"))
            best_elapsed = elapsed
            continue

        # Save does not necessarily establish control; keep rebounds alive.
        if event_type == "Save":
            continue

        if (
            event_team != team_name
            and successful
            and event_type in CONTROL_GAIN_TYPES
        ):
            return {
                "outcome": (
                    "Shot" if best_outcome == "Shot" else "Possession Lost"
                ),
                "event_id": (
                    best_event_id
                    or event.get("eventId", event.get("id"))
                ),
                "elapsed_seconds": (
                    best_elapsed if best_outcome == "Shot" else elapsed
                ),
            }

        if (
            event_team == team_name
            and not successful
            and event_type in POSSESSION_LOSS_TYPES
        ):
            return {
                "outcome": (
                    "Shot" if best_outcome == "Shot" else "Possession Lost"
                ),
                "event_id": (
                    best_event_id
                    or event.get("eventId", event.get("id"))
                ),
                "elapsed_seconds": (
                    best_elapsed if best_outcome == "Shot" else elapsed
                ),
            }

        if event_type in DEVELOPMENT_STOP_TYPES:
            break

    return {
        "outcome": best_outcome,
        "event_id": best_event_id,
        "elapsed_seconds": best_elapsed,
    }


@cache_derived_result("restart_sequences")
def extract_restart_sequences(df_processed, team_name):
    """Return one immediate-execution sequence per actual restart delivery.

    Important REL-09B contract:
    - the restart delivery event is the whole Restart Analysis sequence;
    - later passes/shots/goals are NOT appended to that sequence;
    - immediate development is stored only as restart_development_* metadata;
    - a successful throw-in followed by a goal 31 seconds later is therefore
      still a Successful Delivery with no Goal development attribution.
    """
    if df_processed is None or df_processed.empty:
        return []

    if "team_name" not in df_processed.columns:
        return []

    df = df_processed.copy().reset_index(drop=True)

    sort_cols = [
        col
        for col in ("periodId", "timeMin", "timeSec", "eventId", "id")
        if col in df.columns
    ]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    sequences = []
    sequence_number = 0

    for event_index, event in df.iterrows():
        if event.get("team_name") != team_name:
            continue

        restart_type = classify_restart_event(event)
        if restart_type is None:
            continue

        sequence_number += 1
        action = event.to_dict()

        event_id = action.get("eventId", action.get("id", sequence_number))
        sequence_id = f"restart-{event_id}-{sequence_number}"

        delivery_type = classify_restart_delivery(event, restart_type)
        execution_outcome = _execution_outcome(event)
        sequence_outcome = _legacy_sequence_outcome(event)
        length_m = _event_length_m(event)
        development = classify_restart_development(
            df,
            event_index,
            team_name,
        )

        if execution_outcome == "Successful Delivery":
            termination_reason = "restart_delivery_completed"
        elif execution_outcome == "Unsuccessful Delivery":
            termination_reason = "restart_delivery_failed"
        else:
            termination_reason = "direct_restart_shot"

        action.update({
            "trigger_sequence_id": sequence_id,
            "trigger_zone": "Restart Delivery",
            "triggering_trigger_Opta_id": event_id,
            "timeMin_at_trigger": action.get("timeMin"),
            "timeSec_at_trigger": action.get("timeSec"),
            "type_of_initial_trigger": restart_type,
            "restart_type": restart_type,
            "restart_delivery_type": delivery_type,
            "restart_execution_outcome": execution_outcome,
            "restart_development_outcome": development["outcome"],
            "restart_development_event_id": development["event_id"],
            "restart_development_elapsed_seconds": development[
                "elapsed_seconds"
            ],
            "restart_length_m": length_m,
            "buildup_pass_count": (
                1
                if (
                    action.get("type_name") == "Pass"
                    and str(action.get("outcome", "")).lower() == "successful"
                )
                else 0
            ),
            "sequence_outcome_type": sequence_outcome,
        })

        sequence_df = pd.DataFrame([action])
        sequence_df = apply_sequence_outcome_contract(
            sequence_df,
            viewpoint="attacking",
            legacy_outcome=sequence_outcome,
            termination_reason=termination_reason,
        )
        sequences.append(sequence_df)

    return sequences
