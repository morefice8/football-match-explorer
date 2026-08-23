"""Canonical restart execution semantics for REL-09B.

Restart Analysis answers one narrow question:
    How was play restarted?

It therefore stops at the restart delivery itself. It does not trace the whole
subsequent possession. Longer possession development belongs to Buildup and,
later, goal-origin classification.
"""

from __future__ import annotations

import math

import pandas as pd

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


def extract_restart_sequences(df_processed, team_name):
    """Return one immediate-execution sequence per actual restart delivery.

    Important REL-09B contract:
    - the restart delivery event is the whole Restart Analysis sequence;
    - later passes/shots/goals are NOT appended here;
    - a successful throw-in followed by a goal 31 seconds later is therefore
      a Successful Delivery in Restart Analysis, while its first phase belongs
      to Buildup and its full origin belongs to GOAL-01.
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

    for _, event in df.iterrows():
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
