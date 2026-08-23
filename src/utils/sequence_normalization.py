"""Shared normalization contract for Match Analysis sequence explorers.

Existing detectors keep their current DataFrame contracts and analytical
semantics. This module adapts those legacy DataFrames into one stable
presentation object consumed by the shared Sequence Explorer.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


SEQUENCE_TYPES = frozenset({
    "buildup",
    "defensive_transition",
    "offensive_transition",
    "set_piece",
})

SHOT_TYPES = frozenset({
    "Goal",
    "Miss",
    "Attempt Saved",
    "Post",
})

RECOVERY_TYPES = frozenset({
    "Ball recovery",
    "Tackle",
    "Interception",
})

TURNOVER_TYPES = frozenset({
    "Dispossessed",
    "Error",
    "Challenge",
})

DEFAULT_CARRY_MIN_DISTANCE = 0.5


def _is_missing(value: Any) -> bool:
    if value is None:
        return True

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _first_value(row, *names, default=None):
    for name in names:
        value = row.get(name)
        if not _is_missing(value):
            return value
    return default


def _number(value):
    if _is_missing(value):
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result):
        return None

    return result


def _event_second(row):
    explicit = _number(
        _first_value(
            row,
            "total_seconds",
            "event_seconds",
        )
    )
    if explicit is not None:
        return explicit

    minute = _number(
        _first_value(
            row,
            "timeMin",
            "minute",
        )
    )
    second = _number(
        _first_value(
            row,
            "timeSec",
            "second",
            default=0,
        )
    )

    if minute is None:
        return None

    return minute * 60.0 + (second or 0.0)


def _metadata_time(first_row, sequence_type):
    if sequence_type == "buildup":
        minute = _number(
            _first_value(
                first_row,
                "timeMin_at_active_start",
                "timeMin_at_trigger",
                "timeMin",
            )
        )
        second = _number(
            _first_value(
                first_row,
                "timeSec_at_active_start",
                "timeSec_at_trigger",
                "timeSec",
                default=0,
            )
        )

    elif sequence_type == "set_piece":
        minute = _number(
            _first_value(
                first_row,
                "timeMin_at_trigger",
                "timeMin",
            )
        )
        second = _number(
            _first_value(
                first_row,
                "timeSec_at_trigger",
                "timeSec",
                default=0,
            )
        )

    else:
        minute = _number(
            _first_value(
                first_row,
                "timeMin_at_loss",
                "timeMin",
            )
        )
        second = _number(
            _first_value(
                first_row,
                "timeSec_at_loss",
                "timeSec",
                default=0,
            )
        )

    if minute is None:
        return None

    return minute * 60.0 + (second or 0.0)


def _sequence_id(first_row, sequence_type):
    if sequence_type in {
        "defensive_transition",
        "offensive_transition",
    }:
        names = (
            "loss_sequence_id",
            "trigger_sequence_id",
            "sequence_id",
            "eventId",
            "id",
        )
    else:
        names = (
            "trigger_sequence_id",
            "sequence_id",
            "loss_sequence_id",
            "eventId",
            "id",
        )

    return _first_value(
        first_row,
        *names,
        default=None,
    )


def _trigger_label(first_row, sequence_type):
    if sequence_type == "set_piece":
        return str(
            _first_value(
                first_row,
                "restart_type",
                "type_of_initial_trigger",
                "type_name",
                default="Restart",
            )
        )

    if sequence_type == "buildup":
        return str(
            _first_value(
                first_row,
                "type_of_initial_trigger",
                "restart_type",
                "type_name",
                default="Build-up start",
            )
        )

    return str(
        _first_value(
            first_row,
            "type_of_initial_loss",
            "type_of_initial_trigger",
            "type_name",
            default="Turnover",
        )
    )


def _display_outcome(last_row, sequence_type):
    if sequence_type == "set_piece":
        value = _first_value(
            last_row,
            "restart_execution_outcome",
            "sequence_outcome_type",
            "terminal_outcome",
            "outcome",
            default="Unknown",
        )
    else:
        value = _first_value(
            last_row,
            "sequence_outcome_type",
            "terminal_outcome",
            "outcome",
            default="Unknown",
        )

    return str(value)


def _duration_seconds(
    first_row,
    last_row,
    sequence_type,
    start_second,
):
    if sequence_type == "buildup":
        canonical = _number(
            first_row.get(
                "buildup_active_duration_seconds"
            )
        )
        if canonical is not None:
            return max(0.0, canonical)

    end_second = _event_second(
        last_row
    )

    if (
        start_second is None
        or end_second is None
    ):
        return None

    duration = end_second - start_second

    if duration < 0:
        return None

    if (
        sequence_type == "set_piece"
        and duration == 0
    ):
        return None

    return duration


def _event_type(
    raw_type,
    *,
    sequence_type,
    is_first,
):
    if (
        sequence_type == "set_piece"
        and is_first
    ):
        return "restart"

    if raw_type in {
        "Pass",
        "Offside Pass",
    }:
        return "pass"

    if raw_type in SHOT_TYPES:
        return "shot"

    # A transition contains one possession-change trigger. Recovery,
    # tackle/interception and later duel events remain in the chronology
    # but must not inflate the Turnovers KPI.
    if (
        sequence_type in {
            "defensive_transition",
            "offensive_transition",
        }
        and is_first
        and (
            raw_type in RECOVERY_TYPES
            or raw_type in TURNOVER_TYPES
        )
    ):
        return "turnover"

    if (
        raw_type in RECOVERY_TYPES
        or raw_type in TURNOVER_TYPES
    ):
        return "other"

    return "other"


def _event_outcome(row):
    value = _first_value(
        row,
        "outcome",
        "restart_execution_outcome",
        "sequence_outcome_type",
        default=None,
    )

    if _is_missing(value):
        return None

    return str(value)


def _is_successful(row):
    outcome = str(
        _first_value(
            row,
            "outcome",
            default="",
        )
    ).strip().lower()

    if not outcome:
        return None

    return outcome in {
        "successful",
        "complete",
        "completed",
        "success",
        "won",
    }


def _distance(x0, y0, x1, y1):
    values = (
        _number(x0),
        _number(y0),
        _number(x1),
        _number(y1),
    )

    if any(
        value is None
        for value in values
    ):
        return None

    return math.hypot(
        values[2] - values[0],
        values[3] - values[1],
    )


def _row_metadata(row):
    keys = (
        "Mapped Jersey Number",
        "receiver",
        "receiver_jersey_number",
        "restart_type",
        "restart_delivery_type",
        "restart_execution_outcome",
        "restart_length_m",
        "terminal_outcome",
        "termination_reason",
        "viewpoint",
        "type_of_initial_trigger",
        "type_of_initial_loss",
        "loss_zone",
        "trigger_zone",
    )

    return {
        key: row.get(key)
        for key in keys
        if (
            key in row.index
            and not _is_missing(
                row.get(key)
            )
        )
    }


def _transition_trigger_event(
    first_row,
    sequence_type,
    team_name,
):
    if sequence_type not in {
        "defensive_transition",
        "offensive_transition",
    }:
        return None

    x = _number(
        _first_value(
            first_row,
            "loss_x",
            "x_at_loss",
            "x_of_loss",
            "turnover_x",
            "x",
            default=None,
        )
    )
    y = _number(
        _first_value(
            first_row,
            "loss_y",
            "y_at_loss",
            "y_of_loss",
            "turnover_y",
            "y",
            default=None,
        )
    )

    trigger = _trigger_label(
        first_row,
        sequence_type,
    )

    return {
        "event_id":
            "transition-turnover-trigger",
        "event_type": "turnover",
        "raw_event_type": trigger,
        "team_name": team_name,
        "player_name": _first_value(
            first_row,
            "playerName",
            "player_name",
            default=None,
        ),
        "jersey_number":
            _jersey_number(first_row),
        "second": _metadata_time(
            first_row,
            sequence_type,
        ),
        "x": x,
        "y": y,
        "end_x": None,
        "end_y": None,
        "outcome":
            "Possession change",
        "successful": None,
        "metadata": {
            "synthetic_trigger": True,
            "trigger": trigger,
        },
    }


def _jersey_number(row):
    value = _first_value(
        row,
        "Mapped Jersey Number",
        "jersey_number",
        "jerseyNumber",
        "shirtNumber",
        "shirt_number",
        default=None,
    )

    if _is_missing(value):
        return None

    number = _number(value)

    if number is not None:
        if float(number).is_integer():
            return int(number)
        return number

    text = str(value).strip()
    return text or None


def normalize_sequence(
    sequence_df,
    *,
    sequence_type,
    team_name=None,
    opponent_name=None,
    carry_min_distance=DEFAULT_CARRY_MIN_DISTANCE,
):
    """Adapt a legacy sequence DataFrame to the shared presentation contract."""

    if sequence_type not in SEQUENCE_TYPES:
        raise ValueError(
            "sequence_type must be one of: "
            + ", ".join(
                sorted(SEQUENCE_TYPES)
            )
        )

    if (
        sequence_df is None
        or not isinstance(
            sequence_df,
            pd.DataFrame,
        )
        or sequence_df.empty
    ):
        return {
            "sequence_id": None,
            "sequence_type": sequence_type,
            "team_name": team_name,
            "opponent_name": opponent_name,
            "period_id": None,
            "start_second": None,
            "end_second": None,
            "duration_seconds": None,
            "outcome": "Unknown",
            "terminal_outcome": None,
            "termination_reason": None,
            "viewpoint": None,
            "trigger": None,
            "events": [],
        }

    df = sequence_df.copy().reset_index(
        drop=True
    )

    first_row = df.iloc[0]
    last_row = df.iloc[-1]

    if team_name is None:
        team_name = _first_value(
            first_row,
            "team_name",
            default=None,
        )

    start_second = _metadata_time(
        first_row,
        sequence_type,
    )

    end_second = _event_second(
        last_row
    )

    period_id = _first_value(
        first_row,
        "periodId",
        "period_id",
        default=None,
    )

    normalized_events = []
    controlled_x = None
    controlled_y = None

    for index, row in df.iterrows():
        raw_type = str(
            _first_value(
                row,
                "type_name",
                "event_type",
                default="Unknown",
            )
        )

        x = _number(
            row.get("x")
        )
        y = _number(
            row.get("y")
        )
        end_x = _number(
            row.get("end_x")
        )
        end_y = _number(
            row.get("end_y")
        )

        player_name = _first_value(
            row,
            "playerName",
            "player_name",
            default=None,
        )

        event_second = _event_second(
            row
        )

        event_kind = _event_type(
            raw_type,
            sequence_type=sequence_type,
            is_first=(index == 0),
        )

        if (
            controlled_x is not None
            and controlled_y is not None
            and x is not None
            and y is not None
        ):
            carry_distance = _distance(
                controlled_x,
                controlled_y,
                x,
                y,
            )

            if (
                carry_distance is not None
                and carry_distance
                >= float(carry_min_distance)
            ):
                normalized_events.append({
                    "event_id":
                        f"carry-before-{index}",
                    "event_type": "carry",
                    "raw_event_type":
                        "Inferred carry",
                    "team_name": team_name,
                    "player_name":
                        player_name,
                    "jersey_number":
                        _jersey_number(row),
                    "second":
                        event_second,
                    "x": controlled_x,
                    "y": controlled_y,
                    "end_x": x,
                    "end_y": y,
                    "outcome":
                        "Controlled movement",
                    "successful": True,
                    "metadata": {
                        "inferred": True,
                        "distance_coordinate_units":
                            carry_distance,
                    },
                })

        successful = _is_successful(
            row
        )

        normalized_events.append({
            "event_id": _first_value(
                row,
                "eventId",
                "id",
                default=index,
            ),
            "event_type": event_kind,
            "raw_event_type": raw_type,
            "team_name": _first_value(
                row,
                "team_name",
                default=team_name,
            ),
            "player_name": player_name,
            "jersey_number":
                _jersey_number(row),
            "second": event_second,
            "x": x,
            "y": y,
            "end_x": end_x,
            "end_y": end_y,
            "outcome":
                _event_outcome(row),
            "successful": successful,
            "metadata":
                _row_metadata(row),
        })

        if (
            event_kind in {
                "pass",
                "restart",
            }
            and end_x is not None
            and end_y is not None
            and successful is not False
        ):
            controlled_x = end_x
            controlled_y = end_y

        elif (
            raw_type in RECOVERY_TYPES
            and successful is not False
            and x is not None
            and y is not None
        ):
            # Recovery/control can seed a following inferred carry even when
            # the event is not counted as a separate turnover.
            controlled_x = x
            controlled_y = y

        else:
            controlled_x = None
            controlled_y = None

    if sequence_type in {
        "defensive_transition",
        "offensive_transition",
    }:
        turnover_indexes = [
            index
            for index, event
            in enumerate(normalized_events)
            if event.get("event_type")
            == "turnover"
        ]

        if not turnover_indexes:
            trigger_event = (
                _transition_trigger_event(
                    first_row,
                    sequence_type,
                    team_name,
                )
            )

            if trigger_event is not None:
                normalized_events.insert(
                    0,
                    trigger_event,
                )

        elif len(turnover_indexes) > 1:
            # Defensive guard: the normalized presentation contract allows
            # exactly one possession-change trigger per transition.
            for index in turnover_indexes[1:]:
                normalized_events[index][
                    "event_type"
                ] = "other"

    terminal_outcome = _first_value(
        last_row,
        "terminal_outcome",
        default=None,
    )

    termination_reason = _first_value(
        last_row,
        "termination_reason",
        default=None,
    )

    viewpoint = _first_value(
        last_row,
        "viewpoint",
        default=None,
    )

    return {
        "sequence_id":
            _sequence_id(
                first_row,
                sequence_type,
            ),
        "sequence_type": sequence_type,
        "team_name": team_name,
        "opponent_name": opponent_name,
        "period_id": period_id,
        "start_second": start_second,
        "end_second": end_second,
        "duration_seconds":
            _duration_seconds(
                first_row,
                last_row,
                sequence_type,
                start_second,
            ),
        "outcome":
            _display_outcome(
                last_row,
                sequence_type,
            ),
        "terminal_outcome":
            None
            if _is_missing(terminal_outcome)
            else str(terminal_outcome),
        "termination_reason":
            None
            if _is_missing(
                termination_reason
            )
            else str(
                termination_reason
            ),
        "viewpoint":
            None
            if _is_missing(viewpoint)
            else str(viewpoint),
        "trigger":
            _trigger_label(
                first_row,
                sequence_type,
            ),
        "events":
            normalized_events,
    }
