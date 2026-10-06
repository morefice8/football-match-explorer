from __future__ import annotations

import math

import numpy as np

from src.metrics.sequence_outcome_metrics import calculate_sequence_milestones
from src.visualization.coordinate_contract import orient_point


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def _event_seconds(row):
    minute = None

    for key in ("expandedMinute", "timeMin", "minute"):
        minute = _number(row.get(key))
        if minute is not None:
            break

    if minute is None:
        return None

    second = _number(row.get("timeSec", 0))
    if second is None:
        second = 0.0

    return minute * 60.0 + second


def sequence_duration_seconds(sequence):
    if sequence is None or sequence.empty:
        return None

    first = sequence.iloc[0]
    last = sequence.iloc[-1]

    for key in ("sequence_duration_seconds", "duration_seconds"):
        value = _number(last.get(key))
        if value is None:
            value = _number(first.get(key))
        if value is not None and value >= 0:
            return value

    start_min = _number(first.get("timeMin_at_loss"))
    start_sec = _number(first.get("timeSec_at_loss"))

    if start_min is not None:
        start = start_min * 60.0 + (start_sec or 0.0)
    else:
        start = _event_seconds(first)

    end = _event_seconds(last)

    if start is None or end is None:
        return None

    duration = end - start
    if duration < 0:
        return None

    return duration


def transition_kpis(sequences, *, transition_team_is_away=False):
    """Summarize a team's transition sequences.

    Final-third/penalty-area/shot rates are computed by
    calculate_sequence_milestones -- the same "controlled location" rule
    the live app's sequence-comparison panel uses (every event's start
    counts as controlled; an end location only counts when that event was
    successful). This used to be a separate, more lenient implementation
    here (any x/end_x/shot_end_x counted regardless of outcome, and a
    different penalty-area line), which silently disagreed with the app
    for the same match under the same KPI labels.
    """
    del transition_team_is_away  # Match Analysis coordinates are already
    # team-relative (x=0 own goal, x=100 opponent goal) per
    # coordinate_contract.py; orient_point is a no-op, so this flag never
    # changed anything. Kept for call-site compatibility.

    clean = [
        sequence
        for sequence in (sequences or [])
        if sequence is not None and not sequence.empty
    ]

    total = len(clean)

    durations = [
        duration
        for duration in (
            sequence_duration_seconds(sequence)
            for sequence in clean
        )
        if duration is not None
    ]

    median_duration = (
        float(np.median(durations))
        if durations
        else None
    )

    milestones = [calculate_sequence_milestones(sequence) for sequence in clean]

    reached_final_third = sum(
        1 for milestone in milestones if milestone["reached_final_third"]
    )
    reached_penalty_area = sum(
        1 for milestone in milestones if milestone["entered_penalty_area"]
    )
    ended_in_shot = sum(
        1 for milestone in milestones if milestone["produced_shot"]
    )
    reached_or_shot = sum(
        1
        for milestone in milestones
        if milestone["produced_shot"] or milestone["reached_final_third"]
    )

    def percentage(count):
        return count / total * 100.0 if total else 0.0

    return {
        "transition_count": total,
        "median_duration_seconds": median_duration,
        "final_third_pct": percentage(reached_final_third),
        "penalty_area_pct": percentage(reached_penalty_area),
        "shot_pct": percentage(ended_in_shot),
        # Backward-compatible KPI used by the interactive transition panel.
        "final_third_or_shot_pct": percentage(reached_or_shot),
    }


def _raw_location(sequence, location_kind):
    if sequence is None or sequence.empty:
        return None

    first = sequence.iloc[0]

    if location_kind == "loss":
        x = _number(first.get("loss_x"))
        y = _number(first.get("loss_y"))
    elif location_kind == "recovery":
        x = _number(first.get("x"))
        y = _number(first.get("y"))
    else:
        raise ValueError(
            "location_kind must be 'loss' or 'recovery'"
        )

    if x is None or y is None:
        return None

    return x, y


def sequence_display_location(
    sequence,
    *,
    location_kind,
    is_away=False,
    loss_to_transition_frame=False,
):
    location = _raw_location(
        sequence,
        location_kind,
    )
    if location is None:
        return None

    x, y = location

    # Processed Match Analysis coordinates are already team-relative.
    # loss_x/loss_y on defensive-transition sequences belong to the
    # team that LOST possession, while the Sequence Explorer is drawn
    # in the opponent/transition-team frame. Rotate that metadata once
    # so the heatmap cell and the turnover marker share one frame.
    if (
        location_kind == "loss"
        and loss_to_transition_frame
    ):
        x = 100.0 - x
        y = 100.0 - y

    x, y = orient_point(
        x,
        y,
        is_away=is_away,
    )
    return float(x), float(y)


def cell_from_click(click_data):
    if not click_data:
        return None

    points = click_data.get("points") or []
    if not points:
        return None

    custom = points[0].get("customdata")

    if isinstance(custom, (list, tuple)) and len(custom) >= 4:
        values = [_number(custom[index]) for index in range(4)]
        if all(value is not None for value in values):
            return {
                "x0": float(values[0]),
                "x1": float(values[1]),
                "y0": float(values[2]),
                "y1": float(values[3]),
            }

    return None


def filter_sequences_by_cell(
    sequences,
    cell,
    *,
    location_kind,
    is_away=False,
    loss_to_transition_frame=False,
):
    if not cell:
        return list(sequences or [])

    x0 = float(cell["x0"])
    x1 = float(cell["x1"])
    y0 = float(cell["y0"])
    y1 = float(cell["y1"])

    selected = []

    for sequence in (sequences or []):
        location = sequence_display_location(
            sequence,
            location_kind=location_kind,
            is_away=is_away,
            loss_to_transition_frame=
                loss_to_transition_frame,
        )
        if location is None:
            continue

        x, y = location

        in_x = x0 <= x < x1 or (
            math.isclose(x1, 100.0) and x0 <= x <= x1
        )
        in_y = y0 <= y < y1 or (
            math.isclose(y1, 100.0) and y0 <= y <= y1
        )

        if in_x and in_y:
            selected.append(sequence)

    return selected


def format_cell_label(cell):
    if not cell:
        return None

    return (
        "Selected cell · "
        f"x {cell['x0']:.0f}–{cell['x1']:.0f} · "
        f"y {cell['y0']:.0f}–{cell['y1']:.0f}"
    )
