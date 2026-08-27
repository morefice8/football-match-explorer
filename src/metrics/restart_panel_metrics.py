from __future__ import annotations

import math
from collections import Counter

import pandas as pd


FILTER_SPECS = (
    ("action", "Restart type", "restart_type"),
    ("side", "Side", "side"),
    ("delivery", "Delivery", "delivery"),
    ("destination", "Destination", "destination"),
    ("outcome", "Execution", "outcome"),
    ("development", "Development", "development_outcome"),
)


def _clean(value, fallback="Unknown"):
    if value is None:
        return fallback

    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    return text or fallback


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def _sequence_lookup(sequences):
    lookup = {}

    for sequence in sequences or []:
        if (
            sequence is None
            or sequence.empty
        ):
            continue

        first = sequence.iloc[0]
        sequence_id = first.get(
            "trigger_sequence_id"
        )

        if sequence_id is None:
            continue

        lookup[
            str(sequence_id)
        ] = sequence

    return lookup


def _spatial_destination(
    end_x,
    end_y,
):
    """Presentation-only fallback for missing restart destination labels."""
    x = _number(end_x)
    y = _number(end_y)

    if x is None:
        return "Unknown"

    if (
        x >= 83.5
        and y is not None
        and 21.1 <= y <= 78.9
    ):
        return "Penalty Area"

    if x >= 66.67:
        return "Final Third"

    if x >= 33.33:
        return "Middle Third"

    return "Own Third"


def build_restart_records(
    df_analyzed,
    sequences,
):
    """
    Build the PLOT-13 presentation model.

    Type, delivery and execution outcome prefer REL-09 canonical restart
    metadata from the actual restart delivery. Side and destination reuse
    the already-active set-piece analysis adapter because those are
    presentation classifications, not restart-detection semantics.
    """
    if (
        df_analyzed is None
        or df_analyzed.empty
    ):
        return []

    sequence_lookup = (
        _sequence_lookup(
            sequences
        )
    )

    records = []

    for index, row in (
        df_analyzed
        .reset_index(drop=True)
        .iterrows()
    ):
        sequence_id = _clean(
            row.get(
                "sequence_id"
            ),
            fallback=f"restart-row-{index}",
        )

        sequence = sequence_lookup.get(
            sequence_id
        )

        first = (
            sequence.iloc[0]
            if (
                sequence is not None
                and not sequence.empty
            )
            else row
        )

        raw_type = _clean(
            first.get(
                "restart_type"
            ),
            fallback=_clean(
                row.get(
                    "Action Type"
                )
            ),
        )

        if (
            raw_type == "Unknown"
            and _clean(
                row.get(
                    "Action Type"
                )
            ).lower()
            == "penalty"
        ):
            raw_type = "Penalty"

        delivery = _clean(
            first.get(
                "restart_delivery_type"
            ),
            fallback=_clean(
                row.get(
                    "Delivery"
                )
            ),
        )

        outcome = _clean(
            first.get(
                "restart_execution_outcome"
            ),
            fallback=_clean(
                row.get(
                    "Outcome"
                )
            ),
        )

        development_outcome = _clean(
            first.get(
                "restart_development_outcome"
            ),
            fallback=outcome,
        )

        start_x = _number(
            row.get(
                "x_start",
                first.get("x"),
            )
        )
        start_y = _number(
            row.get(
                "y_start",
                first.get("y"),
            )
        )
        end_x = _number(
            row.get(
                "x_end",
                first.get("end_x"),
            )
        )
        end_y = _number(
            row.get(
                "y_end",
                first.get("end_y"),
            )
        )

        minute = _number(
            first.get(
                "timeMin_at_trigger",
                first.get("timeMin"),
            )
        )
        second = _number(
            first.get(
                "timeSec_at_trigger",
                first.get("timeSec"),
            )
        )

        match_second = (
            minute * 60.0
            + (
                second
                if second
                is not None
                else 0.0
            )
            if minute is not None
            else None
        )

        jersey = first.get(
            "Mapped Jersey Number"
        )

        try:
            if pd.isna(jersey):
                jersey = None
        except (TypeError, ValueError):
            pass

        legacy_destination = _clean(
            row.get(
                "Destination"
            ),
            fallback="N/A",
        )

        destination = (
            legacy_destination
            if legacy_destination
            not in {
                "N/A",
                "Unknown",
                "None",
                "",
            }
            else _spatial_destination(
                end_x,
                end_y,
            )
        )

        records.append(
            {
                "sequence_id":
                    sequence_id,
                "restart_type":
                    raw_type,
                "side":
                    _clean(
                        row.get("Side")
                    ),
                "delivery":
                    delivery,
                "destination":
                    destination,
                "outcome":
                    outcome,
                "development_outcome":
                    development_outcome,
                "player_name":
                    _clean(
                        row.get(
                            "Player",
                            first.get(
                                "playerName"
                            ),
                        ),
                        fallback="Unknown player",
                    ),
                "jersey_number":
                    jersey,
                "start_x":
                    start_x,
                "start_y":
                    start_y,
                "end_x":
                    end_x,
                "end_y":
                    end_y,
                "match_second":
                    match_second,
            }
        )

    return records


def apply_restart_filters(
    records,
    active_filter,
):
    active_filter = (
        active_filter
        or {}
    )

    field_map = {
        filter_key:
            field_name
        for (
            filter_key,
            _,
            field_name,
        ) in FILTER_SPECS
    }

    filtered = []

    for record in records or []:
        keep = True

        for (
            filter_key,
            selected_value,
        ) in active_filter.items():
            field_name = field_map.get(
                filter_key
            )

            if (
                field_name is None
                or selected_value
                in (
                    None,
                    "",
                )
            ):
                continue

            if (
                str(
                    record.get(
                        field_name,
                        "Unknown",
                    )
                )
                != str(
                    selected_value
                )
            ):
                keep = False
                break

        if keep:
            filtered.append(
                record
            )

    return filtered


def option_counts(
    records,
    active_filter,
    filter_key,
):
    """
    Contextual option counts: apply every active filter except the filter
    currently being rendered, so users can switch values without having to
    reset the whole panel first.
    """
    active_filter = dict(
        active_filter
        or {}
    )

    active_filter.pop(
        filter_key,
        None,
    )

    contextual = (
        apply_restart_filters(
            records,
            active_filter,
        )
    )

    field_name = next(
        (
            field_name
            for (
                key,
                _,
                field_name,
            ) in FILTER_SPECS
            if key == filter_key
        ),
        None,
    )

    if field_name is None:
        return {}

    counts = Counter(
        str(
            record.get(
                field_name,
                "Unknown",
            )
        )
        for record
        in contextual
    )

    return dict(
        sorted(
            counts.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )
    )


def sequence_json_lookup(
    sequences,
):
    result = {}

    for sequence in sequences or []:
        if (
            sequence is None
            or sequence.empty
        ):
            continue

        sequence_id = sequence.iloc[0].get(
            "trigger_sequence_id"
        )

        if sequence_id is None:
            continue

        result[
            str(sequence_id)
        ] = sequence.to_json(
            orient="split"
        )

    return result


def record_lookup(
    records,
):
    return {
        str(
            record[
                "sequence_id"
            ]
        ):
            record
        for record
        in records
    }
