"""Pure helpers for exact sequence filtering and safe carousel state."""


def filter_sequences_exact(
    sequences,
    active_filter,
    value_getters,
):
    """
    Return non-empty sequences matching every active filter.

    No implicit fallback is ever applied. If the selected combination
    matches nothing, the function returns an empty list.

    Unknown filter keys fail closed instead of silently broadening the
    result set.
    """

    valid_sequences = [
        seq
        for seq in (sequences or [])
        if (
            seq is not None
            and not getattr(
                seq,
                "empty",
                False,
            )
        )
    ]

    if not active_filter:
        return valid_sequences

    filtered_sequences = []

    for seq in valid_sequences:
        matches = True

        for (
            filter_key,
            expected_value,
        ) in active_filter.items():
            getter = value_getters.get(
                filter_key
            )

            if getter is None:
                matches = False
                break

            try:
                actual_value = getter(
                    seq
                )
            except (
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ):
                matches = False
                break

            if (
                str(actual_value).strip()
                != str(expected_value).strip()
            ):
                matches = False
                break

        if matches:
            filtered_sequences.append(
                seq
            )

    return filtered_sequences


def make_carousel_controller(
    total_items,
    active_index=0,
):
    """
    Build a safe carousel state.

    Zero items is a valid state and always maps to index zero.
    """

    total_items = max(
        int(total_items or 0),
        0,
    )

    if total_items == 0:
        return {
            "active_index": 0,
            "total_items": 0,
        }

    active_index = int(
        active_index or 0
    )

    active_index = min(
        max(active_index, 0),
        total_items - 1,
    )

    return {
        "active_index": active_index,
        "total_items": total_items,
    }


def step_carousel(
    controller_data,
    delta,
):
    """
    Move a carousel without ever performing modulo by zero.

    A zero-item carousel remains in the canonical zero state.
    """

    if not controller_data:
        return controller_data

    total_items = max(
        int(
            controller_data.get(
                "total_items",
                0,
            )
            or 0
        ),
        0,
    )

    if total_items == 0:
        return {
            "active_index": 0,
            "total_items": 0,
        }

    active_index = int(
        controller_data.get(
            "active_index",
            0,
        )
        or 0
    )

    active_index %= total_items

    return {
        "active_index": (
            active_index
            + int(delta)
        ) % total_items,
        "total_items": total_items,
    }
