from __future__ import annotations

import pandas as pd


TACKLE_TYPES = frozenset({
    "tackle",
})
INTERCEPTION_TYPES = frozenset({
    "interception",
})
RECOVERY_TYPES = frozenset({
    "ball recovery",
    "recovery",
})
CLEARANCE_TYPES = frozenset({
    "clearance",
})
BLOCK_TYPES = frozenset({
    "blocked pass",
    "blocked shot",
    "block",
})
FOUL_TYPES = frozenset({
    "foul",
    "foul committed",
})


def _clean_text(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def _truthy(value):
    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return float(value) != 0.0

    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def _truthy_series(frame, column):
    if column not in frame.columns:
        return pd.Series(
            False,
            index=frame.index,
        )

    return frame[column].map(
        _truthy
    )


def _successful_mask(frame):
    outcome = frame.get(
        "outcome",
        pd.Series(
            "",
            index=frame.index,
        ),
    )

    text = (
        outcome
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    numeric = pd.to_numeric(
        outcome,
        errors="coerce",
    )

    return (
        text.eq("successful")
        | numeric.eq(1)
    )


def _event_key_series(frame):
    for column in (
        "id",
        "eventId",
        "event_id",
    ):
        if column in frame.columns:
            return (
                frame[column]
                .fillna("")
                .astype(str)
            )

    return pd.Series(
        [
            f"row-{index}"
            for index in frame.index
        ],
        index=frame.index,
        dtype="object",
    )


def _jersey_label(value):
    try:
        if pd.isna(value):
            return "?"
    except (TypeError, ValueError):
        pass

    try:
        return str(
            int(
                float(value)
            )
        )
    except (TypeError, ValueError):
        return "?"


def _event_type_series(frame):
    return (
        frame["type_name"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


def _event_type_id_series(frame):
    if "typeId" not in frame.columns:
        return pd.Series(
            pd.NA,
            index=frame.index,
            dtype="Float64",
        )

    return pd.to_numeric(
        frame["typeId"],
        errors="coerce",
    )


def player_defensive_profile(
    player_events,
):
    if (
        player_events is None
        or player_events.empty
        or "type_name"
        not in player_events.columns
    ):
        return {
            "unique": 0,
            "tackles_won": 0,
            "tackles_attempted": 0,
            "interceptions": 0,
            "recoveries": 0,
            "clearances": 0,
            "blocks": 0,
            "fouls": 0,
        }

    frame = player_events.copy()

    event_type = _event_type_series(
        frame
    )

    type_id = _event_type_id_series(
        frame
    )

    success = _successful_mask(
        frame
    )

    event_key = _event_key_series(
        frame
    )

    tackle_mask = (
        event_type.isin(
            TACKLE_TYPES
        )
        | type_id.eq(7)
    )

    successful_tackle_mask = (
        tackle_mask
        & success
    )

    interception_mask = (
        event_type.isin(
            INTERCEPTION_TYPES
        )
        | type_id.eq(8)
    )

    recovery_mask = (
        event_type.isin(
            RECOVERY_TYPES
        )
        | type_id.eq(49)
    )

    clearance_mask = (
        event_type.isin(
            CLEARANCE_TYPES
        )
        | type_id.eq(12)
    )

    # Opta block semantics:
    # - event 74 = blocked pass
    # - event 10 + qualifier 94 ("Def block") = defender blocks a shot
    # The qualifier is flattened by preprocess into a "Def block" flag.
    block_mask = (
        event_type.isin(
            BLOCK_TYPES
        )
        | type_id.eq(74)
        | (
            type_id.eq(10)
            & _truthy_series(
                frame,
                "Def block",
            )
        )
    )

    # Opta foul semantics:
    # committed foul = unsuccessful event
    # suffered foul   = successful event
    foul_event_mask = (
        event_type.isin(
            FOUL_TYPES
        )
        | type_id.eq(4)
    )

    foul_mask = (
        foul_event_mask
        & ~success
    )

    unique_mask = (
        successful_tackle_mask
        | interception_mask
        | recovery_mask
        | clearance_mask
        | block_mask
    )

    return {
        "unique": int(
            event_key.loc[
                unique_mask
            ].nunique()
        ),
        "tackles_won": int(
            successful_tackle_mask.sum()
        ),
        "tackles_attempted": int(
            tackle_mask.sum()
        ),
        "interceptions": int(
            interception_mask.sum()
        ),
        "recoveries": int(
            recovery_mask.sum()
        ),
        "clearances": int(
            clearance_mask.sum()
        ),
        "blocks": int(
            block_mask.sum()
        ),
        "fouls": int(
            foul_mask.sum()
        ),
    }


def build_defensive_ranking(
    df_processed,
    *,
    num_players=10,
):
    columns = [
        "playerName",
        "team_name",
        "jersey",
        "unique",
        "tackles_won",
        "tackles_attempted",
        "interceptions",
        "recoveries",
        "clearances",
        "blocks",
        "fouls",
    ]

    if (
        df_processed is None
        or df_processed.empty
        or "playerName"
        not in df_processed.columns
        or "type_name"
        not in df_processed.columns
    ):
        return pd.DataFrame(
            columns=columns
        )

    rows = []

    for (
        player_name,
        player_events,
    ) in (
        df_processed
        .dropna(
            subset=[
                "playerName"
            ]
        )
        .groupby(
            "playerName",
            sort=False,
        )
    ):
        profile = player_defensive_profile(
            player_events
        )

        if (
            profile["unique"] <= 0
            and profile["fouls"] <= 0
        ):
            continue

        last_row = player_events.iloc[-1]

        rows.append({
            "playerName":
                str(
                    player_name
                ),
            "team_name":
                _clean_text(
                    last_row.get(
                        "team_name"
                    )
                ),
            "jersey":
                _jersey_label(
                    last_row.get(
                        "Mapped Jersey Number"
                    )
                ),
            **profile,
        })

    if not rows:
        return pd.DataFrame(
            columns=columns
        )

    ranking = pd.DataFrame(
        rows
    )

    return (
        ranking
        .sort_values(
            [
                "unique",
                "interceptions",
                "tackles_won",
                "recoveries",
                "clearances",
                "blocks",
                "fouls",
                "playerName",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
            ],
            kind="stable",
        )
        .head(
            int(
                num_players
            )
        )
        .reset_index(
            drop=True
        )
    )
