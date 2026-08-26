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


def _defensive_masks(frame):
    """Return the canonical PLOT-17 defensive event masks.

    This is deliberately shared by the ranking and the player map so both
    views count exactly the same Opta events.
    """
    event_type = _event_type_series(
        frame
    )
    type_id = _event_type_id_series(
        frame
    )
    success = _successful_mask(
        frame
    )

    tackle_mask = (
        event_type.isin(
            TACKLE_TYPES
        )
        | type_id.eq(7)
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

    successful_tackle_mask = (
        tackle_mask
        & success
    )

    unique_mask = (
        successful_tackle_mask
        | interception_mask
        | recovery_mask
        | clearance_mask
        | block_mask
    )

    return {
        "success": success,
        "tackle": tackle_mask,
        "successful_tackle": successful_tackle_mask,
        "interception": interception_mask,
        "recovery": recovery_mask,
        "clearance": clearance_mask,
        "block": block_mask,
        "foul": foul_mask,
        "unique": unique_mask,
    }


def classify_defensive_events(
    player_events,
):
    """Classify each relevant event into one non-overlapping map category."""
    if (
        player_events is None
        or player_events.empty
        or "type_name"
        not in player_events.columns
    ):
        return pd.DataFrame()

    frame = player_events.copy()
    masks = _defensive_masks(
        frame
    )

    frame["defensive_category"] = pd.NA
    frame["defensive_detail"] = pd.NA
    frame["defensive_positive"] = False
    frame["tackle_won"] = False

    assignments = (
        (
            "Tackle",
            masks["tackle"],
        ),
        (
            "Interception",
            masks["interception"],
        ),
        (
            "Recovery",
            masks["recovery"],
        ),
        (
            "Clearance",
            masks["clearance"],
        ),
        (
            "Block",
            masks["block"],
        ),
        (
            "Foul",
            masks["foul"],
        ),
    )

    # The masks are semantically disjoint for Opta data. The ``isna`` guard
    # also guarantees exactly one visual category if a malformed row happens
    # to satisfy more than one fallback rule.
    for category, mask in assignments:
        available = (
            mask
            & frame["defensive_category"].isna()
        )
        frame.loc[
            available,
            "defensive_category",
        ] = category

    frame.loc[
        masks["unique"],
        "defensive_positive",
    ] = True
    frame.loc[
        masks["successful_tackle"],
        "tackle_won",
    ] = True

    type_id = _event_type_id_series(
        frame
    )
    event_type = _event_type_series(
        frame
    )

    blocked_pass = (
        type_id.eq(74)
        | event_type.eq("blocked pass")
    )
    shot_block = (
        type_id.eq(10)
        & _truthy_series(
            frame,
            "Def block",
        )
    )

    frame.loc[
        frame["defensive_category"].eq("Tackle")
        & frame["tackle_won"],
        "defensive_detail",
    ] = "Tackle won"
    frame.loc[
        frame["defensive_category"].eq("Tackle")
        & ~frame["tackle_won"],
        "defensive_detail",
    ] = "Tackle unsuccessful"
    frame.loc[
        frame["defensive_category"].eq("Interception"),
        "defensive_detail",
    ] = "Interception"
    frame.loc[
        frame["defensive_category"].eq("Recovery"),
        "defensive_detail",
    ] = "Ball recovery"
    frame.loc[
        frame["defensive_category"].eq("Clearance"),
        "defensive_detail",
    ] = "Clearance"
    frame.loc[
        frame["defensive_category"].eq("Block"),
        "defensive_detail",
    ] = "Block"
    frame.loc[
        blocked_pass
        & frame["defensive_category"].eq("Block"),
        "defensive_detail",
    ] = "Blocked pass"
    frame.loc[
        shot_block
        & frame["defensive_category"].eq("Block"),
        "defensive_detail",
    ] = "Shot block"
    frame.loc[
        frame["defensive_category"].eq("Foul"),
        "defensive_detail",
    ] = "Foul committed"

    return (
        frame[
            frame["defensive_category"]
            .notna()
        ]
        .copy()
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

    event_key = _event_key_series(
        frame
    )
    masks = _defensive_masks(
        frame
    )

    return {
        "unique": int(
            event_key.loc[
                masks["unique"]
            ].nunique()
        ),
        "tackles_won": int(
            masks["successful_tackle"].sum()
        ),
        "tackles_attempted": int(
            masks["tackle"].sum()
        ),
        "interceptions": int(
            masks["interception"].sum()
        ),
        "recoveries": int(
            masks["recovery"].sum()
        ),
        "clearances": int(
            masks["clearance"].sum()
        ),
        "blocks": int(
            masks["block"].sum()
        ),
        "fouls": int(
            masks["foul"].sum()
        ),
    }


def team_defensive_player_options(
    df_processed,
    team_name,
):
    """Return map selector options ranked by the canonical defensive profile."""
    if (
        df_processed is None
        or df_processed.empty
        or "team_name"
        not in df_processed.columns
        or "playerName"
        not in df_processed.columns
    ):
        return []

    team_events = df_processed.loc[
        df_processed["team_name"].eq(
            team_name
        )
    ].copy()

    if team_events.empty:
        return []

    ranking = build_defensive_ranking(
        team_events,
        num_players=max(
            int(
                team_events[
                    "playerName"
                ]
                .nunique(
                    dropna=True
                )
            ),
            1,
        ),
    )

    if ranking.empty:
        return []

    return [
        {
            "player_name": row["playerName"],
            "jersey": row["jersey"],
            "unique": int(row["unique"]),
            "fouls": int(row["fouls"]),
        }
        for row in ranking.to_dict(
            orient="records"
        )
    ]


def player_team_and_jersey(
    df_processed,
    player_name,
):
    if (
        df_processed is None
        or df_processed.empty
        or "playerName"
        not in df_processed.columns
    ):
        return "", "?"

    rows = df_processed.loc[
        df_processed["playerName"].eq(
            player_name
        )
    ]

    if rows.empty:
        return "", "?"

    row = rows.iloc[-1]
    return (
        _clean_text(
            row.get(
                "team_name"
            )
        ),
        _jersey_label(
            row.get(
                "Mapped Jersey Number"
            )
        ),
    )


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
