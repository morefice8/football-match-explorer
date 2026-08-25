from __future__ import annotations

import pandas as pd

from src.data_processing import pass_processing
from src.metrics import (
    shot_sequence_involvement_metrics,
    shot_sequence_metrics,
)

CONTRIBUTOR_ROLES = frozenset({
    "shooter",
    "shot_assist",
    "pre_assist",
})

SHOT_TYPES = frozenset({
    "goal",
    "miss",
    "attempt saved",
    "post",
    "shot",
})

SEQUENCE_ID_CANDIDATES = (
    "shot_sequence_id",
    "sequence_id",
    "possession_sequence_id",
    "shot_sequence",
)

EVENT_ID_CANDIDATES = (
    "id",
    "eventId",
    "event_id",
)


def _clean_text(value, fallback=""):
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or fallback


def _successful_mask(frame):
    outcome = frame.get(
        "outcome",
        pd.Series("", index=frame.index),
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


def _valid_xy(frame):
    required = (
        "x",
        "y",
        "end_x",
        "end_y",
    )
    if not all(
        column in frame.columns
        for column in required
    ):
        return pd.Series(
            False,
            index=frame.index,
        )

    return (
        pd.to_numeric(
            frame["x"],
            errors="coerce",
        ).notna()
        & pd.to_numeric(
            frame["y"],
            errors="coerce",
        ).notna()
        & pd.to_numeric(
            frame["end_x"],
            errors="coerce",
        ).notna()
        & pd.to_numeric(
            frame["end_y"],
            errors="coerce",
        ).notna()
    )


def _is_shot(row):
    return (
        _clean_text(
            row.get("type_name")
        ).lower()
        in SHOT_TYPES
    )


def _sequence_id_column(frame):
    for column in SEQUENCE_ID_CANDIDATES:
        if column in frame.columns:
            return column
    return None


def _event_id_column(frame):
    for column in EVENT_ID_CANDIDATES:
        if column in frame.columns:
            return column
    return None


def _with_sequence_key(sequences):
    if sequences is None or sequences.empty:
        result = pd.DataFrame()
        result["_shot_sequence_key"] = pd.Series(dtype="object")
        return result

    result = sequences.copy()
    sequence_id_column = _sequence_id_column(result)

    if sequence_id_column is not None:
        result["_shot_sequence_key"] = result[sequence_id_column].map(
            lambda value: _clean_text(value)
        )
        return result

    keys = []
    sequence_number = 0

    for _, row in result.iterrows():
        keys.append(
            f"sequence-{sequence_number}"
        )

        if _is_shot(row):
            sequence_number += 1

    result["_shot_sequence_key"] = keys
    return result


def _base_received_passes(
    df_processed,
    selected_player,
    team_name,
):
    if (
        df_processed is None
        or df_processed.empty
        or not selected_player
    ):
        return pd.DataFrame()

    passes = (
        pass_processing
        .get_passes_df(
            df_processed.copy()
        )
    )

    if (
        passes is None
        or passes.empty
        or "receiver" not in passes.columns
    ):
        return pd.DataFrame()

    mask = (
        passes["receiver"]
        .fillna("")
        .astype(str)
        .eq(
            str(selected_player)
        )
        & _successful_mask(
            passes
        )
        & _valid_xy(
            passes
        )
    )

    if (
        team_name
        and "team_name" in passes.columns
    ):
        mask &= (
            passes["team_name"]
            .fillna("")
            .astype(str)
            .eq(
                str(team_name)
            )
        )

    if (
        "receiver_is_reliable"
        in passes.columns
    ):
        mask &= (
            passes[
                "receiver_is_reliable"
            ]
            .fillna(False)
            .astype(bool)
        )

    return (
        passes.loc[
            mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


def _focus_sequence_passes(
    df_processed,
    selected_player,
):
    """
    Return pass events from the shot-ending sequences in which selected_player
    appears as shooter, shot-assist provider or pre-assist provider.

    Important: this does not require the sequence frame to expose a receiver
    column. We identify the relevant sequences first and then intersect their
    pass event IDs with the already validated received-pass frame.
    """
    if (
        df_processed is None
        or df_processed.empty
        or not selected_player
    ):
        return pd.DataFrame()

    sequences = (
        shot_sequence_metrics
        .build_shot_sequences(
            df_processed
        )
    )

    if (
        sequences is None
        or sequences.empty
    ):
        return pd.DataFrame()

    frame = _with_sequence_key(
        sequences
    )

    if (
        "playerName" not in frame.columns
        or "sequence_role" not in frame.columns
        or "type_name" not in frame.columns
    ):
        return pd.DataFrame()

    role = (
        frame["sequence_role"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    selected_mask = (
        frame["playerName"]
        .fillna("")
        .astype(str)
        .eq(
            str(selected_player)
        )
        & role.isin(
            CONTRIBUTOR_ROLES
        )
    )

    selected_keys = (
        frame.loc[
            selected_mask,
            "_shot_sequence_key",
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not selected_keys:
        return pd.DataFrame()

    type_name = (
        frame["type_name"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    focus = (
        frame[
            frame[
                "_shot_sequence_key"
            ]
            .astype(str)
            .isin(
                selected_keys
            )
            & type_name.eq("pass")
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return focus


def _flag_focus_receptions(
    receptions,
    focus_passes,
):
    result = receptions.copy()
    result["_in_shot_sequence"] = False

    if (
        result.empty
        or focus_passes is None
        or focus_passes.empty
    ):
        return result

    reception_id = _event_id_column(
        result
    )
    focus_id = _event_id_column(
        focus_passes
    )

    if (
        reception_id is not None
        and focus_id is not None
    ):
        focus_ids = set(
            focus_passes[
                focus_id
            ]
            .dropna()
            .astype(str)
        )

        result.loc[
            result[
                reception_id
            ]
            .fillna("")
            .astype(str)
            .isin(
                focus_ids
            ),
            "_in_shot_sequence",
        ] = True

        if result[
            "_in_shot_sequence"
        ].any():
            return result

    # Conservative fallback when source event IDs are unavailable or transformed.
    fallback_keys = [
        column
        for column in (
            "playerName",
            "timeMin",
            "timeSec",
            "x",
            "y",
            "end_x",
            "end_y",
        )
        if (
            column in result.columns
            and column in focus_passes.columns
        )
    ]

    if not fallback_keys:
        return result

    focus_keys = set(
        tuple(row)
        for row in (
            focus_passes[
                fallback_keys
            ]
            .fillna("")
            .astype(str)
            .itertuples(
                index=False,
                name=None,
            )
        )
    )

    result_keys = (
        result[
            fallback_keys
        ]
        .fillna("")
        .astype(str)
        .apply(
            tuple,
            axis=1,
        )
    )

    result.loc[
        result_keys.isin(
            focus_keys
        ),
        "_in_shot_sequence",
    ] = True

    return result


def received_passes_for_player(
    df_processed,
    selected_player,
    team_name,
):
    receptions = _base_received_passes(
        df_processed,
        selected_player,
        team_name,
    )

    focus_passes = _focus_sequence_passes(
        df_processed,
        selected_player,
    )

    return _flag_focus_receptions(
        receptions,
        focus_passes,
    )


def reception_summary(
    receptions,
):
    summary = {
        "receptions": 0,
        "final_third": 0,
        "box": 0,
        "passers": 0,
        "dangerous_receptions": 0,
        "median_x": None,
        "median_y": None,
    }

    if (
        receptions is None
        or receptions.empty
    ):
        return summary

    end_x = pd.to_numeric(
        receptions["end_x"],
        errors="coerce",
    )
    end_y = pd.to_numeric(
        receptions["end_y"],
        errors="coerce",
    )

    summary[
        "receptions"
    ] = int(
        len(receptions)
    )

    summary[
        "final_third"
    ] = int(
        end_x.ge(
            66.67
        ).sum()
    )

    summary[
        "box"
    ] = int(
        (
            end_x.ge(
                83.5
            )
            & end_y.between(
                21.1,
                78.9,
                inclusive="both",
            )
        ).sum()
    )

    if (
        "playerName" in receptions.columns
    ):
        summary[
            "passers"
        ] = int(
            receptions[
                "playerName"
            ]
            .dropna()
            .astype(str)
            .nunique()
        )

    if (
        "_in_shot_sequence" in receptions.columns
    ):
        summary[
            "dangerous_receptions"
        ] = int(
            receptions[
                "_in_shot_sequence"
            ]
            .fillna(False)
            .astype(bool)
            .sum()
        )

    if end_x.notna().any():
        summary[
            "median_x"
        ] = float(
            end_x.median()
        )

    if end_y.notna().any():
        summary[
            "median_y"
        ] = float(
            end_y.median()
        )

    return summary


def _canonical_ranking(
    df_processed,
):
    if (
        df_processed is None
        or df_processed.empty
    ):
        return pd.DataFrame()

    stats = (
        shot_sequence_metrics
        .calculate_shot_sequence_player_stats(
            df_processed
        )
    )

    if (
        stats is None
        or stats.empty
    ):
        return pd.DataFrame()

    return (
        shot_sequence_involvement_metrics
        .prepare_shot_sequence_ranking(
            stats,
            num_players=max(
                len(stats),
                1,
            ),
        )
    )


def team_player_options(
    df_processed,
    team_name,
):
    ranking = _canonical_ranking(
        df_processed
    )

    if ranking.empty:
        return []

    if (
        "playerName" not in df_processed.columns
        or "team_name" not in df_processed.columns
    ):
        return []

    team_rows = (
        df_processed[
            df_processed[
                "team_name"
            ]
            .fillna("")
            .astype(str)
            .eq(
                str(team_name)
            )
        ]
        .copy()
    )

    team_players = set(
        team_rows[
            "playerName"
        ]
        .dropna()
        .astype(str)
    )

    jersey_map = {}

    if (
        "Mapped Jersey Number" in team_rows.columns
    ):
        jersey_map = (
            team_rows
            .dropna(
                subset=[
                    "playerName"
                ]
            )
            .drop_duplicates(
                "playerName",
                keep="last",
            )
            .set_index(
                "playerName"
            )[
                "Mapped Jersey Number"
            ]
            .to_dict()
        )

    options = []

    for (
        player_name,
        row,
    ) in ranking.iterrows():
        if (
            str(player_name)
            not in team_players
        ):
            continue

        try:
            jersey = str(
                int(
                    float(
                        jersey_map.get(
                            player_name
                        )
                    )
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            jersey = "?"

        options.append({
            "player_name":
                str(player_name),
            "jersey":
                jersey,
            "involvements":
                int(
                    row.get(
                        "Shot Sequence Involvements",
                        0,
                    )
                    or 0
                ),
        })

    return options


def player_team_and_jersey(
    df_processed,
    selected_player,
):
    if (
        df_processed is None
        or df_processed.empty
        or "playerName" not in df_processed.columns
    ):
        return None, "?"

    rows = df_processed[
        df_processed[
            "playerName"
        ]
        .fillna("")
        .astype(str)
        .eq(
            str(selected_player)
        )
    ]

    if rows.empty:
        return None, "?"

    row = rows.iloc[-1]

    jersey_raw = row.get(
        "Mapped Jersey Number"
    )

    try:
        jersey = str(
            int(
                float(
                    jersey_raw
                )
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        jersey = "?"

    return (
        row.get(
            "team_name"
        ),
        jersey,
    )
