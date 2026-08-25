from __future__ import annotations

import re

import pandas as pd


CANONICAL_COLUMNS = (
    "Shot Sequence Involvements",
    "Shot Sequence Shots",
    "Shot Sequence Shot Assists",
    "Shot Sequence Pre-Assists",
)


def _normalise_column_name(value):
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(value).lower(),
    ).strip()


def _shot_assist_source_column(columns):
    """
    Find the REL-08 shot-assist metric without falling back to the legacy
    generic 'Shot Assists' player-stat column.

    REL-08 deliberately keeps passing assists and shot-sequence assists
    separate. Accept sequence-prefixed naming variants only.
    """
    exact_aliases = (
        "Shot Sequence Shot Assists",
        "Shot Sequence Assists",
        "Shot Sequence Shot-Creating Passes",
        "Shot Sequence Shot Creating Passes",
    )

    for alias in exact_aliases:
        if alias in columns:
            return alias

    candidates = []

    for column in columns:
        normalised = _normalise_column_name(
            column
        )

        if "shot sequence" not in normalised:
            continue

        if "pre assist" in normalised:
            continue

        is_assist = (
            "assist" in normalised
        )

        is_shot_creating = (
            "shot creating pass"
            in normalised
            or "creating pass"
            in normalised
        )

        if (
            is_assist
            or is_shot_creating
        ):
            candidates.append(
                column
            )

    if len(candidates) == 1:
        return candidates[0]

    return None


def _canonicalise_rel08_columns(
    stats,
):
    ranking = stats.copy()

    assist_source = (
        _shot_assist_source_column(
            ranking.columns
        )
    )

    if (
        "Shot Sequence Shot Assists"
        not in ranking.columns
        and assist_source is not None
    ):
        ranking[
            "Shot Sequence Shot Assists"
        ] = ranking[
            assist_source
        ]

    return ranking


def prepare_shot_sequence_ranking(
    stats,
    *,
    num_players=10,
):
    """
    Presentation adapter for REL-08 shot-sequence player stats.

    Ranking uses the canonical unweighted involvement count from REL-08.
    No weighted score is created or used here.
    """
    if (
        stats is None
        or stats.empty
    ):
        return pd.DataFrame()

    ranking = (
        _canonicalise_rel08_columns(
            stats
        )
    )

    missing = [
        column
        for column
        in CANONICAL_COLUMNS
        if column
        not in ranking.columns
    ]

    if missing:
        raise ValueError(
            "Missing REL-08 shot-sequence columns: "
            + ", ".join(
                missing
            )
            + ". Available columns: "
            + ", ".join(
                map(
                    str,
                    ranking.columns,
                )
            )
        )

    for column in CANONICAL_COLUMNS:
        ranking[column] = (
            pd.to_numeric(
                ranking[column],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )

    ranking = (
        ranking
        .loc[
            ranking[
                "Shot Sequence Involvements"
            ]
            > 0
        ]
        .sort_values(
            [
                "Shot Sequence Involvements",
                "Shot Sequence Shots",
                "Shot Sequence Shot Assists",
                "Shot Sequence Pre-Assists",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
            kind="stable",
        )
        .head(
            int(
                num_players
            )
        )
    )

    return ranking
