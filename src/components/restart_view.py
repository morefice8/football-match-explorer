from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from src.metrics.restart_panel_metrics import (
    FILTER_SPECS,
    option_counts,
)


def _filter_chip(
    *,
    filter_key,
    value,
    count,
    active,
):
    return dbc.Button(
        [
            html.Span(
                value,
                className=
                    "restart-filter-chip-label",
            ),
            html.Span(
                str(count),
                className=
                    "restart-filter-chip-count",
            ),
        ],
        id={
            "type":
                "sp-filter",
            "filter_type":
                filter_key,
            "value":
                value,
        },
        n_clicks=0,
        color="light",
        className=(
            "restart-filter-chip "
            + (
                "restart-filter-chip--active"
                if active
                else ""
            )
        ),
        size="sm",
    )


def filter_panel(
    records,
    active_filter,
):
    active_filter = (
        active_filter
        or {}
    )

    groups = []

    for (
        filter_key,
        label,
        _,
    ) in FILTER_SPECS:
        counts = option_counts(
            records,
            active_filter,
            filter_key,
        )

        chips = [
            _filter_chip(
                filter_key=
                    filter_key,
                value=value,
                count=count,
                active=(
                    active_filter.get(
                        filter_key
                    )
                    == value
                ),
            )
            for (
                value,
                count,
            ) in counts.items()
        ]

        groups.append(
            html.Div(
                [
                    html.Span(
                        label.upper(),
                        className=
                            "restart-filter-label",
                    ),
                    html.Div(
                        chips
                        or [
                            html.Span(
                                "No values",
                                className=
                                    "restart-filter-empty",
                            )
                        ],
                        className=
                            "restart-filter-options",
                    ),
                ],
                className=
                    "restart-filter-group",
            )
        )

    return html.Div(
        groups,
        className=
            "restart-filter-grid",
    )


def active_filter_badges(
    active_filter,
):
    active_filter = (
        active_filter
        or {}
    )

    if not active_filter:
        return None

    labels = {
        key: label
        for (
            key,
            label,
            _,
        ) in FILTER_SPECS
    }

    return html.Div(
        [
            html.Span(
                "Active filters",
                className=
                    "restart-active-filter-title",
            ),
            *[
                html.Span(
                    (
                        f"{labels.get(key, key)}: "
                        f"{value}"
                    ),
                    className=
                        "restart-active-filter-badge",
                )
                for (
                    key,
                    value,
                ) in active_filter.items()
            ],
        ],
        className=
            "restart-active-filters",
    )


def selected_restart_meta(
    record,
):
    if not record:
        return None

    fields = (
        (
            "Type",
            record.get(
                "restart_type"
            ),
        ),
        (
            "Side",
            record.get(
                "side"
            ),
        ),
        (
            "Delivery",
            record.get(
                "delivery"
            ),
        ),
        (
            "Destination",
            record.get(
                "destination"
            ),
        ),
        (
            "Execution",
            record.get(
                "outcome"
            ),
        ),
        (
            "Development",
            record.get(
                "development_outcome"
            ),
        ),
    )

    return html.Div(
        [
            html.Span(
                [
                    html.Strong(
                        label + ": "
                    ),
                    str(
                        value
                    ),
                ],
                className=
                    "restart-selected-meta-chip",
            )
            for (
                label,
                value,
            ) in fields
        ],
        className=
            "restart-selected-meta",
    )
