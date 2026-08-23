"""Shared Dash shell for Match Analysis graph sections.

FOUND-03 centralises the repeated presentation structure around analytical
figures without owning metric computation or Plotly semantics.

The shell provides:
- team header + sample size;
- methodology note;
- graph + optional KPI/sidebar column;
- empty state;
- analyst notes panel.

Callers keep ownership of the actual graph, KPI content and callbacks.
"""

from __future__ import annotations

from typing import Iterable, Optional

import dash_bootstrap_components as dbc
from dash import dcc, html


def methodology_note(
    text,
    *,
    icon_class="fas fa-info-circle",
    class_name="",
):
    """Return the shared Match Analysis methodology note."""
    if not text:
        return None

    classes = "match-analysis-note match-graph-methodology"
    if class_name:
        classes += f" {class_name}"

    return html.Div(
        [
            html.I(className=icon_class),
            html.Span(text),
        ],
        className=classes,
    )


def analyst_notes_panel(
    *,
    textarea_id,
    save_button_id,
    status_id,
    description,
    placeholder,
    title="Analyst notes",
):
    """Return the shared analyst-notes editor while callbacks stay external."""
    return html.Section(
        [
            html.Div(
                [
                    html.I(
                        className="fa-regular fa-note-sticky"
                    ),
                    html.Div(
                        [
                            html.H3(
                                title,
                                className="match-panel-title",
                            ),
                            html.P(
                                description,
                                className="match-panel-description",
                            ),
                        ]
                    ),
                ],
                className="match-comment-heading",
            ),
            dcc.Textarea(
                id=textarea_id,
                placeholder=placeholder,
                className="match-comment-input",
            ),
            html.Div(
                [
                    dbc.Button(
                        [
                            html.I(
                                className=(
                                    "fa-solid fa-floppy-disk me-2"
                                )
                            ),
                            "Save note",
                        ],
                        id=save_button_id,
                        className="match-action-button",
                        size="sm",
                    ),
                    html.Div(
                        id=status_id,
                        className="small",
                    ),
                ],
                className="match-comment-actions",
            ),
        ],
        className=(
            "match-panel match-comment-panel "
            "match-graph-analyst-notes"
        ),
    )


def match_graph_panel(
    *,
    team_name,
    is_away,
    sample_size,
    graph=None,
    sidebar=None,
    empty_message=None,
    eyebrow=None,
    panel_class_name="",
    graph_column_width=8,
    sidebar_column_width=4,
):
    """
    Shared team graph card.

    ``empty_message`` takes precedence over graph/sidebar content. It allows a
    caller to preserve the same shell when no analytical sample is available.
    """
    eyebrow_text = (
        eyebrow
        if eyebrow is not None
        else (
            "AWAY TEAM"
            if is_away
            else "HOME TEAM"
        )
    )

    panel_classes = (
        "match-panel progressive-team-panel "
        "match-graph-team-panel"
    )
    if panel_class_name:
        panel_classes += f" {panel_class_name}"

    header = html.Div(
        [
            html.Div(
                [
                    html.Span(
                        eyebrow_text,
                        className="match-panel-eyebrow",
                    ),
                    html.H4(
                        team_name,
                        className="match-team-name",
                    ),
                ]
            ),
            html.Div(
                [
                    html.Span(
                        sample_size,
                        className=(
                            "progressive-sample-size "
                            "match-graph-sample-size"
                        ),
                    ),
                ],
                className=(
                    "progressive-panel-meta "
                    "match-graph-panel-meta"
                ),
            ),
        ],
        className="match-panel-header match-graph-panel-header",
    )

    if empty_message:
        body = html.Div(
            [
                html.I(
                    className=(
                        "fa-regular fa-chart-bar "
                        "match-graph-empty-icon"
                    )
                ),
                html.Strong("No data"),
                html.Span(empty_message),
            ],
            className="match-graph-empty-state",
        )
    elif sidebar is not None:
        body = dbc.Row(
            [
                dbc.Col(
                    graph,
                    lg=graph_column_width,
                    className="match-graph-main-column",
                ),
                dbc.Col(
                    sidebar,
                    lg=sidebar_column_width,
                    className="match-graph-sidebar-column",
                ),
            ],
            className="g-0 match-graph-panel-body",
        )
    else:
        body = html.Div(
            graph,
            className="match-graph-full-width",
        )

    return html.Section(
        [
            header,
            body,
        ],
        className=panel_classes,
    )


def match_graph_shell(
    *,
    panels: Iterable,
    methodology=None,
    analyst_notes=None,
    class_name="",
):
    """
    Shared section shell around one or more team graph panels.

    ``analyst_notes`` can be an already-built Dash component or a dict accepted
    by :func:`analyst_notes_panel`.
    """
    children = []

    if methodology:
        if isinstance(methodology, str):
            children.append(
                methodology_note(methodology)
            )
        else:
            children.append(methodology)

    children.extend(
        panel
        for panel in panels
        if panel is not None
    )

    if analyst_notes:
        if isinstance(analyst_notes, dict):
            children.append(
                analyst_notes_panel(
                    **analyst_notes
                )
            )
        else:
            children.append(
                analyst_notes
            )

    classes = "match-graph-shell"
    if class_name:
        classes += f" {class_name}"

    return html.Div(
        children,
        className=classes,
    )
