from __future__ import annotations

from dash import dcc, html


ATTACK_CLASS = {
    "Offensive Transition": "goal-origin-badge--transition",
    "Set Piece": "goal-origin-badge--set-piece",
    "Penalty": "goal-origin-badge--set-piece",
    "Positional Attack": "goal-origin-badge--positional",
}


def _minute_label(record):
    try:
        minute = int(float(record.get("minute")))
    except (TypeError, ValueError):
        return "—"

    try:
        period = int(float(record.get("period_id")))
    except (TypeError, ValueError):
        period = None

    if period == 1 and minute > 45:
        return f"45+{minute - 45}'"
    if period == 2 and minute > 90:
        return f"90+{minute - 90}'"
    return f"{minute}'"


def _duration_label(value):
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"

    if value < 10:
        return f"{value:.1f}s"
    return f"{value:.0f}s"


def _detail_line(label, value, detail=None):
    if not value:
        return None

    children = [
        html.Span(label, className="goal-origin-detail-label"),
        html.Strong(str(value), className="goal-origin-detail-value"),
    ]

    if detail:
        children.append(
            html.Span(
                f"· {detail}",
                className="goal-origin-detail-extra",
            )
        )

    return html.Div(children, className="goal-origin-detail-row")


def _goal_card(record, *, home_team, hcol, acol):
    team_name = record.get("team_name")
    team_color = hcol if team_name == home_team else acol

    attack_type = record.get("attack_type") or "Unclassified"
    badge_class = ATTACK_CLASS.get(
        attack_type,
        "goal-origin-badge--positional",
    )

    origin_detail = record.get("origin_detail")
    decisive_player = record.get("decisive_player")

    assist = record.get("official_assist")
    shot_creator = record.get("shot_creating_pass")

    creator_label = None
    creator_value = None

    if assist:
        creator_label = "Assist"
        creator_value = assist
    elif shot_creator:
        creator_label = "Shot-creating pass"
        creator_value = shot_creator

    analysis_tab = record.get("analysis_tab")
    analysis_module = record.get("analysis_module") or "Analysis"

    if analysis_tab:
        route = dcc.Link(
            [
                html.Span(f"View {analysis_module}"),
                html.I(className="fa-solid fa-arrow-right"),
            ],
            href=f"?tab={analysis_tab}",
            className="goal-origin-link",
        )
    else:
        route = html.Div(
            [
                html.Span("Analysis path"),
                html.Strong(analysis_module),
            ],
            className="goal-origin-route-note",
        )

    return html.Article(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                _minute_label(record),
                                className="goal-origin-minute",
                            ),
                            html.Div(
                                [
                                    html.Strong(
                                        record.get("scorer") or "Unknown",
                                        className="goal-origin-scorer",
                                    ),
                                    html.Small(
                                        team_name or "",
                                        className="goal-origin-team",
                                    ),
                                ]
                            ),
                        ],
                        className="goal-origin-title-group",
                    ),
                    html.Span(
                        attack_type.upper(),
                        className=f"goal-origin-badge {badge_class}",
                    ),
                ],
                className="goal-origin-card-header",
            ),

            html.Div(
                [
                    _detail_line(
                        "Origin",
                        record.get("possession_origin"),
                        origin_detail,
                    ),
                    _detail_line(
                        "Decisive moment",
                        record.get("decisive_mechanism"),
                        decisive_player,
                    ),
                    _detail_line(
                        creator_label,
                        creator_value,
                    ) if creator_label else None,
                ],
                className="goal-origin-details",
            ),

            html.Div(
                [
                    html.Div(
                        [
                            html.Strong(
                                _duration_label(
                                    record.get("possession_duration_seconds")
                                )
                            ),
                            html.Span("Possession"),
                        ],
                        className="goal-origin-stat",
                    ),
                    html.Div(
                        [
                            html.Strong(str(record.get("pass_count", 0))),
                            html.Span("Passes"),
                        ],
                        className="goal-origin-stat",
                    ),
                    html.Div(
                        [
                            html.Strong(str(record.get("action_count", 0))),
                            html.Span("Actions"),
                        ],
                        className="goal-origin-stat",
                    ),
                ],
                className="goal-origin-stats",
            ),

            route,
        ],
        className="goal-origin-card",
        style={"borderTopColor": team_color},
    )


def panel(records, *, home_team, hcol, acol):
    if not records:
        return None

    return html.Section(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "GOAL BREAKDOWN",
                                className="match-panel-eyebrow",
                            ),
                            html.H3(
                                "Goal origins",
                                className="match-panel-title",
                            ),
                            html.P(
                                "How each scoring possession started, developed and reached its decisive moment.",
                                className="match-panel-description",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.I(className="fa-solid fa-circle-info"),
                            html.Span(
                                "Origin and attack type are classified separately, so a restart can develop into open play before the goal."
                            ),
                        ],
                        className="match-panel-hint",
                    ),
                ],
                className="match-panel-header",
            ),
            html.Div(
                [
                    _goal_card(
                        record,
                        home_team=home_team,
                        hcol=hcol,
                        acol=acol,
                    )
                    for record in records
                ],
                className="goal-origin-grid",
            ),
        ],
        className="match-panel goal-origin-panel",
    )
