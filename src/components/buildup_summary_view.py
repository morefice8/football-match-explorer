"""Dash presentation components for build-up summary data."""

import dash_bootstrap_components as dbc
from dash import html as dash_html


def create_buildup_summary_cards(stats, active_filter=None):
    """
    Creates a layout of interactive cards displaying the calculated buildup stats.
    The active filter will be highlighted with a custom style.
    """
    if not stats or stats.get("total", 0) == 0:
        return dbc.Alert("No summary data to display.", color="secondary")

    def is_active(filter_type, value):
        if not active_filter:
            return False
        return str(active_filter.get(filter_type)) == str(value)

    # Card 1: Outcomes
    outcome_hierarchy = ['Goals', 'Penalty won', 'Shots', 'Big Chances', 'Lost Possessions', 'Foul', 'Offside', 'Out']
    outcome_rank_map = {outcome: i for i, outcome in enumerate(outcome_hierarchy)}
    sorted_outcome_items = sorted(stats.get('outcomes', {}).items(), key=lambda item: outcome_rank_map.get(item[0], 99))

    outcome_list_items = []
    for outcome, count in sorted_outcome_items:
        if count == 0:
            continue
        if active_filter and 'outcomes' in active_filter and active_filter['outcomes'] != outcome:
            continue
        active = is_active('outcomes', outcome)
        outcome_list_items.append(
            dbc.ListGroupItem(
                [
                    dash_html.Div(f"{outcome}"),
                    dbc.Badge(f"{count} ({count / stats['total']:.0%})", color="light", text_color="dark", className="ms-auto")
                ],
                id={'type': 'buildup-filter', 'filter_type': 'outcomes', 'value': outcome},
                action=True,
                n_clicks=0,
                active=active,
                className=f"match-summary-row {'is-active' if active else ''}"
            )
        )
    outcome_card = dbc.Card([
        dbc.CardHeader([dash_html.I(className="fa-solid fa-bullseye me-2"), "Buildup outcomes"]),
        dbc.ListGroup(outcome_list_items, flush=True)
    ], className="match-summary-card")

    # Card 2: Flanks
    flank_list_items = []
    for flank, count in stats['flanks'].items():
        if count == 0:
            continue
        if active_filter and 'flanks' in active_filter and active_filter['flanks'] != flank:
            continue
        active = is_active('flanks', flank)
        flank_list_items.append(
            dbc.ListGroupItem(
                [
                    dash_html.Div(flank),
                    dbc.Badge(f"{count} ({count / stats['total']:.0%})", color="light", text_color="dark", className="ms-auto")
                ],
                id={'type': 'buildup-filter', 'filter_type': 'flanks', 'value': flank},
                action=True,
                n_clicks=0,
                active=active,
                className=f"match-summary-row {'is-active' if active else ''}"
            )
        )
    flank_card = dbc.Card([
        dbc.CardHeader([dash_html.I(className="fa-solid fa-arrows-left-right me-2"), "Dominant flank"]),
        dbc.ListGroup(flank_list_items, flush=True)
    ], className="match-summary-card")

    # Card 3: Buildup Type
    type_list_items = []
    for b_type, count in stats['types'].items():
        if count == 0 and not (active_filter and 'types' in active_filter and active_filter['types'] == b_type):
            continue
        if active_filter and 'types' in active_filter and active_filter['types'] != b_type:
            continue
        active = is_active('types', b_type)
        type_list_items.append(
            dbc.ListGroupItem(
                [
                    dash_html.Div(b_type),
                    dbc.Badge(f"{count} ({count / stats['total']:.0%})", color="light", text_color="dark", className="ms-auto")
                ],
                id={'type': 'buildup-filter', 'filter_type': 'types', 'value': b_type},
                action=True,
                n_clicks=0,
                active=active,
                className=f"match-summary-row {'is-active' if active else ''}"
            )
        )
    type_card = dbc.Card([
        dbc.CardHeader([dash_html.I(className="fa-solid fa-route me-2"), "Initial buildup type"]),
        dbc.ListGroup(type_list_items, flush=True)
    ], className="match-summary-card")

    return dbc.Row([
        dbc.Col(outcome_card, md=4),
        dbc.Col(flank_card, md=4),
        dbc.Col(type_card, md=4)
    ], className="g-3 match-summary-grid")
