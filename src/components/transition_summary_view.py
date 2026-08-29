"""Dash presentation components for transition summaries."""

import dash_bootstrap_components as dbc
from dash import html as dash_html, dash_table


def generate_transition_profile_table(df):
    if df.empty:
        return dbc.Alert("No transition profile data available.", color="secondary")

    df_display = df.rename(columns={
        "Loss_Zone": "Loss Zone",
        "Flank": "Flank",
        "Avg_Duration": "⏱ Avg Duration (s)",
        "Avg_Passes": "🔁 Avg Passes",
        "Num_Sequences": "🔢 Sequences"
    })

    return dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in df.columns],
        data=df.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={
            'padding': '5px',
            'textAlign': 'center',
            'fontFamily': 'Arial',
            'backgroundColor': '#f8f9fa'
        },
        style_header={
            'fontWeight': 'bold',
            'backgroundColor': '#dee2e6'
        }
    )


def create_def_transition_summary_cards(stats, active_filter=None):
    if not stats or stats.get("total", 0) == 0:
        return dbc.Alert("No summary data to display.", color="secondary")

    def is_active(filter_type, value):
        if not active_filter:
            return False
        return str(active_filter.get(filter_type)) == str(value)

    # Card 1: Outcomes
    outcome_order = ['Goals conceded', 'Own Goal Conceded', 'Forced Own Goal', 'Penalty conceded', 'Shots conceded', 'Big Chances conceded', 'Regained Possessions', 'Out', 'Offside', 'Foul']
    outcome_rank = {v: i for i, v in enumerate(outcome_order)}
    outcome_items = sorted(stats['outcomes'].items(), key=lambda x: outcome_rank.get(x[0], 99))

    outcome_list = []
    for outcome, count in outcome_items:
        if count == 0 and not is_active('outcomes', outcome):
            continue
        if active_filter and active_filter.get('outcomes') and active_filter['outcomes'] != outcome:
            continue
        active = is_active('outcomes', outcome)
        outcome_list.append(
            dbc.ListGroupItem(
                [dash_html.Div(outcome),
                 dbc.Badge(f"{count} ({count / stats['total']:.0%})", className="ms-auto")],
                id={'type': 'def-filter', 'filter_type': 'outcomes', 'value': outcome},
                action=True, n_clicks=0,
                active=active,
                className="d-flex justify-content-between align-items-center"
            )
        )
    outcome_card = dbc.Card([
        dbc.CardHeader("Conceded Outcomes"),
        dbc.ListGroup(outcome_list, flush=True)
    ], className="mb-3")

    # Card 2: Dominant Flank
    flank_list = []
    for flank, count in stats['flanks'].items():
        if count == 0 and not is_active('flanks', flank):
            continue
        if active_filter and active_filter.get('flanks') and active_filter['flanks'] != flank:
            continue
        active = is_active('flanks', flank)
        flank_list.append(
            dbc.ListGroupItem(
                [dash_html.Div(flank),
                 dbc.Badge(f"{count} ({count / stats['total']:.0%})", className="ms-auto")],
                id={'type': 'def-filter', 'filter_type': 'flanks', 'value': flank},
                action=True, n_clicks=0,
                active=active,
                className="d-flex justify-content-between align-items-center"
            )
        )
    flank_card = dbc.Card([dbc.CardHeader("Counterattack Side"), dbc.ListGroup(flank_list, flush=True)], className="mb-3")

    # Card 3: Type of Loss
    type_list = []
    for loss_type, count in stats['types'].items():
        if count == 0 and not is_active('types', loss_type):
            continue
        if active_filter and active_filter.get('types') and active_filter['types'] != loss_type:
            continue
        active = is_active('types', loss_type)
        type_list.append(
            dbc.ListGroupItem(
                [dash_html.Div(loss_type),
                 dbc.Badge(f"{count} ({count / stats['total']:.0%})", className="ms-auto")],
                id={'type': 'def-filter', 'filter_type': 'types', 'value': loss_type},
                action=True, n_clicks=0,
                active=active,
                className="d-flex justify-content-between align-items-center"
            )
        )
    type_card = dbc.Card([dbc.CardHeader("Type of Loss"), dbc.ListGroup(type_list, flush=True)], className="mb-3")

    return dbc.Row([
        dbc.Col(outcome_card, md=4),
        dbc.Col(flank_card, md=4),
        dbc.Col(type_card, md=4)
    ])


def create_off_transition_summary_cards(stats, active_filter=None):
    """
    Crea le card interattive per il riepilogo delle transizioni offensive.
    """
    if not stats or stats.get("total", 0) == 0:
        return dbc.Alert("No summary data to display.", color="secondary")

    def is_active(filter_type, value):
        return active_filter is not None and str(active_filter.get(filter_type)) == str(value)

    # Card 1: Outcomes
    outcome_order = ['Goals', 'Forced Own Goal', 'Penalty won', 'Shots', 'Big Chances', 'Lost Possessions', 'Out', 'Offside', 'Foul']
    outcome_rank = {v: i for i, v in enumerate(outcome_order)}
    outcome_items = sorted(stats.get('outcomes', {}).items(), key=lambda x: outcome_rank.get(x[0], 99))

    outcome_list = [
        dbc.ListGroupItem(
            [dash_html.Div(outcome), dbc.Badge(f"{count} ({count / stats['total']:.0%})", className="ms-auto")],
            id={'type': 'off-filter', 'filter_type': 'outcomes', 'value': outcome}, # ID cambiato
            action=True, n_clicks=0, active=is_active('outcomes', outcome),
            className="d-flex justify-content-between align-items-center"
        ) for outcome, count in outcome_items
    ]
    outcome_card = dbc.Card([dbc.CardHeader("Attack Outcomes"), dbc.ListGroup(outcome_list, flush=True)], className="mb-3")

    # Card 2: Flank
    flank_list = [
        dbc.ListGroupItem(
            [dash_html.Div(flank), dbc.Badge(f"{count} ({count / stats['total']:.0%})", className="ms-auto")],
            id={'type': 'off-filter', 'filter_type': 'flanks', 'value': flank}, # ID cambiato
            action=True, n_clicks=0, active=is_active('flanks', flank),
            className="d-flex justify-content-between align-items-center"
        ) for flank, count in stats.get('flanks', {}).items()
    ]
    flank_card = dbc.Card([dbc.CardHeader("Attack Side"), dbc.ListGroup(flank_list, flush=True)], className="mb-3")

    # Card 3: Recovery Type
    type_list = [
        dbc.ListGroupItem(
            [dash_html.Div(rec_type), dbc.Badge(f"{count} ({count / stats['total']:.0%})", className="ms-auto")],
            id={'type': 'off-filter', 'filter_type': 'types', 'value': rec_type}, # ID cambiato
            action=True, n_clicks=0, active=is_active('types', rec_type),
            className="d-flex justify-content-between align-items-center"
        ) for rec_type, count in stats.get('types', {}).items()
    ]
    type_card = dbc.Card([dbc.CardHeader("Type of Recovery"), dbc.ListGroup(type_list, flush=True)], className="mb-3")

    return dbc.Row([
        dbc.Col(outcome_card, md=4),
        dbc.Col(flank_card, md=4),
        dbc.Col(type_card, md=4)
    ])


def render_off_transition_profile_table(profile_df):
    """Render the existing update_off_transition_summary_and_heatmap profile table."""
    if profile_df is None or profile_df.empty:
        return (
            dbc.Alert(
                            (
                                "No transition pattern "
                                "data available."
                            ),
                            color="secondary",
                        )
        )

    return (
        dbc.Table.from_dataframe(
                        profile_df,
                        striped=False,
                        bordered=False,
                        hover=True,
                        responsive=True,
                        index=False,
                        className=(
                            "off-transition-pattern-table "
                            "mb-0"
                        ),
                    )
    )
