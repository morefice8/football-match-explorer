"""Dash presentation components for restart summaries."""

import pandas as pd
import dash_bootstrap_components as dbc
from dash import html


def create_set_piece_summary_cards(stats, active_filter=None):
    """Creates a full set of detailed, interactive summary cards for set pieces."""
    if not stats or stats.get("total", 0) == 0:
        return dbc.Alert("No restart data to display.", color="secondary")

    def create_card(title, data_dict, filter_type):
        items = []
        if not data_dict: return None
        data_items = sorted(data_dict.items(), key=lambda item: item[1], reverse=True)
        if filter_type == 'outcome':
            outcome_order = ['Penalty Goal', 'Goals', 'Penalty Saved', 'Penalty Missed', 'Shots', 'Big Chances', 'Lost Possessions', 'Foul']
            outcome_rank = {v: i for i, v in enumerate(outcome_order)}
            # Ri-ordina la lista 'data_items' basandosi sulla gerarchia definita
            data_items = sorted(data_items, key=lambda item: outcome_rank.get(item[0], 99))
        for value, count in data_items:
            active = active_filter is not None and str(active_filter.get(filter_type)) == str(value)
            items.append(dbc.ListGroupItem(
                [html.Div(value), dbc.Badge(f"{count}", color="light", className="ms-auto")],
                id={'type': 'sp-filter', 'filter_type': filter_type, 'value': value},
                action=True, n_clicks=0, active=active,
                className="d-flex justify-content-between align-items-center"
            ))
        return dbc.Col(dbc.Card([dbc.CardHeader(title), dbc.ListGroup(items, flush=True)]), md=4)

    active_filter = active_filter or {}

    # Creiamo una lista di tutte le card che vogliamo visualizzare
    cards = [
        create_card("Action Type", stats.get('action_types'), 'action'),
        create_card("Action Side", stats.get('sides'), 'side'),
        create_card("Delivery Type", stats.get('deliveries'), 'delivery'),
        create_card("Cross Swing", stats.get('swings'), 'swing'),
        create_card("Taker Foot", stats.get('feet'), 'foot'),
        create_card("Corner Cross Destination", stats.get('destinations'), 'destination'),
        create_card("Execution Outcome", stats.get('outcomes'), 'outcome')
    ]

    # Rimuoviamo le card che non hanno dati (es. se non ci sono cross, non mostrare la card "Swing")
    valid_cards = [card for card in cards if card is not None]

    # Dividiamo le card in righe di 3 per una migliore visualizzazione
    rows = [dbc.Row(valid_cards[i:i+3], className="mb-3") for i in range(0, len(valid_cards), 3)]

    return html.Div(rows)


def create_takers_card(df_analyzed, player_jersey_map, active_filter=None):
    """
    Crea una card che mostra i giocatori che hanno battuto i calci piazzati.
    VERSIONE CORRETTA E ROBUSTA.
    """
    if df_analyzed.empty:
        return None

    # REL-09B: taker patterns are meaningful for corners/free kicks,
    # not for a mixed list of throw-ins and goal kicks.
    if 'Action Type' in df_analyzed.columns:
        df_analyzed = df_analyzed[
            df_analyzed['Action Type'].isin(['Corner', 'Free Kick'])
        ].copy()

    if df_analyzed.empty:
        return None

    # Conta quanti calci piazzati ha battuto ogni giocatore e prende il piede
    # (assumendo che un giocatore usi prevalentemente lo stesso piede per i calci piazzati)
    takers_summary = df_analyzed.groupby('playerName').agg(
        Count=('playerName', 'size'),
        Foot=('Foot', lambda x: x.mode()[0] if not x.mode().empty else 'Unknown') # Usa il piede più frequente
    ).sort_values('Count', ascending=False).reset_index()

    active_filter = active_filter or {}
    card_items = []
    # Mostra al massimo i primi 5-6 tiratori per non affollare la card
    for _, row in takers_summary.head(6).iterrows():
        player_name = row['playerName']
        jersey_number = player_jersey_map.get(player_name)
        jersey_display = f"#{int(jersey_number)}" if pd.notna(jersey_number) else "#?"

        active = active_filter.get('taker') == player_name

        card_items.append(
            dbc.ListGroupItem(
                [
                    html.Div([
                        html.Span(f"{jersey_display}", className="fw-bold me-2", style={'minWidth': '35px', 'display': 'inline-block'}),
                        html.Span(player_name)
                    ]),
                    dbc.Badge(f"{row['Foot']} | ✖ {row['Count']}", color="light", className="ms-auto", pill=True)
                ],
                id={'type': 'sp-filter', 'filter_type': 'taker', 'value': player_name},
                action=True, n_clicks=0, active=active,
                className="d-flex justify-content-between align-items-center"
            )
        )

    if not card_items:
        return None

    return dbc.Col(dbc.Card([dbc.CardHeader("Set-piece Takers"), dbc.ListGroup(card_items, flush=True)]), md=4)
