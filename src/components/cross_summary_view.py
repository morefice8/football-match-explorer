"""Dash presentation components for cross summaries."""

import dash_bootstrap_components as dbc
from dash import html


def create_cross_summary_cards(df_analyzed, active_filter=None):
    """Creates a full set of detailed, interactive summary cards for crosses."""
    if df_analyzed.empty:
        return html.Div()

    # La funzione helper interna ora è corretta
    def create_card(title, data_dict, filter_type):
        items = []
        if not data_dict: return None
        sorted_items = sorted(data_dict.items(), key=lambda item: item[1], reverse=True)
        # Usiamo l'active_filter definito nello scope esterno
        for value, count in sorted_items:
            active = active_filter is not None and str(active_filter.get(filter_type)) == str(value)

            items.append(dbc.ListGroupItem(
                [html.Div(value), dbc.Badge(f"{count}", color="light", className="ms-auto")],
                id={'type': 'cross-filter', 'filter_type': filter_type, 'value': value},
                action=True, n_clicks=0, active=active,
                className="d-flex justify-content-between align-items-center"
            ))
        return dbc.Card(
            [
                dbc.CardHeader(
                    title,
                    className="cross-filter-card-header",
                ),

                dbc.ListGroup(
                    items,
                    flush=True,
                ),
            ],
            className="cross-filter-card",
        )

    stats = {
        'origin': df_analyzed['Origin Zone'].value_counts().to_dict(),
        'destination': df_analyzed['Destination Zone'].value_counts().to_dict(),
        'swing': df_analyzed[df_analyzed['Swing'] != 'N/A']['Swing'].value_counts().to_dict(),
        'outcome': df_analyzed['Outcome'].value_counts().to_dict(),
        'feet': df_analyzed[df_analyzed['Foot'] != 'Unknown']['Foot'].value_counts().to_dict(),
        'takers': df_analyzed['playerName'].value_counts().to_dict(),
        'play_type': df_analyzed['Play Type'].value_counts().to_dict()
    }

    active_filter = active_filter or {}

    # --- CHIAMATE CORRETTE (con 3 argomenti) ---
    cards = [
        create_card("Origin Zone", stats.get('origin'), 'origin'),
        create_card("Destination Zone", stats.get('destination'), 'destination'),
        create_card("Play Type", stats.get('play_type'), 'play_type'),
        create_card("Swing Type", stats.get('swing'), 'swing'),
        create_card("Taker Foot", stats.get('feet'), 'foot'),
        create_card("Outcome", stats.get('outcome'), 'outcome'),
    ]

    # Creiamo la card dei crossatori a parte per gestire il filter_type 'taker'
    takers_stats = stats.get('takers')
    if takers_stats:
        takers_items = []
        # Mostra solo i primi 5 per non affollare
        for player, count in sorted(takers_stats.items(), key=lambda item: item[1], reverse=True)[:5]:
            active = active_filter.get('taker') == player
            takers_items.append(dbc.ListGroupItem(
                [html.Div(player), dbc.Badge(f"{count}", color="light", className="ms-auto")],
                id={'type': 'cross-filter', 'filter_type': 'taker', 'value': player},
                action=True, n_clicks=0, active=active,
                className="d-flex justify-content-between align-items-center"
            ))
        cards.append(
            dbc.Card(
                [
                    dbc.CardHeader(
                        "Top Crossers",
                        className="cross-filter-card-header",
                    ),

                    dbc.ListGroup(
                        takers_items,
                        flush=True,
                    ),
                ],
                className="cross-filter-card",
            )
        )

    valid_cards = [
        card
        for card in cards
        if card is not None
    ]

    return html.Div(
        valid_cards,
        className="cross-filter-grid",
    )
