# src/metrics/cross_metrics.py
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html

def analyze_crosses(df_processed, team_name):
    """
    Identifies all crosses for a team and enriches them with detailed metrics.
    """
    # Filtra tutti gli eventi di tipo "Pass" con il qualifier "cross"
    crosses_df = df_processed[
        (df_processed['team_name'] == team_name) &
        (df_processed['type_name'] == 'Pass') &
        (df_processed['cross'] == 1)
    ].copy()

    if crosses_df.empty:
        return pd.DataFrame()

    analyzed_data = []

    # Determina la zona di origine e destinazione per ogni cross
    def get_pitch_zone(x, y):
        if y > 67:
            side = "Left"
        elif y < 33:
            side = "Right"
        else:
            side = "Center"

        if x > 80:
            area = "Deep"
        elif x > 60:
            area = "Advanced"
        else:
            area = "Midfield"

        return f"{side} {area}"

    for _, cross in crosses_df.iterrows():
        play_type = "Open Play"
        if cross.get('Corner taken') == 1:
            play_type = "From Corner"
        elif cross.get('Freekick taken') == 1:
            play_type = "From Free Kick"
        foot = 'Right' if cross.get('Right footed') == 1 else ('Left' if cross.get('Left footed') == 1 else 'Unknown')
        swing = 'N/A'
        if cross.get('In-swinger') == 1: swing = 'In-swinger'
        elif cross.get('Out-swinger') == 1: swing = 'Out-swinger'
        elif cross.get('Straight') == 1: swing = 'Straight'

        origin_zone = get_pitch_zone(cross['x'], cross['y'])
        destination_zone = get_pitch_zone(cross['end_x'], cross['end_y'])

        # Semplificazione dell'outcome
        outcome = (
            "Completed"
            if cross['outcome'] == 'Successful'
            else "Incomplete"
        )
        # Potremmo aggiungere una logica per 'Shot' o 'Goal' se analizziamo la sequenza successiva,
        # per ora ci limitiamo al successo del cross stesso.

        analyzed_data.append({
            'cross_id': cross['eventId'],
            'playerName': cross['playerName'],
            'Play Type': play_type,
            'Foot': foot,
            'Swing': swing,
            'Origin Zone': origin_zone,
            'Destination Zone': destination_zone,
            'Outcome': outcome,
            'x': cross['x'],
            'y': cross['y'],
            'end_x': cross['end_x'],
            'end_y': cross['end_y']
        })

    return pd.DataFrame(analyzed_data)

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

def build_cross_flow_profile(
    df_analyzed,
    limit=8,
):
    """
    Summarise the most common cross pathways.

    A pathway is:
        Origin Zone -> Destination Zone

    Completion refers to the Opta outcome of the
    cross/pass event itself.

    Percentages use all currently filtered crosses
    as denominator.
    """

    default_summary = {
        'total_crosses': 0,
        'top_route': 'N/A',
        'top_route_count': 0,
        'top_route_pct': 0.0,
        'top_three_pct': 0.0,
    }

    columns = [
        'Origin Zone',
        'Destination Zone',
        'Crosses',
        'Share %',
        'Completed',
        'Completion %',
    ]

    if (
        df_analyzed is None
        or df_analyzed.empty
    ):
        return (
            default_summary,
            pd.DataFrame(columns=columns),
        )

    df = df_analyzed.copy()

    total_crosses = len(df)

    # ---------------------------------------------------------
    # ROUTE VOLUME
    # ---------------------------------------------------------

    routes = (
        df.groupby(
            [
                'Origin Zone',
                'Destination Zone',
            ],
            dropna=False,
        )
        .size()
        .reset_index(name='Crosses')
    )

    # ---------------------------------------------------------
    # COMPLETED CROSSES BY ROUTE
    # ---------------------------------------------------------

    completed = (
        df[
            df['Outcome'] == 'Completed'
        ]
        .groupby(
            [
                'Origin Zone',
                'Destination Zone',
            ],
            dropna=False,
        )
        .size()
        .reset_index(name='Completed')
    )

    routes = routes.merge(
        completed,
        on=[
            'Origin Zone',
            'Destination Zone',
        ],
        how='left',
    )

    routes['Completed'] = (
        routes['Completed']
        .fillna(0)
        .astype(int)
    )

    routes['Share %'] = (
        routes['Crosses']
        / total_crosses
        * 100
    )

    routes['Completion %'] = (
        routes['Completed']
        / routes['Crosses']
        * 100
    )

    routes = (
        routes
        .sort_values(
            [
                'Crosses',
                'Completed',
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    if routes.empty:
        return (
            default_summary,
            pd.DataFrame(columns=columns),
        )

    top_route = routes.iloc[0]

    top_three_count = int(
        routes
        .head(3)['Crosses']
        .sum()
    )

    summary = {
        'total_crosses':
            int(total_crosses),

        'top_route': (
            f"{top_route['Origin Zone']} → "
            f"{top_route['Destination Zone']}"
        ),

        'top_route_count':
            int(top_route['Crosses']),

        'top_route_pct':
            float(top_route['Share %']),

        'top_three_pct': (
            top_three_count
            / total_crosses
            * 100
        ),
    }

    return (
        summary,
        routes
        .head(limit)[columns]
        .copy(),
    )