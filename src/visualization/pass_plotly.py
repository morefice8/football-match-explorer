# in src/visualization/pitch_plots.py
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from src.visualization import pitch_plots
from src.visualization.plotly_branding import (
    add_plot_header,
    apply_dark_pitch_layout,
    add_attacking_direction,
)
from src.visualization.plotly_branding import (
    MATCH_COMPARE_HEIGHT,
    MATCH_WARNING,
    add_attacking_direction,
    add_zero_state,
    apply_match_pitch_layout,
    get_team_palette,
)

# def plot_pass_network_plotly(passes_between, avg_locs, team_name, team_color, sub_list, is_away=False):
#     """
#     Versione 5: Corregge la visualizzazione delle linee e migliora lo stile dei subentrati.
#     """
#     fig = go.Figure()
#     pitch_shapes = pitch_plots.get_plotly_pitch_shapes("rgba(255,255,255,0.2)", "white")

#     if avg_locs.empty:
#         fig.add_annotation(text=f"No pass network data for {team_name}", showarrow=False, font=dict(size=16, color="orange"))
#     else:
#         # Copia i DataFrame per evitare SettingWithCopyWarning
#         avg_locs = avg_locs.copy()
#         passes_between = passes_between.copy()

#         # Inverti le coordinate per il team away
#         if is_away:
#             avg_locs[['pass_avg_x', 'pass_avg_y']] = 100 - avg_locs[['pass_avg_x', 'pass_avg_y']]
#             if not passes_between.empty:
#                 passes_between[['pass_avg_x', 'pass_avg_y', 'pass_avg_x_end', 'pass_avg_y_end']] = 100 - passes_between[['pass_avg_x', 'pass_avg_y', 'pass_avg_x_end', 'pass_avg_y_end']]

#         # --- 1. Disegna le linee delle connessioni (se esistono) ---
#         if not passes_between.empty:
#             max_lw = 10
#             max_count = passes_between['pass_count'].max() if not passes_between.empty else 1
#             passes_between['linewidth'] = passes_between['pass_count'] / max_count * max_lw

#             mid_x, mid_y, hover_texts = [], [], []
#             for _, row in passes_between.iterrows():
#                 fig.add_trace(go.Scatter(
#                     x=[row['pass_avg_x'], row['pass_avg_x_end']],
#                     y=[row['pass_avg_y'], row['pass_avg_y_end']],
#                     mode='lines',
#                     line=dict(width=row['linewidth'], color=team_color, shape='spline'),
#                     opacity=0.6,
#                     hoverinfo='none',
#                     showlegend=False
#                 ))
#                 mid_x.append((row['pass_avg_x'] + row['pass_avg_x_end']) / 2)
#                 mid_y.append((row['pass_avg_y'] + row['pass_avg_y_end']) / 2)
#                 hover_texts.append(f"{row['player1']} <> {row['player2']}<br><b>{int(row['pass_count'])}</b> passes")

#             fig.add_trace(go.Scatter(
#                 x=mid_x, y=mid_y, mode='markers',
#                 marker=dict(color=team_color, size=5, opacity=0),
#                 hoverinfo='text', hovertext=hover_texts, showlegend=False
#             ))

#         # --- 2. Disegna i nodi (giocatori) ---
#         max_size = 60
#         max_pass_count = avg_locs['pass_count'].max() if not avg_locs.empty else 1
#         avg_locs['marker_size'] = avg_locs['pass_count'] / max_pass_count * max_size + 20

#         starters_df = avg_locs[~avg_locs['playerName'].isin(sub_list)]
#         subs_df = avg_locs[avg_locs['playerName'].isin(sub_list)]

#         # Aggiungi i titolari (cerchi con bordo bianco)
#         if not starters_df.empty:
#             fig.add_trace(go.Scatter(
#                 x=starters_df['pass_avg_x'], y=starters_df['pass_avg_y'],
#                 mode='markers+text',
#                 text=[f"<b>{int(j)}</b>" if pd.notna(j) else '' for j in starters_df['jersey_number']],
#                 marker=dict(symbol='circle', color=team_color, size=starters_df['marker_size'], line=dict(width=2, color='white')),
#                 hovertext=starters_df['playerName'] + '<br>Passes made: ' + starters_df['pass_count'].astype(int).astype(str),
#                 hoverinfo='text', showlegend=False
#             ))

#         # Aggiungi i subentrati (diamanti con bordo giallo)
#         if not subs_df.empty:
#             fig.add_trace(go.Scatter(
#                 x=subs_df['pass_avg_x'], y=subs_df['pass_avg_y'],
#                 mode='markers+text',
#                 text=[f"<b>{int(j)}</b>" if pd.notna(j) else '' for j in subs_df['jersey_number']],
#                 marker=dict(
#                     symbol='diamond',  # Simbolo diverso
#                     color=team_color,
#                     opacity=0.9, # Leggermente più opaco
#                     size=subs_df['marker_size'],
#                     line=dict(width=3, color='#FFFF00') # Bordo giallo e più spesso
#                 ),
#                 hovertext=subs_df['playerName'] + ' (sub)<br>Passes made: ' + subs_df['pass_count'].astype(int).astype(str),
#                 hoverinfo='text', showlegend=False
#             ))

#     # --- Layout Finale ---
#     fig.update_layout(
#         title=f"Pass Network - {team_name}",
#         showlegend=False,
#         shapes=pitch_shapes,
#         xaxis=dict(range=[-2, 102], showgrid=False, zeroline=False, showticklabels=False),
#         yaxis=dict(range=[-2, 102], showgrid=False, zeroline=False, showticklabels=False),
#         plot_bgcolor='#2E3439',
#         paper_bgcolor='#2E3439',
#         font_color='white',
#         height=700
#     )
#     return fig

def plot_pass_network_plotly(
    passes_between,
    avg_locs,
    team_name,
    team_color,
    sub_list,
    is_away=False,
):
    """
    Plot a team's passing network.

    Node size represents passing volume.
    Connection strength represents combined passes between
    the two players, irrespective of direction.
    """

    fig = go.Figure()

    pitch_shapes = pitch_plots.get_plotly_pitch_shapes(
        "rgba(255,255,255,0.20)",
        "rgba(255,255,255,0.72)",
    )

    if avg_locs is None or avg_locs.empty:
        fig.add_annotation(
            x=50,
            y=50,
            text=f"No pass network data for {team_name}",
            showarrow=False,
            font=dict(
                size=15,
                color='rgba(255,255,255,0.72)',
            ),
        )

    else:
        avg_locs = avg_locs.copy()

        passes_between = (
            passes_between.copy()
            if passes_between is not None
            else pd.DataFrame()
        )

        # -----------------------------------------------------
        # VISUAL ORIENTATION
        # -----------------------------------------------------
        # Opta coordinates are already normalised so that
        # every team attacks towards x=100.
        # Do not mirror the away team.
        del is_away

        # -----------------------------------------------------
        # CONNECTIONS
        # -----------------------------------------------------
        if not passes_between.empty:

            passes_between['pass_count'] = pd.to_numeric(
                passes_between['pass_count'],
                errors='coerce',
            ).fillna(0)

            passes_between = (
                passes_between
                .sort_values(
                    'pass_count',
                    ascending=True,
                )
            )

            max_count = max(
                float(
                    passes_between[
                        'pass_count'
                    ].max()
                ),
                1.0,
            )

            hover_x = []
            hover_y = []
            hover_text = []

            for _, row in passes_between.iterrows():

                count = float(
                    row['pass_count']
                )

                strength = (
                    count / max_count
                ) ** 1.45

                line_width = (
                    0.75
                    + 4.25 * strength
                )

                line_opacity = (
                    0.10
                    + 0.68 * strength
                )

                fig.add_trace(
                    go.Scatter(
                        x=[
                            row['pass_avg_x'],
                            row['pass_avg_x_end'],
                        ],
                        y=[
                            row['pass_avg_y'],
                            row['pass_avg_y_end'],
                        ],
                        mode='lines',

                        line=dict(
                            width=line_width,
                            color=team_color,
                        ),

                        opacity=line_opacity,

                        hoverinfo='skip',
                        showlegend=False,
                    )
                )

                hover_x.append(
                    (
                        row['pass_avg_x']
                        + row['pass_avg_x_end']
                    ) / 2
                )

                hover_y.append(
                    (
                        row['pass_avg_y']
                        + row['pass_avg_y_end']
                    ) / 2
                )

                hover_text.append(
                    (
                        f"<b>{row['player1']}"
                        f" ↔ {row['player2']}</b>"
                        f"<br>{int(count)} combined passes"
                    )
                )

            # Invisible hover targets for connections.
            fig.add_trace(
                go.Scatter(
                    x=hover_x,
                    y=hover_y,

                    mode='markers',

                    marker=dict(
                        size=14,
                        opacity=0,
                    ),

                    text=hover_text,

                    hovertemplate=(
                        "%{text}"
                        "<extra></extra>"
                    ),

                    showlegend=False,
                )
            )

        # -----------------------------------------------------
        # PLAYER NODES
        # -----------------------------------------------------
        avg_locs['pass_count'] = pd.to_numeric(
            avg_locs['pass_count'],
            errors='coerce',
        ).fillna(0)

        max_pass_count = max(
            float(
                avg_locs['pass_count'].max()
            ),
            1.0,
        )

        avg_locs['marker_size'] = (
            22
            + 30
            * np.sqrt(
                avg_locs['pass_count']
                / max_pass_count
            )
        )

        starters_df = avg_locs[
            ~avg_locs[
                'playerName'
            ].isin(sub_list)
        ]

        subs_df = avg_locs[
            avg_locs[
                'playerName'
            ].isin(sub_list)
        ]

        def jersey_labels(df_players):
            labels = []

            for value in df_players[
                'jersey_number'
            ]:
                try:
                    labels.append(
                        f"<b>{int(float(value))}</b>"
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    labels.append("")

            return labels

        # Starters.
        if not starters_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=starters_df['pass_avg_x'],
                    y=starters_df['pass_avg_y'],

                    mode='markers+text',

                    text=jersey_labels(
                        starters_df
                    ),

                    textposition='middle center',

                    textfont=dict(
                        color='white',
                        size=11,
                    ),

                    marker=dict(
                        symbol='circle',
                        color=team_color,
                        size=starters_df[
                            'marker_size'
                        ],
                        opacity=0.94,

                        line=dict(
                            width=1.5,
                            color=(
                                'rgba(255,255,255,0.82)'
                            ),
                        ),
                    ),

                    customdata=np.column_stack([
                        starters_df[
                            'playerName'
                        ],
                        starters_df[
                            'pass_count'
                        ],
                    ]),

                    hovertemplate=(
                        "<b>%{customdata[0]}</b>"
                        "<br>Passes made: "
                        "%{customdata[1]:.0f}"
                        "<br>Starter"
                        "<extra></extra>"
                    ),

                    showlegend=False,
                )
            )

        # Substitutes.
        if not subs_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=subs_df['pass_avg_x'],
                    y=subs_df['pass_avg_y'],

                    mode='markers+text',

                    text=jersey_labels(
                        subs_df
                    ),

                    textposition='middle center',

                    textfont=dict(
                        color='white',
                        size=10,
                    ),

                    marker=dict(
                        symbol='diamond',
                        color=team_color,
                        size=subs_df[
                            'marker_size'
                        ] * 0.88,
                        opacity=0.90,

                        line=dict(
                            width=1.8,
                            color='#d9c98c',
                        ),
                    ),

                    customdata=np.column_stack([
                        subs_df[
                            'playerName'
                        ],
                        subs_df[
                            'pass_count'
                        ],
                    ]),

                    hovertemplate=(
                        "<b>%{customdata[0]}</b>"
                        "<br>Passes made: "
                        "%{customdata[1]:.0f}"
                        "<br>Substitute"
                        "<extra></extra>"
                    ),

                    showlegend=False,
                )
            )

    add_attacking_direction(fig, dark=True)

    # ---------------------------------------------------------
    # LAYOUT
    # ---------------------------------------------------------
    fig.update_layout(
        shapes=pitch_shapes,

        xaxis=dict(
            range=[-2, 102],
            visible=False,
            fixedrange=True,
        ),

        yaxis=dict(
            range=[-5, 107],
            visible=False,
            fixedrange=True,
        ),
    )

    apply_dark_pitch_layout(
        fig,
        height=610,
        top_margin=120,
        showlegend=False,
    )

    add_plot_header(
        fig,
        title=f"{team_name} · Pass network",
        subtitle=(
            "Node size = pass volume · "
            "line strength = combined connection volume · "
            "diamonds = substitutes"
        ),
        dark=True,
    )

    return fig


def plot_progressive_passes_plotly(
    df_prog_passes,
    team_name,
    team_color,
    is_away=False,
):
    """
    FOUND-02 demonstration plot.

    Metric semantics are unchanged. Only presentation is migrated to the
    shared Match Plot Design System.
    """
    del team_color

    palette = get_team_palette(
        is_away=is_away
    )

    fig = go.Figure()

    pitch_shapes = pitch_plots.get_plotly_pitch_shapes(
        "rgba(255,255,255,0.24)",
        "rgba(255,255,255,0.82)",
    )

    pitch_shapes.extend([
        dict(
            type="line",
            x0=0,
            y0=100 / 3,
            x1=100,
            y1=100 / 3,
            line=dict(
                color="rgba(255,255,255,0.18)",
                dash="dot",
                width=1,
            ),
        ),
        dict(
            type="line",
            x0=0,
            y0=200 / 3,
            x1=100,
            y1=200 / 3,
            line=dict(
                color="rgba(255,255,255,0.18)",
                dash="dot",
                width=1,
            ),
        ),
    ])

    df_plot = df_prog_passes.copy()

    if "is_progressive_attempt" in df_plot.columns:
        df_plot = df_plot[
            df_plot[
                "is_progressive_attempt"
            ]
            .fillna(False)
            .astype(bool)
        ].copy()

    outcomes = (
        (
            "Completed",
            True,
            palette["primary"],
            "solid",
            2.4,
            "circle",
        ),
        (
            "Incomplete",
            False,
            MATCH_WARNING,
            "dot",
            1.45,
            "x",
        ),
    )

    for (
        label,
        completed,
        color,
        dash,
        width,
        marker_symbol,
    ) in outcomes:
        if (
            df_plot.empty
            or "is_progressive"
            not in df_plot.columns
        ):
            subset = pd.DataFrame()
        else:
            subset = df_plot[
                df_plot[
                    "is_progressive"
                ]
                .fillna(False)
                .astype(bool)
                .eq(completed)
            ]

        if subset.empty:
            continue

        x_coords = []
        y_coords = []
        hover_texts = []

        for _, row in subset.iterrows():
            minute = row.get(
                "timeMin",
                "?",
            )

            gained = pd.to_numeric(
                row.get(
                    "progressive_distance_m"
                ),
                errors="coerce",
            )

            threshold = pd.to_numeric(
                row.get(
                    "progressive_threshold_m"
                ),
                errors="coerce",
            )

            gained_label = (
                f"{gained:.1f} m"
                if pd.notna(gained)
                else "N/A"
            )

            threshold_label = (
                f"{threshold:.0f} m"
                if pd.notna(threshold)
                else "N/A"
            )

            hover = (
                f"<b>{row.get('playerName', 'Unknown')}</b>"
                f"<br>{label} progressive pass"
                f"<br>Minute: {minute}'"
                f"<br>Progression towards goal: {gained_label}"
                f"<br>Required threshold: {threshold_label}"
                f"<br>Phase: {row.get('progressive_phase', 'N/A')}"
                f"<br>Origin channel: "
                f"{row.get('progressive_channel', 'N/A')}"
            )

            x_coords.extend([
                row["x"],
                row["end_x"],
                None,
            ])

            y_coords.extend([
                row["y"],
                row["end_y"],
                None,
            ])

            hover_texts.extend([
                hover,
                hover,
                None,
            ])

        fig.add_trace(
            go.Scattergl(
                x=x_coords,
                y=y_coords,
                mode="lines",
                line=dict(
                    color=color,
                    width=width,
                    dash=dash,
                ),
                opacity=(
                    0.82
                    if completed
                    else 0.58
                ),
                name=(
                    f"{label} "
                    f"({len(subset)})"
                ),
                hoverinfo="text",
                hovertext=hover_texts,
            )
        )

        fig.add_trace(
            go.Scattergl(
                x=subset["end_x"],
                y=subset["end_y"],
                mode="markers",
                marker=dict(
                    size=(
                        6
                        if completed
                        else 7
                    ),
                    color=color,
                    symbol=marker_symbol,
                    line=dict(
                        color=(
                            "rgba(255,255,255,0.75)"
                        ),
                        width=0.7,
                    ),
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    if df_plot.empty:
        add_zero_state(
            fig,
            "No open-play progressive attempts",
            dark_pitch=True,
        )

    apply_match_pitch_layout(
        fig,
        pitch_shapes=pitch_shapes,
        height=MATCH_COMPARE_HEIGHT,
        showlegend=True,
        header=False,
        x_range=(-2, 102),
        y_range=(-5, 107),
    )

    # The surrounding Dash card already owns team name, sample size and title.
    # Keep the figure free of a second title.
    add_attacking_direction(
        fig,
        dark=True,
        x=0.985,
        y=0.025,
        xanchor="right",
        yanchor="bottom",
    )

    return fig

def plot_final_third_plotly(df_zone14, df_lhs, df_rhs, stats, team_name, team_color, zone14_color='orange', is_away=False):
    """
    Crea una mappa interattiva dei passaggi nel terzo finale (Zone14 e Half-Spaces).
    """
    fig = go.Figure()
    pitch_shapes = pitch_plots.get_plotly_pitch_shapes("rgba(255,255,255,0.2)", "white")

    # --- Definizioni delle zone e dei dati ---
    # Usiamo gli stessi colori e nomi per i dati e le forme
    zones = {
        "Zone 14": (df_zone14, zone14_color),
        "Left Half-Space": (df_lhs, team_color),
        "Right Half-Space": (df_rhs, team_color)
    }

    zone_key_map = {
        "Zone 14": "zone14",
        "Left Half-Space": "hs_left",
        "Right Half-Space": "hs_right"
    }

    # --- Disegna le zone colorate sul campo ---
    zone_shapes = [
        # Zone 14
        dict(type="rect", x0=66.7, y0=33.3, x1=83.3, y1=66.7, fillcolor=zone14_color, opacity=0.2, layer="below", line_width=0),
        # Left HS
        dict(type="rect", x0=66.7, y0=66.7, x1=100, y1=83.3, fillcolor=team_color, opacity=0.2, layer="below", line_width=0),
        # Right HS
        dict(type="rect", x0=66.7, y0=16.7, x1=100, y1=33.3, fillcolor=team_color, opacity=0.2, layer="below", line_width=0)
    ]

    # --- Disegna le frecce per ogni zona ---
    for zone_name, (zone_df, color) in zones.items():
        if not zone_df.empty:
            x_coords, y_coords, hover_texts = [], [], []
            for _, p in zone_df.iterrows():
                x_coords.extend([p['x'], p['end_x'], None])
                y_coords.extend([p['y'], p['end_y'], None])
                receiver = p.get('receiver')
                receiver_label = receiver if pd.notna(receiver) else 'Unresolved receiver'
                confidence = p.get('receiver_confidence')
                confidence_label = (
                    f"<br>Receiver confidence: {str(confidence).title()}"
                    if pd.notna(confidence) else ''
                )
                hover_text = (
                    f"<b>{p['playerName']}</b> to {receiver_label}"
                    f"<br>Min {p.get('timeMin', '?')}'{confidence_label}"
                )
                hover_texts.extend([hover_text, hover_text, None])

            fig.add_trace(go.Scattergl(
                x=x_coords, y=y_coords, mode='lines',
                line=dict(color=color, width=2),
                # name=f"{zone_name} ({stats.get(zone_name.lower().replace(' ', '_'), 0)})",
                name=f"{zone_name} ({stats.get(zone_key_map[zone_name], 0)})",
                hoverinfo='text', hovertext=hover_texts
            ))
            fig.add_trace(go.Scattergl(
                x=zone_df['end_x'], y=zone_df['end_y'], mode='markers',
                marker=dict(size=5, color=color), showlegend=False, hoverinfo='none'
            ))

    # --- Aggiungi Annotazioni con i conteggi totali ---
    annotations = [
        dict(x=75, y=50, text=f"<b>{stats.get('zone14', 0)}</b>", showarrow=False, font=dict(color='white', size=16)),
        dict(x=91.5, y=75, text=f"<b>{stats.get('hs_left', 0)}</b>", showarrow=False, font=dict(color='white', size=16)),
        dict(x=91.5, y=25, text=f"<b>{stats.get('hs_right', 0)}</b>", showarrow=False, font=dict(color='white', size=16)),
    ]

    # --- Layout Finale ---
    fig.update_layout(
        title=dict(
            text=f"<b>{team_name} - Final Third Entries</b>",
            font=dict(size=16, color='white'),
            x=0.5, y=0.98, xanchor='center', yanchor='top'
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02, # Posiziona la legenda sopra il titolo
            xanchor="center",
            x=0.5,
            font=dict(color='white') # **COLORE LEGENDA BIANCO**
        ),
        shapes=pitch_shapes + zone_shapes,
        annotations=annotations,
        # xaxis=dict(range=[-2, 102], visible=False),
        # yaxis=dict(
        #     range=[-2, 102],
        #     visible=False,
        #     # **MODIFICA CHIAVE: Controlla il rapporto d'aspetto**
        #     # Un valore comune per i campi Opta è 0.68 (100 / 68 * larghezza)
        #     # Aggiustalo se necessario per il tuo layout.
        #     scaleanchor="x",
        #     scaleratio=0.68
        # ),
        xaxis=dict(range=[-2, 102], visible=False),
        yaxis=dict(range=[-2, 102], visible=False),
        plot_bgcolor='#2E3439',
        paper_bgcolor='#2E3439',
        # Rimuovi l'altezza fissa, lascia che si adatti al contenitore
        height=700,
        margin=dict(l=20, r=20, t=80, b=20) # Margine per titolo/legenda
    )

    del is_away
    add_attacking_direction(fig, dark=True)

    return fig

def plot_final_third_entries_plotly(
    entries_df,
    stats,
    team_name,
    team_color,
    is_away=False,
    zone14_color='orange',
    carry_color='#ffb366',
):
    """
    Plot all final-third entries for one team.

    Pass entries are shown with solid lines.
    High-confidence inferred carry entries are shown with dashed lines.
    Zone 14 and the half-spaces remain contextual destination zones,
    rather than defining the metric itself.
    """
    fig = go.Figure()

    pitch_shapes = pitch_plots.get_plotly_pitch_shapes(
        "rgba(255,255,255,0.2)",
        "white",
    )

    final_third_x = 100 * 2 / 3

    # Contextual tactical zones.
    zone_shapes = [
        # Final-third boundary
        dict(
            type="line",
            x0=final_third_x,
            y0=0,
            x1=final_third_x,
            y1=100,
            line=dict(
                color="rgba(255,255,255,0.45)",
                width=1.5,
                dash="dot",
            ),
            layer="below",
        ),

        # Zone 14
        dict(
            type="rect",
            x0=final_third_x,
            y0=100 / 3,
            x1=82.0,
            y1=200 / 3,
            fillcolor=zone14_color,
            opacity=0.12,
            layer="below",
            line_width=0,
        ),

        # Left half-space
        dict(
            type="rect",
            x0=final_third_x,
            y0=200 / 3,
            x1=100,
            y1=500 / 6,
            fillcolor=team_color,
            opacity=0.09,
            layer="below",
            line_width=0,
        ),

        # Right half-space
        dict(
            type="rect",
            x0=final_third_x,
            y0=100 / 6,
            x1=100,
            y1=100 / 3,
            fillcolor=team_color,
            opacity=0.09,
            layer="below",
            line_width=0,
        ),
    ]

    if entries_df is not None and not entries_df.empty:

        # -----------------------------------------------------
        # PASS ENTRIES
        # -----------------------------------------------------
        pass_entries = entries_df[
            entries_df['entry_type'] == 'Pass'
        ].copy()

        if not pass_entries.empty:
            x_coords = []
            y_coords = []
            hover_texts = []

            for _, row in pass_entries.iterrows():
                x_coords.extend([
                    row['x'],
                    row['end_x'],
                    None,
                ])
                y_coords.extend([
                    row['y'],
                    row['end_y'],
                    None,
                ])

                receiver = row.get('receiver')
                if pd.notna(receiver):
                    receiver_label = receiver
                else:
                    receiver_label = 'Unresolved receiver'

                confidence = row.get('receiver_confidence')

                confidence_label = (
                    f"<br>Receiver confidence: "
                    f"{str(confidence).title()}"
                    if pd.notna(confidence)
                    else ''
                )

                hover_text = (
                    f"<b>{row.get('playerName', 'Unknown')}</b>"
                    f" → {receiver_label}"
                    f"<br>Pass entry"
                    f"<br>Min {row.get('timeMin', '?')}'"
                    f"<br>Channel: "
                    f"{row.get('final_third_channel', 'Unknown')}"
                    f"<br>Destination: "
                    f"{row.get('destination_zone', 'Unknown')}"
                    f"{confidence_label}"
                )

                hover_texts.extend([
                    hover_text,
                    hover_text,
                    None,
                ])

            fig.add_trace(go.Scattergl(
                x=x_coords,
                y=y_coords,
                mode='lines',
                line=dict(
                    color=team_color,
                    width=2.2,
                ),
                name=f"Pass entries ({len(pass_entries)})",
                hoverinfo='text',
                hovertext=hover_texts,
            ))

            fig.add_trace(go.Scattergl(
                x=pass_entries['end_x'],
                y=pass_entries['end_y'],
                mode='markers',
                marker=dict(
                    size=6,
                    color=team_color,
                ),
                showlegend=False,
                hoverinfo='none',
            ))

        # -----------------------------------------------------
        # CARRY ENTRIES
        # -----------------------------------------------------
        carry_entries = entries_df[
            entries_df['entry_type'] == 'Carry'
        ].copy()

        if not carry_entries.empty:
            x_coords = []
            y_coords = []
            hover_texts = []

            for _, row in carry_entries.iterrows():
                x_coords.extend([
                    row['x'],
                    row['end_x'],
                    None,
                ])
                y_coords.extend([
                    row['y'],
                    row['end_y'],
                    None,
                ])

                distance = row.get('carry_distance_m')
                distance_label = (
                    f"{distance:.1f} m"
                    if pd.notna(distance)
                    else "Unknown"
                )

                hover_text = (
                    f"<b>{row.get('playerName', 'Unknown')}</b>"
                    f"<br>High-confidence inferred carry entry"
                    f"<br>Distance: {distance_label}"
                    f"<br>Channel: "
                    f"{row.get('final_third_channel', 'Unknown')}"
                    f"<br>Destination: "
                    f"{row.get('destination_zone', 'Unknown')}"
                )

                hover_texts.extend([
                    hover_text,
                    hover_text,
                    None,
                ])

            fig.add_trace(go.Scattergl(
                x=x_coords,
                y=y_coords,
                mode='lines',
                line=dict(
                    color=carry_color,
                    width=2.5,
                    dash='dash',
                ),
                name=(
                    "Inferred carry entries "
                    f"({len(carry_entries)})"
                ),
                hoverinfo='text',
                hovertext=hover_texts,
            ))

            fig.add_trace(go.Scattergl(
                x=carry_entries['end_x'],
                y=carry_entries['end_y'],
                mode='markers',
                marker=dict(
                    size=7,
                    color=carry_color,
                    symbol='diamond',
                ),
                showlegend=False,
                hoverinfo='none',
            ))

    # Tactical-zone counts.
    annotations = [
        dict(
            x=74,
            y=50,
            text=f"<b>Z14<br>{stats.get('zone14', 0)}</b>",
            showarrow=False,
            font=dict(
                color='rgba(255,255,255,0.88)',
                size=12,
            ),
        ),
        dict(
            x=90,
            y=75,
            text=f"<b>LHS<br>{stats.get('hs_left', 0)}</b>",
            showarrow=False,
            font=dict(
                color='rgba(255,255,255,0.84)',
                size=11,
            ),
        ),
        dict(
            x=90,
            y=25,
            text=f"<b>RHS<br>{stats.get('hs_right', 0)}</b>",
            showarrow=False,
            font=dict(
                color='rgba(255,255,255,0.84)',
                size=11,
            ),
        ),
    ]

    if entries_df is None or entries_df.empty:
        annotations.append(
            dict(
                x=82,
                y=50,
                text="<b>No final-third entries</b>",
                showarrow=False,
                font=dict(
                    color='rgba(255,255,255,0.7)',
                    size=16,
                ),
            )
        )

    for annotation in annotations:
        fig.add_annotation(**annotation)

    fig.update_layout(
        shapes=pitch_shapes + zone_shapes,

        xaxis=dict(
            range=[-2, 102],
            visible=False,
            fixedrange=True,
        ),

        yaxis=dict(
            range=[-2, 102],
            visible=False,
            fixedrange=True,
        ),

        legend=dict(
            orientation='h',
            x=0.5,
            xanchor='center',
            y=1.045,
            yanchor='bottom',
            font=dict(
                color='white',
                size=11,
            ),
            bgcolor='rgba(0,0,0,0)',
            traceorder='normal',
        ),
    )

    apply_dark_pitch_layout(
        fig,
        height=660,
        top_margin=135,
        showlegend=True,
    )

    add_plot_header(
        fig,
        title=f"{team_name} · Final Third Entries",
        subtitle=(
            f"{stats.get('total_final_third', 0)} total · "
            f"{stats.get('pass_entries', 0)} pass · "
            f"{stats.get('carry_entries', 0)} inferred carry"
        ),
        dark=True,
    )

    del is_away
    add_attacking_direction(fig, dark=True)

    return fig

def plot_pass_locations_plotly(passes_df, team_name, is_away=False):
    """
    Versione 3: Corregge il disegno del campo su subplot e migliora lo stile.
    """
    # Crea la griglia di subplot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Pass Density (KDE)", "Pass Heatmap")
    )

    # Coordinates are already team-relative. Away only changes palette.
    df_plot = passes_df.copy()
    colorscale = 'Reds' if not is_away else 'Blues'

    if not df_plot.empty:
        # --- Subplot 1: Mappa di Densità (KDE) ---
        fig.add_trace(go.Histogram2dContour(
            x=df_plot['x'], y=df_plot['y'],
            colorscale=colorscale, showscale=False,
            line_width=0, name='Density'
        ), row=1, col=1)
        # Aggiungi i punti dei passaggi con opacità per dare contesto
        fig.add_trace(go.Scatter(
            x=df_plot['x'], y=df_plot['y'],
            mode='markers',
            marker=dict(color='white', size=3, opacity=0.3),
            hoverinfo='none', showlegend=False
        ), row=1, col=1)

        # --- Subplot 2: Heatmap a Griglia ---
        x_bins = np.linspace(0, 100, 7)
        y_bins = np.linspace(0, 100, 6)
        counts, y_edges, x_edges = np.histogram2d(df_plot['y'], df_plot['x'], bins=[y_bins, x_bins])

        fig.add_trace(go.Heatmap(
            z=counts,
            x=(x_edges[:-1] + x_edges[1:]) / 2,
            y=(y_edges[:-1] + y_edges[1:]) / 2,
            colorscale=colorscale,
            colorbar=dict(title='Pass Count', x=1.02)
        ), row=1, col=2)

        # Aggiungi i numeri sopra la heatmap
        for i, row in enumerate(counts):
            for j, val in enumerate(row):
                if val > 0:
                    fig.add_annotation(
                        xref="x2", yref="y2", # Riferimento agli assi del subplot 2
                        x=(x_edges[j] + x_edges[j+1]) / 2,
                        y=(y_edges[i] + y_edges[i+1]) / 2,
                        text=f"<b>{int(val)}</b>",
                        showarrow=False,
                        font=dict(color='white' if val > counts.max() / 2 else 'black', size=10)
                    )

    # --- DISEGNO DEL CAMPO SU ENTRAMBI I SUBPLOT ---
    pitch_shapes = pitch_plots.get_plotly_pitch_shapes("rgba(255,255,255,0.4)", "white")

    for shape in pitch_shapes:
        # Aggiungi la forma a entrambi i subplot specificando il riferimento agli assi
        fig.add_shape(shape, row=1, col=1)
        fig.add_shape(shape, row=1, col=2)

    # --- Layout Finale ---
    fig.update_layout(
        title_text=f"<b>{team_name} - Pass Start Locations</b>",
        title_x=0.5,
        plot_bgcolor='#2E3439', paper_bgcolor='#2E3439',
        font_color='white', height=450,
        margin=dict(l=20, r=60, t=80, b=20),
        showlegend=False
    )
    # Applica le impostazioni degli assi a entrambi i subplot
    fig.update_xaxes(range=[-2, 102], visible=False)
    fig.update_yaxes(range=[-2, 102], visible=False, scaleanchor="x", scaleratio=0.68)

    add_attacking_direction(fig, dark=True)

    return fig

# def plot_pass_density_plotly(passes_df, team_name, is_away=False):
#     """
#     Crea una mappa di densità (KDE) interattiva su un campo da calcio.
#     """
#     fig = go.Figure()
#     pitch_shapes = pitch_plots.get_plotly_pitch_shapes()

#     df_plot = passes_df.copy()
#     if is_away:
#         df_plot['x'] = 100 - df_plot['x']
#         df_plot['y'] = 100 - df_plot['y']

#     colorscale = 'Reds' if not is_away else 'Blues'

#     if not df_plot.empty:
#         fig.add_trace(go.Histogram2dContour(
#             x=df_plot['x'], y=df_plot['y'],
#             colorscale=colorscale, showscale=False, line_width=0, name='Density'
#         ))
#         fig.add_trace(go.Scatter(
#             x=df_plot['x'], y=df_plot['y'], mode='markers',
#             marker=dict(color='white', size=3, opacity=0.3),
#             hoverinfo='none', showlegend=False
#         ))

#     fig.update_layout(
#         title=dict(text=f"{team_name} - Pass Density (KDE)", font_color='white', x=0.5),
#         shapes=pitch_shapes,
#         xaxis=dict(range=[-2, 102], visible=False),
#         yaxis=dict(range=[-2, 102], visible=False, scaleanchor="x", scaleratio=0.68),
#         # plot_bgcolor='#2E3439',
#         plot_bgcolor='rgba(0,0,0,0)',
#         paper_bgcolor='#2E3439',
#         height=450, margin=dict(l=10, r=10, t=40, b=10), showlegend=False
#     )
#     return fig

def plot_pass_density_plotly(
    passes_df,
    team_name,
    is_away=False,
):
    """
    Show the spatial concentration of pass origins.

    All teams are displayed attacking towards x=100.
    is_away controls only the visual palette.
    """

    fig = go.Figure()

    pitch_shapes = (
        pitch_plots.get_plotly_pitch_shapes(
            "rgba(255,255,255,0.20)",
            "rgba(255,255,255,0.72)",
        )
    )

    df_plot = (
        passes_df.copy()
        if passes_df is not None
        else pd.DataFrame()
    )

    # Coordinates are already normalised so every team
    # attacks from left to right. Do not mirror Away.

    if is_away:
        colorscale = [
            [0.00, 'rgba(11,143,178,0.00)'],
            [0.15, 'rgba(11,143,178,0.14)'],
            [0.45, 'rgba(55,184,210,0.52)'],
            [0.75, 'rgba(19,157,191,0.78)'],
            [1.00, 'rgba(4,119,151,0.96)'],
        ]
    else:
        colorscale = [
            [0.00, 'rgba(232,93,72,0.00)'],
            [0.15, 'rgba(232,93,72,0.14)'],
            [0.45, 'rgba(243,132,110,0.52)'],
            [0.75, 'rgba(235,95,73,0.78)'],
            [1.00, 'rgba(196,62,45,0.96)'],
        ]

    if not df_plot.empty:

        x = pd.to_numeric(
            df_plot['x'],
            errors='coerce',
        )

        y = pd.to_numeric(
            df_plot['y'],
            errors='coerce',
        )

        valid = (
            x.between(0, 100)
            & y.between(0, 100)
        )

        df_plot = df_plot.loc[valid].copy()

        if not df_plot.empty:

            # ---------------------------------------------
            # DENSITY
            # ---------------------------------------------
            fig.add_trace(
                go.Histogram2dContour(
                    x=df_plot['x'],
                    y=df_plot['y'],

                    colorscale=colorscale,
                    showscale=False,

                    contours=dict(
                        coloring='fill',
                        showlines=False,
                    ),

                    ncontours=14,
                    opacity=0.92,

                    hoverinfo='skip',
                    showlegend=False,
                )
            )

            # ---------------------------------------------
            # RAW PASS ORIGINS
            # ---------------------------------------------
            hover_text = []

            for _, row in df_plot.iterrows():

                player = row.get(
                    'playerName',
                    'Unknown',
                )

                minute = row.get(
                    'timeMin',
                    '?',
                )

                hover_text.append(
                    (
                        f"<b>{player}</b>"
                        f"<br>Minute: {minute}'"
                        f"<br>Origin: "
                        f"{float(row['x']):.1f}, "
                        f"{float(row['y']):.1f}"
                    )
                )

            fig.add_trace(
                go.Scattergl(
                    x=df_plot['x'],
                    y=df_plot['y'],

                    mode='markers',

                    marker=dict(
                        size=4,
                        color='#e8f0f4',
                        opacity=0.32,
                        line=dict(
                            width=0,
                        ),
                    ),

                    text=hover_text,

                    hovertemplate=(
                        "%{text}"
                        "<extra></extra>"
                    ),

                    showlegend=False,
                )
            )

    if df_plot.empty:
        fig.add_annotation(
            x=50,
            y=50,
            text="No pass-location data",
            showarrow=False,
            font=dict(
                color=(
                    'rgba(255,255,255,0.68)'
                ),
                size=14,
            ),
        )

    # ---------------------------------------------
    # ATTACKING DIRECTION
    # ---------------------------------------------
    add_attacking_direction(fig, dark=True)

    fig.update_layout(
        shapes=pitch_shapes,

        xaxis=dict(
            range=[-2, 102],
            visible=False,
            fixedrange=True,
        ),

        yaxis=dict(
            range=[-5, 107],
            visible=False,
            fixedrange=True,
        ),
    )

    apply_dark_pitch_layout(
        fig,
        height=510,
        top_margin=120,
        showlegend=False,
    )

    add_plot_header(
        fig,
        title=f"{team_name} · Pass origin density",
        subtitle=(
            f"{len(df_plot)} pass attempts · "
            "brighter areas = higher concentration"
        ),
        dark=True,
    )

    return fig

# def plot_pass_heatmap_plotly(passes_df, team_name, is_away=False):
#     """
#     Versione 3: Aggiunge bordi ai bin, punti di passaggio e mostra percentuali.
#     """
#     fig = go.Figure()
#     pitch_shapes = pitch_plots.get_plotly_pitch_shapes("rgba(0, 0, 0, 0.5)")

#     df_plot = passes_df.copy()
#     if is_away:
#         df_plot['x'] = 100 - df_plot['x']
#         df_plot['y'] = 100 - df_plot['y']

#     colorscale = 'Reds' if not is_away else 'Blues'

#     if not df_plot.empty:
#         total_passes = len(df_plot)
#         x_bins, y_bins = np.linspace(0, 100, 7), np.linspace(0, 100, 6)
#         counts, y_edges, x_edges = np.histogram2d(df_plot['y'], df_plot['x'], bins=[y_bins, x_bins])

#         # Le percentuali vengono calcolate sui conteggi
#         percentages = (counts / total_passes) * 100 if total_passes > 0 else counts

#         # 1. Disegna la Heatmap con i bordi
#         fig.add_trace(go.Heatmap(
#             z=counts,
#             x=(x_edges[:-1] + x_edges[1:]) / 2,
#             y=(y_edges[:-1] + y_edges[1:]) / 2,
#             colorscale=colorscale,
#             colorbar=dict(
#                 title='Passes',
#                 tickfont=dict(
#                     color='white' # Colore per i numeri (ticks) della colorbar
#                 ),
#                 title_font=dict(
#                     color='white' # Colore per il titolo ("Passes") della colorbar
#                 )
#             ),
#             xgap=1, ygap=1
#         ))

#         # 2. Aggiungi i Punti di Passaggio sopra la heatmap
#         fig.add_trace(go.Scatter(
#             x=df_plot['x'], y=df_plot['y'],
#             mode='markers',
#             marker=dict(color='black', size=3, opacity=0.4),
#             hoverinfo='none', showlegend=False
#         ))

#         # 3. Aggiungi le etichette con le PERCENTUALI
#         annotations = []
#         for i, row in enumerate(percentages):
#             for j, perc in enumerate(row):
#                 if perc > 0:
#                     annotations.append(go.layout.Annotation(
#                         x=(x_edges[j] + x_edges[j+1]) / 2,
#                         y=(y_edges[i] + y_edges[i+1]) / 2,
#                         text=f"<b>{perc:.0f}%</b>", # Mostra la percentuale
#                         showarrow=False,
#                         font=dict(color='white' if counts[i, j] > counts.max() / 1.8 else 'black', size=11)
#                     ))
#         fig.update_layout(annotations=annotations)

#     # Il layout rimane quasi identico, ma ora le shapes sono sopra tutto
#     fig.update_layout(
#         title=dict(text="Pass Heatmap", font_color='white', x=0.5),
#         shapes=pitch_shapes,
#         xaxis=dict(range=[-2, 102], visible=False),
#         yaxis=dict(range=[-2, 102], visible=False, scaleanchor="x", scaleratio=0.68),
#         plot_bgcolor='#2E3439', paper_bgcolor='#2E3439',
#         height=450, margin=dict(l=10, r=40, t=40, b=10), showlegend=False
#     )

#     # Forza le forme del campo ad essere sopra la heatmap
#     for shape in fig.layout.shapes:
#         shape.layer = 'above'

#     return fig

def plot_pass_heatmap_plotly(
    passes_df,
    team_name,
    is_away=False,
):
    """
    Show the distribution of pass origins in pitch bins.

    Percentages use all valid pass origins as denominator.
    Permanent labels are shown only for meaningful cells.
    """

    fig = go.Figure()

    pitch_shapes = (
        pitch_plots.get_plotly_pitch_shapes(
            "rgba(255,255,255,0.22)",
            "rgba(255,255,255,0.74)",
        )
    )

    df_plot = (
        passes_df.copy()
        if passes_df is not None
        else pd.DataFrame()
    )

    # Again: Away changes palette only, not orientation.

    if is_away:
        colorscale = [
            [0.00, 'rgba(11,143,178,0.00)'],
            [0.15, 'rgba(11,143,178,0.16)'],
            [0.50, '#63c5d8'],
            [1.00, '#087f9f'],
        ]
    else:
        colorscale = [
            [0.00, 'rgba(232,93,72,0.00)'],
            [0.15, 'rgba(232,93,72,0.16)'],
            [0.50, '#f09581'],
            [1.00, '#d84f39'],
        ]

    if not df_plot.empty:

        x = pd.to_numeric(
            df_plot['x'],
            errors='coerce',
        )

        y = pd.to_numeric(
            df_plot['y'],
            errors='coerce',
        )

        valid = (
            x.between(0, 100)
            & y.between(0, 100)
        )

        df_plot = df_plot.loc[valid].copy()

    total_passes = len(df_plot)

    if total_passes > 0:

        # Keep the existing useful 6 × 5 grid.
        x_bins = np.linspace(
            0,
            100,
            7,
        )

        y_bins = np.linspace(
            0,
            100,
            6,
        )

        counts, y_edges, x_edges = (
            np.histogram2d(
                df_plot['y'],
                df_plot['x'],
                bins=[
                    y_bins,
                    x_bins,
                ],
            )
        )

        percentages = (
            counts
            / total_passes
            * 100
        )

        x_centres = (
            x_edges[:-1]
            + x_edges[1:]
        ) / 2

        y_centres = (
            y_edges[:-1]
            + y_edges[1:]
        ) / 2

        max_percentage = max(
            float(
                percentages.max()
            ),
            1.0,
        )

        # ---------------------------------------------
        # HEATMAP
        # ---------------------------------------------
        fig.add_trace(
            go.Heatmap(
                z=percentages,
                x=x_centres,
                y=y_centres,

                customdata=counts,

                colorscale=colorscale,

                zmin=0,
                zmax=max_percentage,

                showscale=False,

                xgap=2,
                ygap=2,

                hovertemplate=(
                    "<b>%{customdata:.0f} passes</b>"
                    "<br>%{z:.1f}% of pass origins"
                    "<extra></extra>"
                ),
            )
        )

        # ---------------------------------------------
        # LABEL ONLY MEANINGFUL CELLS
        #
        # At least:
        # - 3 events
        # - roughly 2% of team pass volume
        # ---------------------------------------------
        label_threshold = max(
            3,
            int(
                np.ceil(
                    total_passes * 0.02
                )
            ),
        )

        for i in range(
            counts.shape[0]
        ):
            for j in range(
                counts.shape[1]
            ):

                count = int(
                    counts[i, j]
                )

                if count < label_threshold:
                    continue

                percentage = float(
                    percentages[i, j]
                )

                intensity = (
                    percentage
                    / max_percentage
                )

                text_color = (
                    '#ffffff'
                    if intensity >= 0.48
                    else '#dce8ee'
                )

                fig.add_annotation(
                    x=x_centres[j],
                    y=y_centres[i],

                    text=(
                        f"<b>{percentage:.0f}%</b>"
                        f"<br>"
                        f"<span style='font-size:9px'>"
                        f"{count}"
                        f"</span>"
                    ),

                    showarrow=False,

                    font=dict(
                        color=text_color,
                        size=11,
                    ),
                )

    else:
        fig.add_annotation(
            x=50,
            y=50,
            text="No pass-location data",
            showarrow=False,
            font=dict(
                color=(
                    'rgba(255,255,255,0.68)'
                ),
                size=14,
            ),
        )

    add_attacking_direction(fig, dark=True)

    fig.update_layout(
        shapes=pitch_shapes,

        xaxis=dict(
            range=[-2, 102],
            visible=False,
            fixedrange=True,
        ),

        yaxis=dict(
            range=[-5, 107],
            visible=False,
            fixedrange=True,
        ),
    )

    # Pitch markings must remain readable above the cells.
    for shape in fig.layout.shapes:
        shape.layer = 'above'

    apply_dark_pitch_layout(
        fig,
        height=510,
        top_margin=120,
        showlegend=False,
    )

    add_plot_header(
        fig,
        title=f"{team_name} · Pass origin profile",
        subtitle=(
            f"{total_passes} pass attempts · "
            "share of origins by pitch zone"
        ),
        dark=True,
    )

    return fig


# PLOT-04 — compact reliable pass network
def plot_pass_network_profile_plotly(edges, nodes, team_name, *, is_away=False):
    """Presentation-only renderer for the PLOT-04 network profile."""
    del team_name
    palette = get_team_palette(is_away=is_away)
    fig = go.Figure()
    pitch_shapes = pitch_plots.get_plotly_pitch_shapes(
        "rgba(255,255,255,0.22)",
        "rgba(255,255,255,0.78)",
    )
    edges = edges.copy() if edges is not None else pd.DataFrame()

    # PLOT-04A — only nodes belonging to displayed connections
    if not edges.empty:
        shown_players = set(
            edges["player1"].dropna().astype(str)
        ).union(
            set(
                edges["player2"].dropna().astype(str)
            )
        )

        nodes = nodes[
            nodes["playerName"]
            .astype(str)
            .isin(shown_players)
        ].copy()
    else:
        # A network without qualifying edges should not show a cloud of
        # isolated eligible players.
        nodes = nodes.iloc[0:0].copy()
    nodes = nodes.copy() if nodes is not None else pd.DataFrame()

    if not edges.empty:
        max_count = max(float(pd.to_numeric(edges["pass_count"], errors="coerce").max()), 1.0)
        hx, hy, custom = [], [], []
        for _, row in edges.sort_values("pass_count").iterrows():
            count = float(row["pass_count"])
            strength = (count / max_count) ** 0.72
            fig.add_trace(go.Scatter(
                x=[row["pass_avg_x"], row["pass_avg_x_end"]],
                y=[row["pass_avg_y"], row["pass_avg_y_end"]],
                mode="lines",
                line=dict(width=1.0 + 4.5 * strength, color=palette["primary"]),
                opacity=0.28 + 0.56 * strength,
                hoverinfo="skip",
                showlegend=False,
            ))
            hx.append((row["pass_avg_x"] + row["pass_avg_x_end"]) / 2)
            hy.append((row["pass_avg_y"] + row["pass_avg_y_end"]) / 2)
            custom.append([
                row["player1"], row["player2"], int(row["pass_count"]),
                int(row["player1_to_player2"]), int(row["player2_to_player1"]),
            ])
        fig.add_trace(go.Scatter(
            x=hx, y=hy, mode="markers",
            marker=dict(size=20, opacity=0),
            customdata=np.asarray(custom, dtype=object),
            hovertemplate=(
                "<b>%{customdata[0]} ↔ %{customdata[1]}</b>"
                "<br>Total: %{customdata[2]} passes"
                "<br>%{customdata[0]} → %{customdata[1]}: %{customdata[3]}"
                "<br>%{customdata[1]} → %{customdata[0]}: %{customdata[4]}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    if nodes.empty:
        add_zero_state(fig, "No eligible players in this window", dark=True)
    else:
        involvement = pd.to_numeric(nodes["pass_involvement"], errors="coerce").fillna(0)
        max_involvement = max(float(involvement.max()), 1.0)
        nodes["marker_size"] = 19 + 25 * np.sqrt(involvement / max_involvement)

        def jersey_text(frame):
            labels = []
            for value in frame["jersey_number"]:
                try:
                    labels.append(f"<b>{int(float(value))}</b>")
                except (TypeError, ValueError):
                    labels.append("")
            return labels

        for status, symbol, outline in (
            ("Starter", "circle", "rgba(255,255,255,0.88)"),
            ("Substitute", "diamond", "#d9c98c"),
        ):
            subset = nodes[nodes["status"].eq(status)]
            if subset.empty:
                continue
            fig.add_trace(go.Scatter(
                x=subset["pass_avg_x"], y=subset["pass_avg_y"],
                mode="markers+text", text=jersey_text(subset),
                textposition="middle center",
                textfont=dict(color="white", size=10),
                marker=dict(
                    symbol=symbol, color=palette["primary"],
                    size=subset["marker_size"], opacity=0.96,
                    line=dict(width=1.6, color=outline),
                ),
                customdata=np.column_stack([
                    subset["playerName"], subset["minutes"], subset["pass_sent"],
                    subset["pass_received"], subset["pass_involvement"],
                ]),
                hovertemplate=(
                    "<b>%{customdata[0]}</b>"
                    f"<br>{status}"
                    "<br>Minutes in window: %{customdata[1]:.0f}"
                    "<br>Passes sent: %{customdata[2]}"
                    "<br>Passes received: %{customdata[3]}"
                    "<br>Pass involvement: %{customdata[4]}"
                    "<extra></extra>"
                ),
                name=status,
                showlegend=False,
            ))

    apply_match_pitch_layout(
        fig,
        pitch_shapes=pitch_shapes,
        height=560,
        showlegend=False,
        header=False,
        x_range=(-2, 102),
        y_range=(-5, 107),
    )
    add_attacking_direction(
        fig, dark=True, x=0.985, y=0.025,
        xanchor="right", yanchor="bottom",
    )
    return fig

# PLOT-05 — aggregated Progressive Pass summary map
def _progressive_pass_aggregate_locations(
    df_prog_passes,
    *,
    grid_size=12,
):
    columns = [
        "kind",
        "x",
        "y",
        "count",
        "completed",
        "completion_pct",
    ]

    if (
        df_prog_passes is None
        or df_prog_passes.empty
    ):
        return pd.DataFrame(columns=columns)

    attempts = df_prog_passes.copy()

    if "is_progressive_attempt" in attempts.columns:
        attempts = attempts[
            attempts["is_progressive_attempt"]
            .fillna(False)
            .astype(bool)
        ].copy()

    if attempts.empty:
        return pd.DataFrame(columns=columns)

    completed = attempts.get(
        "is_progressive",
        pd.Series(False, index=attempts.index),
    ).fillna(False).astype(bool)

    records = []

    for kind, x_column, y_column in (
        ("Origin", "x", "y"),
        ("Destination", "end_x", "end_y"),
    ):
        if (
            x_column not in attempts.columns
            or y_column not in attempts.columns
        ):
            continue

        subset = pd.DataFrame({
            "x": pd.to_numeric(
                attempts[x_column],
                errors="coerce",
            ),
            "y": pd.to_numeric(
                attempts[y_column],
                errors="coerce",
            ),
            "completed": completed.astype(int),
        }).dropna(subset=["x", "y"])

        if subset.empty:
            continue

        subset["_bin_x"] = (
            np.floor(
                subset["x"] / float(grid_size)
            )
            .clip(
                lower=0,
                upper=max(int(100 / grid_size), 1),
            )
            .astype(int)
        )
        subset["_bin_y"] = (
            np.floor(
                subset["y"] / float(grid_size)
            )
            .clip(
                lower=0,
                upper=max(int(100 / grid_size), 1),
            )
            .astype(int)
        )

        grouped = (
            subset
            .groupby(
                ["_bin_x", "_bin_y"],
                as_index=False,
            )
            .agg(
                x=("x", "median"),
                y=("y", "median"),
                count=("x", "size"),
                completed=("completed", "sum"),
            )
        )
        grouped["completion_pct"] = (
            grouped["completed"]
            / grouped["count"]
            * 100.0
        )
        grouped["kind"] = kind
        records.append(grouped[columns])

    if not records:
        return pd.DataFrame(columns=columns)

    return pd.concat(
        records,
        ignore_index=True,
    )


def _progressive_pass_tactical_zone(
    x,
    y,
):
    """
    Map one Opta coordinate to a 4 x 3 tactical zone.

    Longitudinal bands:
      0 = build-up
      1 = middle third
      2 = advanced
      3 = final quarter

    Lateral channels:
      0 = left
      1 = central
      2 = right

    Team plots share the same left-to-right attacking orientation.
    """
    x_value = float(x)
    y_value = float(y)

    x_band = min(
        int(
            max(
                x_value,
                0.0,
            )
            // 25.0
        ),
        3,
    )

    if y_value < (100.0 / 3.0):
        channel = 2
    elif y_value < (200.0 / 3.0):
        channel = 1
    else:
        channel = 0

    return (
        x_band,
        channel,
    )


def _progressive_pass_aggregate_routes(
    df_prog_passes,
    *,
    top_n=8,
):
    """
    Aggregate progressive passes by tactical origin -> destination zone.

    The underlying progressive-pass definition is unchanged. Only the
    presentation layer groups geometrically similar attempts.

    The plotted arrow uses the median real origin and destination of all
    passes in that tactical route, rather than the zone centre.
    """
    columns = [
        "start_band",
        "start_channel",
        "end_band",
        "end_channel",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "attempts",
        "completed",
        "completion_pct",
        "average_length_m",
        "average_progression_m",
    ]

    if (
        df_prog_passes is None
        or df_prog_passes.empty
    ):
        return pd.DataFrame(
            columns=columns
        )

    attempts = df_prog_passes.copy()

    if "is_progressive_attempt" in attempts.columns:
        attempts = attempts[
            attempts[
                "is_progressive_attempt"
            ]
            .fillna(False)
            .astype(bool)
        ].copy()

    if attempts.empty:
        return pd.DataFrame(
            columns=columns
        )

    for column in (
        "x",
        "y",
        "end_x",
        "end_y",
    ):
        attempts[column] = pd.to_numeric(
            attempts.get(
                column
            ),
            errors="coerce",
        )

    attempts = attempts.dropna(
        subset=[
            "x",
            "y",
            "end_x",
            "end_y",
        ]
    ).copy()

    if attempts.empty:
        return pd.DataFrame(
            columns=columns
        )

    attempts["_completed"] = (
        attempts.get(
            "is_progressive",
            pd.Series(
                False,
                index=attempts.index,
            ),
        )
        .fillna(False)
        .astype(bool)
        .astype(int)
    )

    dx_m = (
        attempts["end_x"]
        - attempts["x"]
    ) * 1.05

    dy_m = (
        attempts["end_y"]
        - attempts["y"]
    ) * 0.68

    attempts["_pass_length_m"] = np.hypot(
        dx_m,
        dy_m,
    )

    if (
        "progressive_distance_m"
        in attempts.columns
    ):
        attempts["_progression_m"] = (
            pd.to_numeric(
                attempts[
                    "progressive_distance_m"
                ],
                errors="coerce",
            )
        )
    else:
        attempts["_progression_m"] = (
            attempts["end_x"]
            - attempts["x"]
        ) * 1.05

    zone_pairs = attempts.apply(
        lambda row: (
            _progressive_pass_tactical_zone(
                row["x"],
                row["y"],
            ),
            _progressive_pass_tactical_zone(
                row["end_x"],
                row["end_y"],
            ),
        ),
        axis=1,
    )

    attempts[
        "_start_band"
    ] = [
        value[0][0]
        for value in zone_pairs
    ]

    attempts[
        "_start_channel"
    ] = [
        value[0][1]
        for value in zone_pairs
    ]

    attempts[
        "_end_band"
    ] = [
        value[1][0]
        for value in zone_pairs
    ]

    attempts[
        "_end_channel"
    ] = [
        value[1][1]
        for value in zone_pairs
    ]

    grouped = (
        attempts
        .groupby(
            [
                "_start_band",
                "_start_channel",
                "_end_band",
                "_end_channel",
            ],
            as_index=False,
        )
        .agg(
            start_x=("x", "median"),
            start_y=("y", "median"),
            end_x=("end_x", "median"),
            end_y=("end_y", "median"),
            attempts=("x", "size"),
            completed=(
                "_completed",
                "sum",
            ),
            average_length_m=(
                "_pass_length_m",
                "mean",
            ),
            average_progression_m=(
                "_progression_m",
                "mean",
            ),
        )
        .rename(
            columns={
                "_start_band":
                    "start_band",
                "_start_channel":
                    "start_channel",
                "_end_band":
                    "end_band",
                "_end_channel":
                    "end_channel",
            }
        )
    )

    grouped[
        "completion_pct"
    ] = (
        grouped["completed"]
        / grouped["attempts"]
        * 100.0
    )

    grouped = (
        grouped
        .sort_values(
            [
                "attempts",
                "completed",
                "average_progression_m",
            ],
            ascending=[
                False,
                False,
                False,
            ],
            kind="stable",
        )
        .head(
            max(
                int(top_n),
                0,
            )
        )
        .reset_index(
            drop=True
        )
    )

    return grouped[
        columns
    ]

def plot_progressive_pass_summary_plotly(
    df_prog_passes,
    team_name,
    team_color,
    is_away=False,
):
    """
    Summary view using the strongest tactical origin -> destination routes.

    Each arrow aggregates progressive passes sharing the same 4 x 3 tactical
    origin and destination zones. The arrow itself is drawn between the median
    real coordinates of those attempts.
    """
    del team_name
    del team_color

    palette = get_team_palette(
        is_away=is_away
    )

    fig = go.Figure()

    pitch_shapes = (
        pitch_plots
        .get_plotly_pitch_shapes(
            "rgba(255,255,255,0.24)",
            "rgba(255,255,255,0.82)",
        )
    )

    # 4 longitudinal bands.
    for x_value in (
        25,
        50,
        75,
    ):
        pitch_shapes.append(
            dict(
                type="line",
                x0=x_value,
                y0=0,
                x1=x_value,
                y1=100,
                line=dict(
                    color=(
                        "rgba(255,255,255,0.10)"
                    ),
                    dash="dot",
                    width=1,
                ),
            )
        )

    # 3 lateral channels.
    for y_value in (
        100 / 3,
        200 / 3,
    ):
        pitch_shapes.append(
            dict(
                type="line",
                x0=0,
                y0=y_value,
                x1=100,
                y1=y_value,
                line=dict(
                    color=(
                        "rgba(255,255,255,0.10)"
                    ),
                    dash="dot",
                    width=1,
                ),
            )
        )

    routes = (
        _progressive_pass_aggregate_routes(
            df_prog_passes,
            top_n=8,
        )
    )

    if routes.empty:
        add_zero_state(
            fig,
            (
                "No open-play progressive pass "
                "attempts for this team"
            ),
            dark=True,
        )
    else:
        max_attempts = max(
            int(
                routes[
                    "attempts"
                ].max()
            ),
            1,
        )

        midpoint_x = []
        midpoint_y = []
        midpoint_size = []
        midpoint_text = []
        midpoint_custom = []

        band_labels = {
            0: "Build-up",
            1: "Middle",
            2: "Advanced",
            3: "Final quarter",
        }

        channel_labels = {
            0: "Left",
            1: "Central",
            2: "Right",
        }

        for rank, row in routes.iterrows():
            route_attempts = int(
                row["attempts"]
            )

            strength = (
                route_attempts
                / max_attempts
            ) ** 0.72

            width = (
                1.8
                + 4.8 * strength
            )

            opacity = (
                0.42
                + 0.48 * strength
            )

            fig.add_annotation(
                x=float(
                    row["end_x"]
                ),
                y=float(
                    row["end_y"]
                ),
                ax=float(
                    row["start_x"]
                ),
                ay=float(
                    row["start_y"]
                ),
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                text="",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.15,
                arrowwidth=width,
                arrowcolor=(
                    palette["primary"]
                ),
                opacity=opacity,
            )

            start_x = float(
                row["start_x"]
            )
            start_y = float(
                row["start_y"]
            )
            end_x = float(
                row["end_x"]
            )
            end_y = float(
                row["end_y"]
            )

            route_dx = (
                end_x
                - start_x
            )
            route_dy = (
                end_y
                - start_y
            )

            route_norm = max(
                float(
                    np.hypot(
                        route_dx,
                        route_dy,
                    )
                ),
                1e-9,
            )

            # Put the attempt badge beside the arrow instead of on top of it.
            # Alternate sides to reduce collisions between nearby routes.
            badge_side = (
                1.0
                if rank % 2 == 0
                else -1.0
            )
            badge_offset = 4.0

            badge_x = (
                (
                    start_x
                    + end_x
                )
                / 2.0
                + (
                    -route_dy
                    / route_norm
                    * badge_offset
                    * badge_side
                )
            )

            badge_y = (
                (
                    start_y
                    + end_y
                )
                / 2.0
                + (
                    route_dx
                    / route_norm
                    * badge_offset
                    * badge_side
                )
            )

            midpoint_x.append(
                badge_x
            )

            midpoint_y.append(
                badge_y
            )

            midpoint_size.append(
                19.0
                + 10.0 * strength
            )

            midpoint_text.append(
                str(
                    route_attempts
                )
            )

            start_zone = (
                f"{band_labels[int(row['start_band'])]} "
                f"{channel_labels[int(row['start_channel'])]}"
            )

            end_zone = (
                f"{band_labels[int(row['end_band'])]} "
                f"{channel_labels[int(row['end_channel'])]}"
            )

            midpoint_custom.append([
                rank + 1,
                route_attempts,
                int(
                    row[
                        "completed"
                    ]
                ),
                float(
                    row[
                        "completion_pct"
                    ]
                ),
                float(
                    row[
                        "average_length_m"
                    ]
                ),
                float(
                    row[
                        "average_progression_m"
                    ]
                ),
                start_zone,
                end_zone,
            ])

        fig.add_trace(
            go.Scatter(
                x=midpoint_x,
                y=midpoint_y,
                mode="markers+text",
                marker=dict(
                    size=midpoint_size,
                    symbol="circle",
                    color=(
                        "rgba(16,47,69,0.98)"
                    ),
                    line=dict(
                        color=(
                            palette[
                                "primary"
                            ]
                        ),
                        width=2.1,
                    ),
                ),
                text=midpoint_text,
                textposition=(
                    "middle center"
                ),
                textfont=dict(
                    color="#ffffff",
                    size=10,
                ),
                customdata=np.asarray(
                    midpoint_custom,
                    dtype=object,
                ),
                hovertemplate=(
                    "<b>%{customdata[6]} → %{customdata[7]}</b>"
                    "<br>Route rank: #%{customdata[0]}"
                    "<br>Attempts: %{customdata[1]}"
                    "<br>Completed: %{customdata[2]}"
                    "<br>Completion: %{customdata[3]:.0f}%"
                    "<br>Average pass length: %{customdata[4]:.1f} m"
                    "<br>Average progression: %{customdata[5]:.1f} m"
                    "<extra></extra>"
                ),
                showlegend=False,
                name="Top tactical routes",
            )
        )

        fig.add_annotation(
            x=0.02,
            y=0.985,
            xref="paper",
            yref="paper",
            text=(
                "<b>Top 8 tactical routes</b>"
                "<br><span style='font-size:11px'>"
                "4 longitudinal bands × 3 channels · "
                "width = volume · badge = attempts"
                "</span>"
            ),
            showarrow=False,
            xanchor="left",
            yanchor="top",
            align="left",
            font=dict(
                color="#ffffff",
                size=12,
            ),
            bgcolor=(
                "rgba(16,47,69,0.86)"
            ),
            bordercolor=(
                "rgba(255,255,255,0.16)"
            ),
            borderwidth=1,
            borderpad=5,
        )

    apply_match_pitch_layout(
        fig,
        pitch_shapes=pitch_shapes,
        height=560,
        showlegend=False,
        header=False,
        x_range=(-2, 102),
        y_range=(-5, 107),
    )

    add_attacking_direction(
        fig,
        dark=True,
        x=0.985,
        y=0.025,
        xanchor="right",
        yanchor="bottom",
    )

    return fig
