import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from ..config import BG_COLOR, LINE_COLOR, GREEN, VIOLET, CARRY_COLOR, SHOT_TYPES, UNSUCCESSFUL_COLOR
from plotly.colors import sample_colorscale
from src.config import TEAM_NAME_TO_LOGO_CODE, LOGO_PREFIX, LOGO_EXTENSION, DEFAULT_LOGO_PATH

# This is the helper function we created for the defender map. We can reuse it.
def draw_plotly_pitch(fig):
    """Helper function to draw an Opta pitch using Plotly shapes."""
    pitch_shapes = [
        # Outer lines & halfway line
        go.layout.Shape(type="rect", x0=0, y0=0, x1=100, y1=100, line=dict(color=LINE_COLOR, width=2)),
        go.layout.Shape(type="line", x0=50, y0=0, x1=50, y1=100, line=dict(color=LINE_COLOR, width=2)),
        # Center circle
        go.layout.Shape(type="circle", x0=42, y0=42, x1=58, y1=58, line=dict(color=LINE_COLOR, width=2)),
        go.layout.Shape(type="circle", x0=49.5, y0=49.5, x1=50.5, y1=50.5, line=dict(color=LINE_COLOR, width=2), fillcolor=LINE_COLOR),
        # Penalty Areas
        go.layout.Shape(type="rect", x0=0, y0=21.1, x1=16.5, y1=78.9, line=dict(color=LINE_COLOR, width=2)),
        go.layout.Shape(type="rect", x0=83.5, y0=21.1, x1=100, y1=78.9, line=dict(color=LINE_COLOR, width=2)),
        # 6-yard boxes
        go.layout.Shape(type="rect", x0=0, y0=36.8, x1=5.5, y1=63.2, line=dict(color=LINE_COLOR, width=2)),
        go.layout.Shape(type="rect", x0=94.5, y0=36.8, x1=100, y1=63.2, line=dict(color=LINE_COLOR, width=2)),
    ]
    fig.update_layout(shapes=pitch_shapes)
    return fig

def get_team_logo_src(team_name, default_logo_path="/assets/logos/_default_badge.png"):
    if not team_name:
        return default_logo_path

    team_code = TEAM_NAME_TO_LOGO_CODE.get(str(team_name).strip()) # Use .strip() for safety

    if not team_code:
        # Fallback if team name not in mapping: try to generate a code from the name
        # This is less reliable but can be a fallback.
        # For example, take the first 3 letters if you don't have an explicit code.
        # Or, if you expect team names like "Wolverhampton Wanderers" and want "WOL"
        # you might need a more complex fallback logic.
        # For now, if not in map, use default.
        print(f"Warning: Team name '{team_name}' not found in TEAM_NAME_TO_LOGO_CODE mapping. Using default logo.")
        return default_logo_path

    logo_filename = f"{LOGO_PREFIX}{team_code}{LOGO_EXTENSION}" # e.g., ENG_BOU.png
    prospective_src = f"/assets/logos/{logo_filename}"
    
    # Optional server-side check (as before)
    # logo_path_on_server = os.path.join("assets", "logos", logo_filename)
    # if not os.path.exists(logo_path_on_server):
    #     print(f"DEV_NOTE: Logo file not found on server: {logo_path_on_server} for team '{team_name}' (code: {team_code})")
    #     return default_logo_path
        
    return prospective_src

def get_team_logo_src_by_code(team_short_code): # Removed default here, handle in show_cards
    if not team_short_code:
        # print(f"Warning: No team short code provided for logo. Using default.")
        return DEFAULT_LOGO_PATH

    # Assuming your logo files like ENG_BOU.png use uppercase codes
    logo_filename = f"{LOGO_PREFIX}{str(team_short_code).upper()}{LOGO_EXTENSION}"
    prospective_src = f"/assets/logos/{logo_filename}"
    
    # Optional server-side check for debugging (uncomment if needed)
    # logo_path_on_server = os.path.join("assets", "logos", logo_filename)
    # if not os.path.exists(logo_path_on_server):
    #     print(f"DEV_NOTE: Logo file not found on server: {logo_path_on_server} (code: {team_short_code})")
    #     return DEFAULT_LOGO_PATH # Fallback if server check fails
            
    return prospective_src

def plot_loss_heatmap_on_pitch(
    sequences,
    losing_team_is_away=False,
    grid_size=6,
):
    """
    Plot the locations where possession was lost before
    defensive transitions.

    Coordinates come directly from the triggering turnover
    event stored by transition_metrics.
    """

    x_coords = []
    y_coords = []
    hover_texts = []

    # ---------------------------------------------------------
    # POSSESSION LOSS LOCATIONS
    # ---------------------------------------------------------

    for seq in sequences:
        if seq.empty:
            continue

        first = seq.iloc[0]

        x = pd.to_numeric(
            first.get('loss_x'),
            errors='coerce',
        )

        y = pd.to_numeric(
            first.get('loss_y'),
            errors='coerce',
        )

        if pd.isna(x) or pd.isna(y):
            continue

        # Coordinates are metric-normalized.
        # Mirror only for the visual convention used
        # for the away team.
        if losing_team_is_away:
            x = 100.0 - float(x)
            y = 100.0 - float(y)

        x_coords.append(
            float(x)
        )

        y_coords.append(
            float(y)
        )

        loss_type = first.get(
            'type_of_initial_loss',
            'Unknown turnover',
        )

        loss_zone = first.get(
            'loss_zone',
            'Unknown zone',
        )

        outcome = first.get(
            'sequence_outcome_type',
            'Unknown',
        )

        hover_texts.append(
            (
                f"<b>{loss_type}</b>"
                f"<br>Zone: {loss_zone}"
                f"<br>Outcome: {outcome}"
            )
        )

    # ---------------------------------------------------------
    # EMPTY STATE
    # ---------------------------------------------------------

    if not x_coords:
        fig = go.Figure()

        draw_plotly_pitch(fig)

        fig.update_shapes(
            line_color='#718797',
            line_width=1.15,
        )

        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref='paper',
            yref='paper',
            text='No possession-loss data to plot',
            showarrow=False,
            font=dict(
                color='#647c8e',
                size=13,
            ),
        )

        fig.update_layout(
            paper_bgcolor='#ffffff',
            plot_bgcolor='#ffffff',
        )

        return fig

    # ---------------------------------------------------------
    # BINNING
    # ---------------------------------------------------------

    bin_edges = np.linspace(
        0,
        100,
        grid_size + 1,
    )

    heatmap, _, _ = np.histogram2d(
        x_coords,
        y_coords,
        bins=[
            bin_edges,
            bin_edges,
        ],
    )

    total = heatmap.sum()

    heatmap_pct = (
        heatmap / total * 100
        if total > 0
        else heatmap
    )

    max_val = (
        heatmap_pct.max()
        if heatmap_pct.max() > 0
        else 1
    )

    # ---------------------------------------------------------
    # MATCH ANALYSIS PALETTE
    #
    # Home = coral
    # Away = cyan
    # ---------------------------------------------------------

    if losing_team_is_away:
        colorscale = [
            [0.0, '#edf9fc'],
            [0.5, '#74cbdc'],
            [1.0, '#0b88a8'],
        ]
    else:
        colorscale = [
            [0.0, '#fff1ed'],
            [0.5, '#f3a08d'],
            [1.0, '#e7644a'],
        ]

    def rectangle(
        x0,
        x1,
        y0,
        y1,
    ):
        return {
            'x': [
                x0,
                x1,
                x1,
                x0,
                x0,
            ],
            'y': [
                y0,
                y0,
                y1,
                y1,
                y0,
            ],
        }

    fig = go.Figure()

    draw_plotly_pitch(fig)

    fig.update_shapes(
        line_color='#718797',
        line_width=1.15,
    )

    # ---------------------------------------------------------
    # HEATMAP CELLS
    # ---------------------------------------------------------

    for i, x0 in enumerate(
        bin_edges[:-1]
    ):
        x1 = bin_edges[i + 1]

        for j, y0 in enumerate(
            bin_edges[:-1]
        ):
            y1 = bin_edges[j + 1]

            count = int(
                heatmap[i, j]
            )

            if count == 0:
                continue

            perc = float(
                heatmap_pct[i, j]
            )

            intensity = min(
                perc / max_val,
                1.0,
            )

            color = sample_colorscale(
                colorscale,
                [intensity],
            )[0]

            poly = rectangle(
                x0,
                x1,
                y0,
                y1,
            )

            fig.add_trace(
                go.Scatter(
                    x=poly['x'],
                    y=poly['y'],
                    fill='toself',
                    mode='lines',
                    fillcolor=color,
                    line=dict(
                        color=(
                            'rgba(75, 101, 119, 0.22)'
                        ),
                        width=1,
                    ),
                    text=(
                        f'{count} possession losses'
                        f'<br>{perc:.1f}%'
                    ),
                    hovertemplate=(
                        '%{text}'
                        '<extra></extra>'
                    ),
                    showlegend=False,
                )
            )

            # One event stays visible as a dot only.
            # Permanent percentage labels start at 2 events.
            if count >= 2:

                cx = (
                    x0 + x1
                ) / 2

                cy = (
                    y0 + y1
                ) / 2

                text_color = (
                    '#ffffff'
                    if intensity >= 0.48
                    else '#17354d'
                )

                fig.add_trace(
                    go.Scatter(
                        x=[cx],
                        y=[cy],
                        mode='text',
                        text=[
                            f'{perc:.1f}%'
                        ],
                        textfont=dict(
                            size=13,
                            color=text_color,
                            weight='bold',
                        ),
                        showlegend=False,
                        hoverinfo='skip',
                    )
                )

    # ---------------------------------------------------------
    # INDIVIDUAL TURNOVERS
    # ---------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=6,
                color='#17354d',
                opacity=0.70,
                line=dict(
                    color='#ffffff',
                    width=0.8,
                ),
            ),
            text=hover_texts,
            hovertemplate=(
                '%{text}'
                '<br>X: %{x:.1f}'
                '<br>Y: %{y:.1f}'
                '<extra></extra>'
            ),
            name='Possession loss',
            showlegend=False,
        )
    )

    # ---------------------------------------------------------
    # LAYOUT
    # ---------------------------------------------------------

    fig.update_layout(
        title=None,

        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',

        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=True,
            range=[0, 100],
        ),

        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=True,
            range=[0, 100],
            scaleanchor='x',
            scaleratio=0.68,
        ),

        margin=dict(
            l=8,
            r=8,
            t=12,
            b=8,
        ),

        height=540,

        showlegend=False,

        hoverlabel=dict(
            bgcolor='#ffffff',
            bordercolor='#dbe7ed',
            font=dict(
                color='#17354d',
                size=12,
            ),
        ),
    )

    return fig


def plot_defensive_block_plotly(df_def_actions, df_player_agg, team_color, is_away=False):
    """
    Creates an interactive and aesthetically improved defensive block plot using Plotly.
    - Inverts axes for the away team for conventional viewing.
    - Colors player nodes with the team color.
    - Includes a properly calculated and placed average line label.
    """
    fig = go.Figure()
    fig = draw_plotly_pitch(fig)
    thirds = [100/3, 2*100/3]
    for x in thirds:
        fig.add_shape(type="line", x0=x, y0=0, x1=x, y1=100,
                      line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dash"))

    # --- 1. GESTIONE ASSI CORRETTA ---
    # Gli assi vengono invertiti per la squadra in trasferta per una visualizzazione standard
    if is_away:
        fig.update_xaxes(range=[100, 0])
        fig.update_yaxes(range=[100, 0])
    else:
        fig.update_xaxes(range=[0, 100])
        fig.update_yaxes(range=[0, 100])

    # Heatmap
    if not df_def_actions.empty:
        fig.add_trace(go.Histogram2dContour(
            x=df_def_actions['x'], y=df_def_actions['y'],
            colorscale=[[0, 'rgba(0,0,0,0)'], [1, team_color]],
            showscale=False, contours=dict(coloring='fill', showlines=False),
            name='Defensive Heatmap', hoverinfo='none', opacity=0.6
        ))

    if not df_def_actions.empty:
        fig.add_trace(go.Scatter(
            x=df_def_actions['x'],
            y=df_def_actions['y'],
            mode='markers',
            marker=dict(
                color='yellow',
                size=5,
                opacity=0.5,
                line=dict(width=1, color='black')
            ),
            hoverinfo='text',
            hovertext=df_def_actions['type_name'] + ' by ' + df_def_actions['playerName'],
            name='Defensive Actions' # Nome per la legenda (che è nascosta)
        ))
    
    # Player nodes
    if not df_player_agg.empty:
        max_actions = df_player_agg['action_count'].max()
        df_player_agg['marker_size'] = 30 + (df_player_agg['action_count'] / max_actions * 30) if max_actions > 0 else 25

        for i, row in df_player_agg.iterrows():
            jersey_text = str(int(row['Mapped Jersey Number'])) if pd.notna(row['Mapped Jersey Number']) else ''
            hover_text = f"<b>{row['playerName']}</b><br>Def. Actions: {row['action_count']}"
            
            is_starter = row['Is Starter']
            
            marker_symbol = 'circle' if is_starter else 'diamond'
            # Usa il colore pieno per i titolari, un colore con opacità per i sostituti
            node_color = team_color if is_starter else f"rgba({int(team_color[1:3], 16)}, {int(team_color[3:5], 16)}, {int(team_color[5:7], 16)}, 0.6)"
            # Converte l'hex (es. #FF5733) in RGBA (es. rgba(255, 87, 51, 0.6)) per i sostituti
            
            # Testo bianco o nero a seconda del colore base (non dell'opacità)
            text_color = 'white' if team_color.lower() in ['red', 'blue', 'green', 'black', 'purple', 'tomato', 'skyblue'] else 'black'
            
            fig.add_trace(go.Scatter(
                x=[row['median_x']], y=[row['median_y']],
                mode='markers+text',
                marker=dict(
                    color=node_color, # 
                    size=row['marker_size'],
                    symbol=marker_symbol,
                    line=dict(color='white' if is_starter else 'yellow', width=2)
                ),
                text=jersey_text,
                textfont=dict(color=text_color, size=10, weight='bold'),
                hovertext=hover_text, hoverinfo='text', name=row['playerName']
            ))

    # Linea difensiva media e etichetta
    avg_line_x = df_player_agg['median_x'].mean()
    if pd.notna(avg_line_x):
        fig.add_shape(
            type='line',
            x0=avg_line_x, y0=-5, x1=avg_line_x, y1=105,
            line=dict(color='black', width=3, dash='dashdot')
        )
        
        # Calcolo corretto che tiene conto dell'inversione degli assi per la visualizzazione
        pitch_length_meters = 105.0
        # La distanza è sempre calcolata dalla coordinata x non invertita
        avg_line_meters = avg_line_x * (pitch_length_meters / 100.0)
        
        # Posizionamento relativo dell'etichetta (funziona indipendentemente dall'inversione)
        x_paper_coord = avg_line_x / 100.0
        if is_away:
             x_paper_coord = 1 - x_paper_coord # Inverti la posizione relativa per il team away

        fig.add_annotation(
            x=x_paper_coord, y=1.05,
            xref="paper", yref="paper",
            text=f"<b>Avg. Line: {avg_line_meters:.1f}m</b>",
            showarrow=False,
            font=dict(color="white", size=14, family="Arial"),
            bgcolor="rgba(46, 52, 57, 0.8)",
            bordercolor="white", borderwidth=1, borderpad=4
        )

    # Layout Finale
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="#2E3439",
        margin=dict(l=10, r=10, t=40, b=10),
        height=600,
        xaxis=dict(showgrid=False, zeroline=False, visible=False, fixedrange=True),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, fixedrange=True, scaleanchor="x", scaleratio=0.68)
    )

    return fig

def plot_defensive_hull_plotly(df_player_agg, team_color, is_away=False):
    """
    Creates an interactive and aesthetically improved defensive shape plot using a Convex Hull.
    - Colors player nodes with the team color.
    - Uses a diamond shape for substitutes.
    - Has a denser, more visible hull area and outline.
    """
    fig = go.Figure()
    fig = draw_plotly_pitch(fig)

    if is_away:
        fig.update_xaxes(range=[100, 0])
        fig.update_yaxes(range=[100, 0])
    else:
        fig.update_xaxes(range=[0, 100])
        fig.update_yaxes(range=[0, 100])

    # --- Escludi il portiere per una forma più realistica ---
    if 'Mapped Jersey Number' in df_player_agg.columns:
        df_outfield = df_player_agg[df_player_agg['Mapped Jersey Number'] != 1]
    else:
        df_outfield = df_player_agg
    
    # --- Calcola e disegna il Convex Hull con colori più densi ---
    if len(df_outfield) >= 3:
        points = df_outfield[['median_x', 'median_y']].values
        try:
            hull = ConvexHull(points)
            hull_x = list(points[hull.vertices, 0]) + [points[hull.vertices, 0][0]]
            hull_y = list(points[hull.vertices, 1]) + [points[hull.vertices, 1][0]]

            # Disegna l'area del Convex Hull
            fig.add_trace(go.Scatter(
                x=hull_x, y=hull_y,
                fill="toself",
                fillcolor=team_color,
                opacity=0.4,  # <-- MODIFICA: Opacità aumentata per un colore più denso
                line=dict(color=team_color, width=3, dash='dash'), # <-- MODIFICA: Linea più spessa
                hoverinfo="none",
                showlegend=False
            ))
        except Exception as e:
            print(f"Could not compute Convex Hull: {e}")

    # --- Disegna i nodi dei giocatori con colori e forme personalizzate ---
    if not df_player_agg.empty:
        # Dimensione fissa per i nodi, ma più grande
        df_player_agg['marker_size'] = 50

        for i, row in df_player_agg.iterrows():
            jersey_text = str(int(row['Mapped Jersey Number'])) if pd.notna(row['Mapped Jersey Number']) else ''
            hover_text = f"<b>{row['playerName']}</b><br>Def. Actions: {row['action_count']}"

            is_starter = row['Is Starter']
            
            marker_symbol = 'circle' if is_starter else 'diamond'
            # Colore pieno per titolari, sbiadito per sostituti
            node_color = team_color if is_starter else f"rgba({int(team_color[1:3], 16)}, {int(team_color[3:5], 16)}, {int(team_color[5:7], 16)}, 0.6)"
            # Bordo bianco per titolari, giallo per sostituti
            marker_line_color = 'white' if is_starter else 'yellow'
            
            # Colore del testo a contrasto
            text_color = 'white' if team_color.lower() in ['red', 'blue', 'green', 'black', 'purple', 'tomato', 'skyblue'] else 'black'
            
            fig.add_trace(go.Scatter(
                x=[row['median_x']], y=[row['median_y']],
                mode='markers+text',
                marker=dict(
                    color=node_color, # Colore dinamico
                    size=row['marker_size'],
                    symbol=marker_symbol,
                    line=dict(color=marker_line_color, width=2)
                ),
                text=jersey_text,
                textfont=dict(color=text_color, size=12, weight='bold'),
                hovertext=hover_text, hoverinfo='text', name=row['playerName']
            ))

    # --- Layout Finale con dimensioni maggiori ---
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="#2E3439",
        margin=dict(l=10, r=10, t=40, b=10),
        height=700,  # <-- MODIFICA: Altezza del grafico aumentata
        xaxis=dict(showgrid=False, zeroline=False, visible=False, fixedrange=True),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, fixedrange=True, scaleanchor="x", scaleratio=0.68)
    )

    return fig

def plot_ppda_plotly(
    ppda_value,
    df_def_actions,
    df_opponent_passes,
    team_name,
    team_color,
    opponent_color,
    is_away=False,
    pass_zone_threshold=60.0,
    defensive_zone_threshold=40.0,
):
    """Plot the two PPDA zones and the defensive actions used in the denominator."""
    fig = go.Figure()
    fig = draw_plotly_pitch(fig)

    # --- 1. Gestione assi (con inversione per away team) ---
    if is_away:
        fig.update_xaxes(range=[100, 0])
        fig.update_yaxes(range=[100, 0])
    else:
        fig.update_xaxes(range=[0, 100])
        fig.update_yaxes(range=[0, 100])

    # The numerator and denominator overlap between x=40 and x=60 by design.
    fig.add_shape(
        type="rect",
        x0=0, y0=0,
        x1=pass_zone_threshold, y1=100,
        fillcolor=opponent_color,
        opacity=0.055,
        layer="below",
        line_width=0,
    )
    fig.add_shape(
        type="rect",
        x0=defensive_zone_threshold, y0=0,
        x1=100, y1=100,
        fillcolor=team_color,
        opacity=0.075,
        layer="below",
        line_width=0
    )
    fig.add_vline(x=pass_zone_threshold, line_width=1.5, line_dash="dot", line_color="#8296a6")
    fig.add_vline(x=defensive_zone_threshold, line_width=1.5, line_dash="dash", line_color=team_color)

    # Opponent pass starts are deliberately quiet: they provide denominator context
    # without competing with the pressing actions.
    if not df_opponent_passes.empty:
        fig.add_trace(go.Scattergl(
            x=df_opponent_passes['x'], y=df_opponent_passes['y'],
            mode='markers',
            marker=dict(color="#8799a7", size=3.5, opacity=0.2),
            name='Opponent pass starts',
            hoverinfo='skip',
        ))

    if not df_def_actions.empty:
        player_names = df_def_actions.get('playerName', pd.Series('Unknown', index=df_def_actions.index)).fillna('Unknown')
        action_names = df_def_actions.get('type_name', pd.Series('Defensive action', index=df_def_actions.index)).fillna('Defensive action')
        hover_text = [
            f"<b>{action}</b><br>{player}<br>x={x:.1f}, y={y:.1f}"
            for action, player, x, y in zip(
                action_names,
                player_names,
                pd.to_numeric(df_def_actions['x'], errors='coerce').fillna(0),
                pd.to_numeric(df_def_actions['y'], errors='coerce').fillna(0),
            )
        ]
        fig.add_trace(go.Scatter(
            x=df_def_actions['x'],
            y=df_def_actions['y'],
            mode='markers',
            marker=dict(
                color=team_color,
                size=10,
                opacity=0.9,
                line=dict(width=1.5, color='white')
            ),
            hoverinfo='text',
            hovertext=hover_text,
            name=f'{team_name} pressing actions',
        ))

    ppda_text = f"{ppda_value:.2f}" if ppda_value != float('inf') else "N/A"
    pass_count = len(df_opponent_passes)
    action_count = len(df_def_actions)
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{team_name}</b> · PPDA {ppda_text}"
                f"<br><span style='font-size:11px;color:#657d8f'>"
                f"{pass_count} opponent passes ÷ {action_count} pressing actions</span>"
            ),
            x=0.5,
            xanchor='center',
        ),
        font=dict(color='#18344d', family='Arial'),
        showlegend=False,
        plot_bgcolor="#f8fbfc",
        paper_bgcolor="white",
        margin=dict(l=18, r=18, t=78, b=18),
        height=520,
        xaxis=dict(showgrid=False, zeroline=False, visible=False, fixedrange=True),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, fixedrange=True, scaleanchor="x", scaleratio=0.68)
    )

    fig.add_annotation(
        x=pass_zone_threshold - 1,
        y=98,
        text=f"Pass zone · x &lt; {pass_zone_threshold:g}",
        showarrow=False,
        xanchor='right',
        font=dict(size=10, color='#657d8f'),
        bgcolor='rgba(255,255,255,0.8)',
    )
    fig.add_annotation(
        x=defensive_zone_threshold + 1,
        y=2,
        text=f"Action zone · x ≥ {defensive_zone_threshold:g}",
        showarrow=False,
        xanchor='left',
        font=dict(size=10, color='#657d8f'),
        bgcolor='rgba(255,255,255,0.8)',
    )

    return fig


def plot_ppda_timeline(
    home_profile,
    away_profile,
    key_events,
    home_team,
    away_team,
    home_color,
    away_color,
):
    """Compare fixed 15-minute pressing rates with goals and dismissals."""
    fig = go.Figure()

    for profile, team_name, color in (
        (home_profile, home_team, home_color),
        (away_profile, away_team, away_color),
    ):
        timeline = profile.get('timeline', pd.DataFrame())
        if timeline.empty:
            continue
        timeline = timeline.sort_values('minute').copy()
        timeline['ppda_label'] = timeline['ppda'].map(
            lambda value: f"{value:.2f}" if pd.notna(value) and np.isfinite(value) else "N/A"
        )
        timeline['sample_label'] = np.where(
            timeline['low_sample'],
            'Limited sample · fewer than 2 pressing actions',
            'Stable sample',
        )
        customdata = np.column_stack([
            timeline['interval_label'],
            timeline['ppda_label'],
            timeline['opponent_passes'],
            timeline['defensive_actions'],
            timeline['sample_label'],
        ])
        fig.add_trace(go.Bar(
            x=timeline['minute'],
            y=timeline['pressure_rate'],
            width=5.8,
            name=team_name,
            legendgroup=team_name,
            offsetgroup=team_name,
            marker=dict(
                color=color,
                opacity=0.88,
                line=dict(color='white', width=1.3),
            ),
            text=timeline['pressure_rate'].map(lambda value: f"{value:.1f}"),
            textposition='outside',
            textfont=dict(size=10, color='#18344d'),
            cliponaxis=False,
            customdata=customdata,
            hovertemplate=(
                f"<b>{team_name}</b><br>"
                "%{customdata[0]}<br>"
                "Pressing intensity: <b>%{y:.1f}</b> actions per 100 opponent passes<br>"
                "Official PPDA: %{customdata[1]}<br>"
                "Opponent passes: %{customdata[2]}<br>"
                "Pressing actions: %{customdata[3]}<br>"
                "%{customdata[4]}<extra></extra>"
            ),
        ))

    fig.add_vline(x=45, line_width=1.5, line_dash='dash', line_color='#8ba0af')
    fig.add_annotation(
        x=45,
        y=1.02,
        xref='x',
        yref='paper',
        text='Half-time',
        showarrow=False,
        font=dict(size=10, color='#647c8e'),
        bgcolor='white',
    )

    if key_events is not None and not key_events.empty:
        for event_index, (_, event) in enumerate(key_events.iterrows()):
            minute = float(event['minute'])
            is_red = event['event_type'] == 'red_card'
            event_color = '#d64141' if is_red else '#d89100'
            event_symbol = '■' if is_red else '⚽'
            fig.add_vline(
                x=minute,
                line_width=1.2,
                line_dash='dot',
                line_color=event_color,
                opacity=0.72,
            )
            fig.add_annotation(
                x=minute,
                y=1.105 + (event_index % 2) * 0.065,
                xref='x',
                yref='paper',
                text=f"{event_symbol} {minute:.0f}'",
                showarrow=False,
                font=dict(size=10, color=event_color),
                hovertext=event['label'],
                bgcolor='rgba(255,255,255,0.94)',
                bordercolor=event_color,
                borderwidth=1,
                borderpad=3,
            )

    fig.update_layout(
        height=430,
        margin=dict(l=58, r=28, t=86, b=58),
        paper_bgcolor='white',
        plot_bgcolor='#f8fbfc',
        font=dict(color='#18344d', family='Arial'),
        hovermode='x unified',
        barmode='group',
        bargap=0.30,
        bargroupgap=0.08,
        legend=dict(
            orientation='h',
            x=0,
            y=1.18,
            xanchor='left',
            yanchor='bottom',
            bgcolor='rgba(255,255,255,0.8)',
        ),
        xaxis=dict(
            title='Match phase',
            range=[0, max(90, float(key_events['minute'].max()) + 3) if key_events is not None and not key_events.empty else 90],
            tickmode='array',
            tickvals=[7.5, 22.5, 37.5, 52.5, 67.5, 82.5],
            ticktext=['0–15', '15–30', '30–45+', '45–60', '60–75', '75–90+'],
            gridcolor='#e5edf1',
            zeroline=False,
        ),
        yaxis=dict(
            title='Pressing actions per 100 opponent passes · higher = more intense',
            rangemode='tozero',
            gridcolor='#e5edf1',
            zeroline=False,
        ),
    )
    fig.add_annotation(
        x=0,
        y=1.025,
        xref='paper',
        yref='paper',
        text='MORE INTENSE PRESSURE ↑',
        showarrow=False,
        xanchor='left',
        font=dict(size=10, color='#15845f'),
        bgcolor='rgba(231,247,240,0.92)',
        bordercolor='#b9e2d2',
        borderwidth=1,
        borderpad=4,
    )
    return fig
