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
    is_away=False,
    grid_size=6
):
    x_coords, y_coords, hover_texts = [], [], []

    for seq in sequences:
        if seq.empty:
            continue
        first = seq.iloc[0]
        x = first['end_x'] if first['type_name'] in ('Pass') else first['x']
        y = first['end_y'] if first['type_name'] in ('Pass') else first['y']
        if is_away:
            x = 100 - x
            y = 100 - y
        x_coords.append(x)
        y_coords.append(y)
        hover_texts.append(first.get("sequence_outcome_type", "Unknown"))

    # Binning
    bin_edges = np.linspace(0, 100, grid_size + 1)
    heatmap, _, _ = np.histogram2d(x_coords, y_coords, bins=[bin_edges, bin_edges])
    total = heatmap.sum()
    heatmap_pct = heatmap / total * 100 if total > 0 else heatmap
    max_val = heatmap_pct.max() if heatmap_pct.max() > 0 else 1

    # Helper per rettangolo
    def rectangle(x0, x1, y0, y1):
        return {
            "x": [x0, x1, x1, x0, x0],
            "y": [y0, y0, y1, y1, y0]
        }

    fig = go.Figure()
    fig = draw_plotly_pitch(fig)  # Disegna il campo

    # Disegna i poligoni bin
    for i, x0 in enumerate(bin_edges[:-1]):
        x1 = bin_edges[i+1]
        for j, y0 in enumerate(bin_edges[:-1]):
            y1 = bin_edges[j+1]
            perc = heatmap_pct[i, j]
            intensity = perc / max_val  # Normalizzazione
            colorscale = 'Blues' if is_away else 'Reds'
            color = sample_colorscale(colorscale, [intensity])[0] if perc > 0 else "rgba(0,0,0,0)"
            poly = rectangle(x0, x1, y0, y1)

            # Riempimento colorato
            fig.add_trace(go.Scatter(
                x=poly["x"], y=poly["y"],
                fill="toself",
                mode="lines",
                fillcolor=color,
                line=dict(color='rgba(0,0,0,0.2)'),
                hoverinfo="skip",
                showlegend=False
            ))

            # Etichetta della % al centro del bin
            if perc > 1:
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                fig.add_trace(go.Scatter(
                    x=[cx], y=[cy],
                    mode="text",
                    text=[f"{perc:.1f}%"],
                    textfont=dict(size=15, color='white', weight='bold'),
                    showlegend=False,
                    hoverinfo="skip"
                ))

    # Eventi singoli con hover personalizzato
    fig.add_trace(go.Scatter(
        x=x_coords, y=y_coords,
        mode='markers',
        marker=dict(size=8, color='black'),
        text=hover_texts,
        hovertemplate="Outcome: %{text}<br>X: %{x:.1f}, Y: %{y:.1f}<extra></extra>",
        name="Loss Events"
    ))

    arrow_y = 50 
    if is_away:
        arrow_x_start = 125
        arrow_x_end = 105
        text_position = "middle left"
    else:
        arrow_x_start = -25
        arrow_x_end = -5
        text_position = "middle right"

    fig.add_trace(go.Scatter(
        x=[arrow_x_start, arrow_x_end],
        y=[arrow_y, arrow_y],
        mode="lines+markers",
        marker=dict(symbol="arrow", size=15, angleref="previous", color="black"),
        line=dict(color="black", width=3),
        # text=[None, direction_symbol],
        textposition=text_position,
        hoverinfo="skip",
        showlegend=False
    ))

    # Layout tipo pitch (senza assi visibili)
    fig.update_layout(
        title="Possession Loss Heatmap",
        title_font_color='black', title_x=0.5,
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, range=[0, 100]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, range=[0, 100], scaleanchor="x", scaleratio=0.68),
        margin=dict(l=10, r=10, t=40, b=10),
        height=600,
        showlegend=False
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

def plot_ppda_plotly(ppda_value, df_def_actions, df_opponent_passes, team_name, team_color, opponent_color, is_away=False, zone_threshold=40.0):
    """
    Crea un grafico interattivo del PPDA con due heatmap sovrapposte,
    puntini per le azioni, e un'area di pressione evidenziata.
    """
    fig = go.Figure()
    fig = draw_plotly_pitch(fig)

    # --- 1. Gestione assi (con inversione per away team) ---
    if is_away:
        fig.update_xaxes(range=[100, 0])
        fig.update_yaxes(range=[100, 0])
    else:
        fig.update_xaxes(range=[0, 100])
        fig.update_yaxes(range=[0, 100])

    # --- 2. Evidenzia l'area di pressione (zona di calcolo del PPDA) ---
    fig.add_shape(
        type="rect",
        x0=zone_threshold, y0=0,
        x1=100, y1=100,
        fillcolor=team_color,
        opacity=0.1,  # Molto leggero, solo per dare contesto
        layer="below",
        line_width=0
    )

    # 3. Heatmap dei passaggi dell'avversario (sfondo)
    if not df_opponent_passes.empty:
        fig.add_trace(go.Histogram2dContour(
            x=df_opponent_passes['x'], y=df_opponent_passes['y'],
            colorscale=[[0, 'rgba(0,0,0,0)'], [1, opponent_color]],
            showscale=False, contours=dict(coloring='fill', showlines=False),
            name=f'Opponent Passes', hoverinfo='none', opacity=0.3
        ))
    
    # 4. Heatmap delle azioni difensive (in primo piano)
    if not df_def_actions.empty:
        fig.add_trace(go.Histogram2dContour(
            x=df_def_actions['x'], y=df_def_actions['y'],
            colorscale=[[0, 'rgba(0,0,0,0)'], [1, team_color]],
            showscale=False, contours=dict(coloring='fill', showlines=False),
            name=f'{team_name} Def. Actions', hoverinfo='none', opacity=0.6
        ))

    # --- 5. Puntini per le singole azioni difensive ---
    if not df_def_actions.empty:
        fig.add_trace(go.Scatter(
            x=df_def_actions['x'],
            y=df_def_actions['y'],
            mode='markers',
            marker=dict(
                color='yellow',
                size=5,
                opacity=0.7,
                line=dict(width=1, color='black')
            ),
            hoverinfo='text',
            hovertext=df_def_actions['type_name'] + ' by ' + df_def_actions['playerName'],
            name='Defensive Actions'
        ))

    # 6. Etichetta con il valore PPDA (invariata)
    ppda_text = f"{ppda_value:.2f}" if ppda_value != float('inf') else "N/A"
    fig.add_annotation(
        x=0.5, y=1.05, xref="paper", yref="paper",
        text=f"<b>PPDA: {ppda_text}</b>",
        showarrow=False, font=dict(color="white", size=16, family="Arial"),
        bgcolor="rgba(46, 52, 57, 0.9)", bordercolor="white", borderwidth=1, borderpad=4
    )

    # 7. Layout
    fig.update_layout(
        title_text=f"{team_name} - Pressing Intensity (PPDA)",
        title_x=0.5, font=dict(color='white'),
        showlegend=False,
        plot_bgcolor="white", paper_bgcolor="#2E3439",
        margin=dict(l=10, r=10, t=80, b=10), height=700,
        xaxis=dict(showgrid=False, zeroline=False, visible=False, fixedrange=True),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, fixedrange=True, scaleanchor="x", scaleratio=0.68)
    )

    return fig