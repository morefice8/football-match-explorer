# in src/visualization/pitch_plots.py
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from src.visualization import pitch_plots # Non serve importare se stesso

def plot_pass_network_plotly(passes_between, avg_locs, team_name, team_color, sub_list, is_away=False):
    """
    Versione 5: Corregge la visualizzazione delle linee e migliora lo stile dei subentrati.
    """
    fig = go.Figure()
    pitch_shapes = pitch_plots.get_plotly_pitch_shapes("rgba(255,255,255,0.2)", "white")
    
    if avg_locs.empty:
        fig.add_annotation(text=f"No pass network data for {team_name}", showarrow=False, font=dict(size=16, color="orange"))
    else:
        # Copia i DataFrame per evitare SettingWithCopyWarning
        avg_locs = avg_locs.copy()
        passes_between = passes_between.copy()

        # Inverti le coordinate per il team away
        if is_away:
            avg_locs[['pass_avg_x', 'pass_avg_y']] = 100 - avg_locs[['pass_avg_x', 'pass_avg_y']]
            if not passes_between.empty:
                passes_between[['pass_avg_x', 'pass_avg_y', 'pass_avg_x_end', 'pass_avg_y_end']] = 100 - passes_between[['pass_avg_x', 'pass_avg_y', 'pass_avg_x_end', 'pass_avg_y_end']]
        
        # --- 1. Disegna le linee delle connessioni (se esistono) ---
        if not passes_between.empty:
            max_lw = 10
            max_count = passes_between['pass_count'].max() if not passes_between.empty else 1
            passes_between['linewidth'] = passes_between['pass_count'] / max_count * max_lw

            mid_x, mid_y, hover_texts = [], [], []
            for _, row in passes_between.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row['pass_avg_x'], row['pass_avg_x_end']],
                    y=[row['pass_avg_y'], row['pass_avg_y_end']],
                    mode='lines',
                    line=dict(width=row['linewidth'], color=team_color, shape='spline'),
                    opacity=0.6,
                    hoverinfo='none',
                    showlegend=False
                ))
                mid_x.append((row['pass_avg_x'] + row['pass_avg_x_end']) / 2)
                mid_y.append((row['pass_avg_y'] + row['pass_avg_y_end']) / 2)
                hover_texts.append(f"{row['player1']} <> {row['player2']}<br><b>{int(row['pass_count'])}</b> passes")

            fig.add_trace(go.Scatter(
                x=mid_x, y=mid_y, mode='markers',
                marker=dict(color=team_color, size=5, opacity=0),
                hoverinfo='text', hovertext=hover_texts, showlegend=False
            ))

        # --- 2. Disegna i nodi (giocatori) ---
        max_size = 60
        max_pass_count = avg_locs['pass_count'].max() if not avg_locs.empty else 1
        avg_locs['marker_size'] = avg_locs['pass_count'] / max_pass_count * max_size + 20
        
        starters_df = avg_locs[~avg_locs['playerName'].isin(sub_list)]
        subs_df = avg_locs[avg_locs['playerName'].isin(sub_list)]

        # Aggiungi i titolari (cerchi con bordo bianco)
        if not starters_df.empty:
            fig.add_trace(go.Scatter(
                x=starters_df['pass_avg_x'], y=starters_df['pass_avg_y'],
                mode='markers+text',
                text=[f"<b>{int(j)}</b>" if pd.notna(j) else '' for j in starters_df['jersey_number']],
                marker=dict(symbol='circle', color=team_color, size=starters_df['marker_size'], line=dict(width=2, color='white')),
                hovertext=starters_df['playerName'] + '<br>Passes made: ' + starters_df['pass_count'].astype(int).astype(str),
                hoverinfo='text', showlegend=False
            ))

        # Aggiungi i subentrati (diamanti con bordo giallo)
        if not subs_df.empty:
            fig.add_trace(go.Scatter(
                x=subs_df['pass_avg_x'], y=subs_df['pass_avg_y'],
                mode='markers+text',
                text=[f"<b>{int(j)}</b>" if pd.notna(j) else '' for j in subs_df['jersey_number']],
                marker=dict(
                    symbol='diamond',  # Simbolo diverso
                    color=team_color,
                    opacity=0.9, # Leggermente più opaco
                    size=subs_df['marker_size'],
                    line=dict(width=3, color='#FFFF00') # Bordo giallo e più spesso
                ),
                hovertext=subs_df['playerName'] + ' (sub)<br>Passes made: ' + subs_df['pass_count'].astype(int).astype(str),
                hoverinfo='text', showlegend=False
            ))

    # --- Layout Finale ---
    fig.update_layout(
        title=f"Pass Network - {team_name}",
        showlegend=False,
        shapes=pitch_shapes,
        xaxis=dict(range=[-2, 102], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-2, 102], showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='#2E3439',
        paper_bgcolor='#2E3439',
        font_color='white',
        height=700
    )
    return fig


def plot_progressive_passes_plotly(df_prog_passes, team_name, team_color, is_away=False):
    """
    Crea una mappa interattiva dei passaggi progressivi con Plotly.
    Mostra i passaggi per zona e le statistiche come annotazioni.
    """
    fig = go.Figure()
    pitch_shapes = pitch_plots.get_plotly_pitch_shapes("rgba(255,255,255,0.2)", "white")

    df_plot = df_prog_passes.copy()
    
    if df_plot.empty:
        # Gestione caso vuoto
        fig.update_layout(title=f"No Progressive Passes for {team_name}")
        return fig
    
    # La divisione in zone ora avviene sui dati già orientati correttamente
    left_passes = df_plot[df_plot['y'] >= 66.67]
    mid_passes = df_plot[(df_plot['y'] >= 33.33) & (df_plot['y'] < 66.67)]
    right_passes = df_plot[df_plot['y'] < 33.33]

    zones_data = [
        ("Left Channel", left_passes, 'rgba(31, 119, 180, 0.8)'),
        ("Central Channel", mid_passes, 'rgba(44, 160, 44, 0.8)'),
        ("Right Channel", right_passes, 'rgba(255, 127, 14, 0.8)')
    ]
    total_count = len(df_plot)

    for zone_name, zone_df, color in zones_data:
        if not zone_df.empty:
            x_coords, y_coords, hover_texts = [], [], []
            for _, p in zone_df.iterrows():
                x_coords.extend([p['x'], p['end_x'], None])
                y_coords.extend([p['y'], p['end_y'], None])
                hover_text = f"<b>{p['playerName']}</b> at {p.get('timeMin', '?')}'"
                hover_texts.extend([hover_text, hover_text, None])
            legend_name = f"{zone_name} ({len(zone_df)})"
            fig.add_trace(go.Scattergl(x=x_coords, y=y_coords, mode='lines', line=dict(color=color, width=2.5), name=legend_name, hoverinfo='text', hovertext=hover_texts))
            fig.add_trace(go.Scattergl(x=zone_df['end_x'], y=zone_df['end_y'], mode='markers', marker=dict(size=5, color=color), showlegend=False, hoverinfo='none'))

    # --- Annotazioni e Layout ---
    pitch_shapes.extend([
        dict(type="line", x0=0, y0=33.33, x1=100, y1=33.33, line=dict(color="grey", dash="dot")),
        dict(type="line", x0=0, y0=66.67, x1=100, y1=66.67, line=dict(color="grey", dash="dot"))
    ])
    
    annotation_x = 15
    annotations = []
    if total_count > 0:
        annotations.extend([
            dict(x=annotation_x, y=83, text=f"<b>{len(left_passes)}</b><br>({len(left_passes)/total_count:.0%})", showarrow=False, font=dict(size=14, color='white')),
            dict(x=annotation_x, y=50, text=f"<b>{len(mid_passes)}</b><br>({len(mid_passes)/total_count:.0%})", showarrow=False, font=dict(size=14, color='white')),
            dict(x=annotation_x, y=17, text=f"<b>{len(right_passes)}</b><br>({len(right_passes)/total_count:.0%})", showarrow=False, font=dict(size=14, color='white')),
        ])
            
    fig.update_layout(
        title=dict(text=f"<b>{team_name} - Progressive Passes ({total_count})</b>", font=dict(size=18, color='white'), x=0.5, y=0.98),
        showlegend=True,
        legend=dict(
            orientation="h", y=1.05, yanchor="top", x=0.5, xanchor="center",
            font=dict(color='white') # **Colore legenda bianco**
        ),
        shapes=pitch_shapes,
        annotations=annotations,
        xaxis=dict(range=[-2, 102], visible=False),
        yaxis=dict(range=[-2, 102], visible=False),
        plot_bgcolor='#2E3439', paper_bgcolor='#2E3439',
        height=700,
        margin=dict(l=20, r=20, t=80, b=20)
    )

    if is_away:
        fig.update_layout(xaxis_autorange="reversed", yaxis_autorange="reversed")

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
                hover_text = f"<b>{p['playerName']}</b> to {p.get('receiver', '?')}<br>Min {p.get('timeMin', '?')}'"
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

    if is_away:
        fig.update_layout(xaxis_autorange="reversed", yaxis_autorange="reversed")
    
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

    # Prepara i dati
    df_plot = passes_df.copy()
    if is_away:
        df_plot['x'] = 100 - df_plot['x']
        df_plot['y'] = 100 - df_plot['y']

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
    
    return fig

def plot_pass_density_plotly(passes_df, team_name, is_away=False):
    """
    Crea una mappa di densità (KDE) interattiva su un campo da calcio.
    """
    fig = go.Figure()
    pitch_shapes = pitch_plots.get_plotly_pitch_shapes()
    
    df_plot = passes_df.copy()
    if is_away:
        df_plot['x'] = 100 - df_plot['x']
        df_plot['y'] = 100 - df_plot['y']
    
    colorscale = 'Reds' if not is_away else 'Blues'

    if not df_plot.empty:
        fig.add_trace(go.Histogram2dContour(
            x=df_plot['x'], y=df_plot['y'],
            colorscale=colorscale, showscale=False, line_width=0, name='Density'
        ))
        fig.add_trace(go.Scatter(
            x=df_plot['x'], y=df_plot['y'], mode='markers',
            marker=dict(color='white', size=3, opacity=0.3),
            hoverinfo='none', showlegend=False
        ))

    fig.update_layout(
        title=dict(text=f"{team_name} - Pass Density (KDE)", font_color='white', x=0.5),
        shapes=pitch_shapes,
        xaxis=dict(range=[-2, 102], visible=False),
        yaxis=dict(range=[-2, 102], visible=False, scaleanchor="x", scaleratio=0.68),
        # plot_bgcolor='#2E3439', 
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='#2E3439',
        height=450, margin=dict(l=10, r=10, t=40, b=10), showlegend=False
    )
    return fig

def plot_pass_heatmap_plotly(passes_df, team_name, is_away=False):
    """
    Versione 3: Aggiunge bordi ai bin, punti di passaggio e mostra percentuali.
    """
    fig = go.Figure()
    pitch_shapes = pitch_plots.get_plotly_pitch_shapes("rgba(0, 0, 0, 0.5)")

    df_plot = passes_df.copy()
    if is_away:
        df_plot['x'] = 100 - df_plot['x']
        df_plot['y'] = 100 - df_plot['y']

    colorscale = 'Reds' if not is_away else 'Blues'

    if not df_plot.empty:
        total_passes = len(df_plot)
        x_bins, y_bins = np.linspace(0, 100, 7), np.linspace(0, 100, 6)
        counts, y_edges, x_edges = np.histogram2d(df_plot['y'], df_plot['x'], bins=[y_bins, x_bins])
        
        # Le percentuali vengono calcolate sui conteggi
        percentages = (counts / total_passes) * 100 if total_passes > 0 else counts
        
        # 1. Disegna la Heatmap con i bordi
        fig.add_trace(go.Heatmap(
            z=counts,
            x=(x_edges[:-1] + x_edges[1:]) / 2,
            y=(y_edges[:-1] + y_edges[1:]) / 2,
            colorscale=colorscale,
            colorbar=dict(
                title='Passes',
                tickfont=dict(
                    color='white' # Colore per i numeri (ticks) della colorbar
                ),
                title_font=dict(
                    color='white' # Colore per il titolo ("Passes") della colorbar
                )
            ),
            xgap=1, ygap=1 
        ))
        
        # 2. Aggiungi i Punti di Passaggio sopra la heatmap
        fig.add_trace(go.Scatter(
            x=df_plot['x'], y=df_plot['y'],
            mode='markers',
            marker=dict(color='black', size=3, opacity=0.4),
            hoverinfo='none', showlegend=False
        ))

        # 3. Aggiungi le etichette con le PERCENTUALI
        annotations = []
        for i, row in enumerate(percentages):
            for j, perc in enumerate(row):
                if perc > 0:
                    annotations.append(go.layout.Annotation(
                        x=(x_edges[j] + x_edges[j+1]) / 2,
                        y=(y_edges[i] + y_edges[i+1]) / 2,
                        text=f"<b>{perc:.0f}%</b>", # Mostra la percentuale
                        showarrow=False,
                        font=dict(color='white' if counts[i, j] > counts.max() / 1.8 else 'black', size=11)
                    ))
        fig.update_layout(annotations=annotations)
    
    # Il layout rimane quasi identico, ma ora le shapes sono sopra tutto
    fig.update_layout(
        title=dict(text="Pass Heatmap", font_color='white', x=0.5),
        shapes=pitch_shapes,
        xaxis=dict(range=[-2, 102], visible=False),
        yaxis=dict(range=[-2, 102], visible=False, scaleanchor="x", scaleratio=0.68),
        plot_bgcolor='#2E3439', paper_bgcolor='#2E3439',
        height=450, margin=dict(l=10, r=40, t=40, b=10), showlegend=False
    )
    
    # Forza le forme del campo ad essere sopra la heatmap
    for shape in fig.layout.shapes:
        shape.layer = 'above'

    return fig