# src/visualization/cross_plots.py
import plotly.graph_objects as go
from .buildup_plotly import draw_plotly_pitch # Riusiamo il disegnatore
import pandas as pd
import numpy as np
from ..config import BG_COLOR, LINE_COLOR
from plotly.colors import sample_colorscale

# def plot_cross_heatmap(df_analyzed, location_type='origin', team_color='blue', grid_size=6):
#     """
#     Crea una heatmap per le origini ('origin') o destinazioni ('destination') dei cross.
#     """
#     if location_type == 'origin':
#         x_coords, y_coords = df_analyzed['x'], df_analyzed['y']
#         title = "Cross Origins Heatmap"
#     else:
#         x_coords, y_coords = df_analyzed['end_x'], df_analyzed['end_y']
#         title = "Cross Destinations Heatmap"

#     fig = go.Figure(go.Densitymapbox(
#         lon=x_coords, lat=y_coords,
#         radius=15,
#         colorscale=[[0, 'rgba(0,0,0,0)'], [1, team_color]],
#         showscale=False
#     ))
#     # Binning e calcolo heatmap (logica identica a quella difensiva)
#     bin_edges = np.linspace(0, 100, grid_size + 1)
#     heatmap, _, _ = np.histogram2d(x_coords, y_coords, bins=[bin_edges, bin_edges])
#     total = heatmap.sum()
#     heatmap_pct = heatmap / total * 100 if total > 0 else heatmap
#     max_val = heatmap_pct.max() if heatmap_pct.max() > 0 else 1

#     # Helper per rettangolo
#     def rectangle(x0, x1, y0, y1):
#         return {
#             "x": [x0, x1, x1, x0, x0],
#             "y": [y0, y0, y1, y1, y0]
#         }

#     fig = go.Figure()
#     draw_plotly_pitch(fig)
#     return fig

def plot_cross_heatmap(df_analyzed, location_type='origin', is_away=False, grid_size=6, selected_cross_id=None):
    """
    Crea una heatmap per le origini ('origin') o destinazioni ('destination') dei cross.
    """
    if df_analyzed.empty:
        return go.Figure().update_layout(title_text=f"No Cross {location_type.title()} Data", template="plotly_dark")
    
    if selected_cross_id:
        # Se un cross è selezionato, lavoriamo solo con quella riga
        df_plot = df_analyzed[df_analyzed['cross_id'] == selected_cross_id]
        if df_plot.empty: # Fallback nel caso l'ID non sia più nei dati filtrati
            df_plot = df_analyzed
            selected_cross_id = None # Annulla la selezione
    else:
        # Altrimenti, usiamo tutti i dati
        df_plot = df_analyzed


    if location_type == 'origin':
        x_coords, y_coords = df_plot['x'], df_plot['y']
        title = "Cross Origins Heatmap"
    else:
        x_coords, y_coords = df_plot['end_x'], df_plot['end_y']
        title = "Cross Destinations Heatmap"

    fig = go.Figure()
    draw_plotly_pitch(fig)

    if x_coords.empty or y_coords.empty:
        fig.add_annotation(text="No data to plot", showarrow=False, font=dict(color='white'))
        return fig

    # Binning e calcolo heatmap (logica identica a quella difensiva)
    if not selected_cross_id:
        bin_edges = np.linspace(0, 100, grid_size + 1)
        all_x = df_analyzed['x'] if location_type == 'origin' else df_analyzed['end_x']
        all_y = df_analyzed['y'] if location_type == 'origin' else df_analyzed['end_y']
        heatmap, _, _ = np.histogram2d(all_x, all_y, bins=[bin_edges, bin_edges])
        total = heatmap.sum()
        heatmap_pct = heatmap / total * 100 if total > 0 else heatmap
        max_val = heatmap_pct.max() if heatmap_pct.max() > 0 else 1

        # Helper per rettangolo
        def rectangle(x0, x1, y0, y1):
            return {
                "x": [x0, x1, x1, x0, x0],
                "y": [y0, y0, y1, y1, y0]
            }
        
        total_crosses = heatmap.sum()
        if total_crosses == 0:
            fig.add_annotation(text="No crosses in this area", showarrow=False, font=dict(color='white'))
            return fig
            

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

    #hover_texts = df_analyzed['Destination Zone']
    hover_texts = [
        f"<b>{row['playerName']}</b><br>{row['Foot']} Foot | {row['Swing']}" 
        for index, row in df_analyzed.iterrows()
    ]


    # Eventi singoli con hover personalizzato
    fig.add_trace(go.Scatter(
        x=x_coords, y=y_coords,
        mode='markers',
        marker=dict(
            color='yellow' if selected_cross_id else 'black',
            size=8,
            opacity=0.8,
            line=dict(color='black', width=2)
        ),
        hoverinfo='text',
        text=hover_texts,
        # hovertemplate="Outcome: %{text}<br>X: %{x:.1f}, Y: %{y:.1f}<extra></extra>",
        customdata=df_plot['cross_id'], # Passa l'ID per l'hover
        name='cross_points',
        showlegend=False
    ))

    # if selected_cross_id and not df_analyzed.empty:
    #     highlight_row = df_analyzed[df_analyzed['cross_id'] == selected_cross_id]
    #     if not highlight_row.empty:
    #         if location_type == 'origin':
    #             hx, hy = highlight_row.iloc[0]['x'], highlight_row.iloc[0]['y']
    #         else:
    #             hx, hy = highlight_row.iloc[0]['end_x'], highlight_row.iloc[0]['end_y']
            
    #         fig.add_trace(go.Scatter(
    #             x=[hx], y=[hy],
    #             mode='markers',
    #             marker=dict(color='yellow', size=14, line=dict(color='black', width=2)),
    #             hoverinfo='skip',
    #             name='highlighted_point'
    #         ))

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font_color='black',
        title_x=0.5,
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, range=[0, 100]), 
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, range=[0, 100], scaleanchor="x", scaleratio=0.68),
        margin=dict(l=10, r=10, t=40, b=10),
        height=600, # Altezza leggermente ridotta
        showlegend=False,
    )
    return fig

def plot_cross_sankey(df_analyzed):
    """
    Crea un grafico Sankey per mostrare il flusso da Origin Zone a Destination Zone.
    """
    if df_analyzed.empty:
        return go.Figure().update_layout(title="No data for Sankey plot")

    df_sankey = df_analyzed.groupby(['Origin Zone', 'Destination Zone']).size().reset_index(name='count')
    
    all_nodes = pd.concat([df_sankey['Origin Zone'], df_sankey['Destination Zone']]).unique()
    node_map = {node: i for i, node in enumerate(all_nodes)}
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
        ),
        link=dict(
            source=[node_map[origin] for origin in df_sankey['Origin Zone']],
            target=[node_map[dest] for dest in df_sankey['Destination Zone']],
            value=df_sankey['count']
        )
    )])
    fig.update_layout(title_text="Cross Flow: Origin to Destination", font_size=12)
    return fig