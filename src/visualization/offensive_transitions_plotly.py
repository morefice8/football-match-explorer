# src/visualization/offensive_transitions_plotly.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from .defensive_transitions_plotly import draw_plotly_pitch # Riusiamo il disegnatore del campo
from ..config import BG_COLOR

def plot_recovery_heatmap_on_pitch(sequences, is_away=False, grid_size=6):
    """
    Crea una heatmap Plotly che mostra dove sono avvenuti i recuperi palla
    che hanno dato inizio a una transizione offensiva.
    """
    x_coords, y_coords, hover_texts, hover_outcomes = [], [], [], []

    for seq in sequences:
        if seq.empty:
            continue
        
        first_event = seq.iloc[0]
        # La coordinata del recupero è contenuta in 'loss_zone' e nelle coordinate dell'evento di perdita
        # La funzione find_buildup_after_possession_loss ci dà le coordinate della perdita avversaria,
        # che per noi sono le coordinate del recupero.
        
        # Prendiamo le coordinate dell'evento che ha scatenato la sequenza.
        # Dobbiamo capire se sono le coordinate di inizio o fine.
        trigger_event_type = first_event.get("type_of_initial_loss", "")
        
        # Se il recupero deriva da un passaggio sbagliato avversario, il recupero avviene su end_x/end_y
        # In find_buildup... abbiamo già calcolato la loss_zone in modo speculare.
        # Quindi usiamo le coordinate dell'evento scatenante nel DataFrame originale.
        # Per semplicità, ci basiamo sulla x,y del primo evento della *nostra* sequenza,
        # che è il primo controllo/passaggio dopo il recupero.
        x = first_event['end_x'] if first_event['type_name'] in ('Pass') else first_event['x']
        y = first_event['end_y'] if first_event['type_name'] in ('Pass') else first_event['y']

        # Non è necessario invertire le coordinate qui, perché le analizziamo
        # sempre dal nostro punto di vista (0-100 da sinistra a destra).
        if is_away:
            x = 100 - x
            y = 100 - y
        
        x_coords.append(x)
        y_coords.append(y)
        hover_texts.append(first_event.get("sequence_outcome_type", "Unknown"))
        hover_outcomes.append(first_event.get("sequence_outcome_type", "Unknown"))

    if not x_coords:
        fig = go.Figure()
        draw_plotly_pitch(fig)
        fig.add_annotation(text="No recovery data to plot", showarrow=False, font=dict(color="black"))
        return fig

    # Binning e calcolo heatmap (logica identica a quella difensiva)
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
    draw_plotly_pitch(fig)

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
        name="Loss Events",
        showlegend=False
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

    fig.update_layout(
        title="<b>Ball Recovery Heatmap</b>",
        title_font_color='black', title_x=0.5,
        plot_bgcolor=BG_COLOR,  # Usa il colore di sfondo della tua app
        paper_bgcolor=BG_COLOR, # Usa il colore di sfondo della tua app
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, range=[0, 100]), 
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, range=[0, 100], scaleanchor="x", scaleratio=0.68),
        margin=dict(l=10, r=10, t=40, b=10),
        height=600, # Altezza leggermente ridotta
        showlegend=False,
        # shapes=[s.update(layer='below') for s in fig.layout.shapes] # Assicura che il campo sia sotto la heatmap
    )

    return fig