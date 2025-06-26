# src/visualization/league_plots.py
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_quadrant_plot(df, x_metric, y_metric, invert_y=False, quadrant_labels=None, template="plotly_dark"):
    """
    Crea uno scatter plot a quadranti con sfondi colorati, linea di tendenza e loghi.
    """
    if x_metric not in df.columns or y_metric not in df.columns:
        return go.Figure().update_layout(
            title_text=f"Error: One or more metrics not found ('{x_metric}', '{y_metric}')", 
            template=template,
            font=dict(color='red' if template == 'plotly_dark' else 'black')
        )

    # Rimuovi eventuali righe con dati mancanti per le metriche selezionate
    df_plot = df.dropna(subset=[x_metric, y_metric]).copy()
    if df_plot.empty:
        return go.Figure().update_layout(title_text="No data available for the selected metrics.", template=template)

    x_mean = df_plot[x_metric].mean()
    y_mean = df_plot[y_metric].mean()
    
    fig = go.Figure()

    # Punti delle squadre
    text_color = 'black' if template == 'plotly_white' else 'white'
    for i, row in df_plot.iterrows():
        fig.add_trace(go.Scatter(
            x=[row[x_metric]], y=[row[y_metric]],
            mode='markers+text',
            marker=dict(size=14, line=dict(width=1, color=text_color)),
            text=row['equipo'], textposition="top center", textfont=dict(size=11, color=text_color),
            name=row['equipo'],
            hoverinfo='text',
            hovertext=f"<b>{row['equipo']}</b><br>{x_metric}: {row[x_metric]:.2f}<br>{y_metric}: {row[y_metric]:.2f}"
        ))

    if invert_y:
        fig.update_yaxes(autorange="reversed")

    fig.update_layout(template=template, showlegend=False)

    # Calcola i limiti degli assi X e Y con un margine extra
    x_min, x_max = df_plot[x_metric].min(), df_plot[x_metric].max()
    y_min, y_max = df_plot[y_metric].min(), df_plot[y_metric].max()
    x_margin = (x_max - x_min) * 0.1 if x_max > x_min else 1
    y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 1
    x_axis_range = [x_min - x_margin, x_max + x_margin]
    y_axis_range = [y_min - y_margin, y_max + y_margin]
    fig.update_xaxes(range=x_axis_range)
    fig.update_yaxes(range=y_axis_range)

    
    # --- 1. Aggiungi i quadranti colorati ---
    quadrant_colors = {
        'top_right': 'rgba(46, 204, 113, 0.1)',   # Verde (Buono/Buono)
        'bottom_right': 'rgba(241, 196, 15, 0.1)',# Giallo (Buono/Cattivo)
        'bottom_left': 'rgba(231, 76, 60, 0.1)', # Rosso (Cattivo/Cattivo)
        'top_left': 'rgba(52, 152, 219, 0.1)',   # Blu (Cattivo/Buono)
    }

    fig.add_shape(type="rect", x0=x_mean, y0=y_mean, x1=x_axis_range[1], y1=y_axis_range[1], fillcolor=quadrant_colors['top_right'], layer="below", line_width=0)
    fig.add_shape(type="rect", x0=x_mean, y0=y_axis_range[0], x1=x_axis_range[1], y1=y_mean, fillcolor=quadrant_colors['bottom_right'], layer="below", line_width=0)
    fig.add_shape(type="rect", x0=x_axis_range[0], y0=y_axis_range[0], x1=x_mean, y1=y_mean, fillcolor=quadrant_colors['bottom_left'], layer="below", line_width=0)
    fig.add_shape(type="rect", x0=x_axis_range[0], y0=y_mean, x1=x_mean, y1=y_axis_range[1], fillcolor=quadrant_colors['top_left'], layer="below", line_width=0)

    # Sposta i label dei quadranti più vicino alle estremità
    if quadrant_labels:
        x_pos_right = x_axis_range[0] + 0.995 * (x_axis_range[1] - x_axis_range[0])
        y_pos_top = y_axis_range[0] + 0.995 * (y_axis_range[1] - y_axis_range[0])
        x_pos_left = x_axis_range[0] + 0.005 * (x_axis_range[1] - x_axis_range[0])
        y_pos_bottom = y_axis_range[0] + 0.005 * (y_axis_range[1] - y_axis_range[0])
        if invert_y:
            y_pos_top, y_pos_bottom = y_pos_bottom, y_pos_top

        fig.add_annotation(x=x_pos_right, y=y_pos_top, xanchor='right', yanchor='top', text=f"<b>{quadrant_labels[0]}</b>", showarrow=False, font=dict(color='white', size=14), bgcolor='rgba(0,0,0,0.5)')
        fig.add_annotation(x=x_pos_right, y=y_pos_bottom, xanchor='right', yanchor='bottom', text=f"<b>{quadrant_labels[1]}</b>", showarrow=False, font=dict(color='white', size=14), bgcolor='rgba(0,0,0,0.5)')
        fig.add_annotation(x=x_pos_left, y=y_pos_bottom, xanchor='left', yanchor='bottom', text=f"<b>{quadrant_labels[2]}</b>", showarrow=False, font=dict(color='white', size=14), bgcolor='rgba(0,0,0,0.5)')
        fig.add_annotation(x=x_pos_left, y=y_pos_top, xanchor='left', yanchor='top', text=f"<b>{quadrant_labels[3]}</b>", showarrow=False, font=dict(color='white', size=14), bgcolor='rgba(0,0,0,0.5)')

    
    # --- 2. Aggiungi la linea di tendenza (regressione lineare) ---
    
    coeffs = np.polyfit(df_plot[x_metric], df_plot[y_metric], 1)
    trendline_y = np.polyval(coeffs, df_plot[x_metric])
    fig.add_trace(go.Scatter(
        x=df_plot[x_metric], y=trendline_y,
        mode='lines',
        name='Trendline',
        line=dict(color='blue', dash='longdash', width=2)
    ))

    # --- 3. Aggiungi le linee delle medie ---
    fig.add_shape(type='line', x0=x_axis_range[0], y0=y_mean, x1=x_axis_range[1], y1=y_mean, line=dict(dash='dash', color='grey', width=1))
    fig.add_shape(type='line', x0=x_mean, y0=y_axis_range[0], x1=x_mean, y1=y_axis_range[1], line=dict(dash='dash', color='grey', width=1))
   

    fig.update_layout(
        title=f"<b>{y_metric} vs. {x_metric}</b>",
        xaxis_title=x_metric,
        yaxis_title=y_metric,
        template=template,
        showlegend=False,
        font=dict(color=text_color)
    )
    
    if invert_y:
        fig.update_yaxes(autorange="reversed")
        
    return fig

def create_team_radar(df, team_names, template="plotly_dark"):
    """
    Crea un radar plot comparativo con le nuove metriche di stile.
    """
    
    # Rivediamo le categorie per includere le nuove metriche
    categories = {
        'Attacking': [
            'Goals', 
            'Goal Conversion', 
            'Total Shots'
        ],
        'Style': [
            'Possession Percentage',
            'Passing Tempo',
            'Pass vs Carry Index',
            'Short vs Long Ratio'
        ],
        'Defending': [
            'Goals Conceded', 
            'Tackles Won', 
            'Interceptions', 
            'Clean Sheets',
            'Aerial Duels Won %'
        ]
    }
    inverted_metrics = ['Goals Conceded']
    
    radar_metrics = []
    for cat_metrics in categories.values():
        radar_metrics.extend(cat_metrics)

    # Normalizzazione dati...
    df_norm = df.copy()
    for metric in radar_metrics:
        # Pulisci i nomi delle colonne da eventuali spazi extra
        if metric in df_norm.columns:
            min_val, max_val = df_norm[metric].min(), df_norm[metric].max()
            # Gestione del caso in cui tutti i valori sono uguali
            if (max_val - min_val) > 0:
                df_norm[metric] = (df_norm[metric] - min_val) / (max_val - min_val)
            else:
                df_norm[metric] = 0.5 # Assegna un valore medio se non c'è varianza

            # Inverti le metriche dove "meno è meglio"
            if metric in inverted_metrics:
                df_norm[metric] = 1 - df_norm[metric]
        else:
            print(f"Warning: Metric '{metric}' not found in DataFrame for radar plot.")
            df_norm[metric] = 0.5 # Assegna un valore neutro se la metrica manca
    
    fig = go.Figure()

    # Aggiungi le tracce delle squadre
    for team in team_names:
        team_data = df_norm[df_norm['equipo'] == team]
        if team_data.empty: continue
        team_data = team_data.iloc[0]
        
        values = [team_data.get(metric, 0.5) for metric in radar_metrics]
        
        fig.add_trace(go.Scatterpolar(
            r=values + values[:1],
            theta=radar_metrics + [radar_metrics[0]],
            fill='toself',
            name=team,
            hovertemplate='<b>%{theta}</b>: %{r:.2f}<extra></extra>'
        ))
        
    # Colori per gli sfondi
    category_colors = {
        'Attacking': 'rgba(231, 76, 60, 0.2)',   # Rosso
        'Style': 'rgba(52, 152, 219, 0.2)',  # Blu
        'Defending': 'rgba(46, 139, 87, 0.2)'    # Verde
    }
    
    bar_widths = []
    bar_colors = []
    for cat, metrics in categories.items():
        bar_widths.extend([1] * len(metrics))
        bar_colors.extend([category_colors[cat]] * len(metrics))

    fig.add_trace(go.Barpolar(
        r=[1] * len(radar_metrics), # barre alte fino al massimo
        theta=radar_metrics,
        width=bar_widths,
        marker_color=bar_colors,
        hoverinfo='none',
        showlegend=False,
        opacity=0.7
    ))
        
    fig.update_layout(
      polar=dict(
        radialaxis=dict(
            visible=True, range=[0, 1],
            showticklabels=True, # Ora mostriamo i tick su sfondo chiaro
            ticksuffix=" ", # Aggiunge spazio
            gridcolor='rgba(0, 0, 0, 0.2)'
        ),
        angularaxis=dict(
            tickfont=dict(size=11, color='black'), # Testo nero
            direction="clockwise"
        ),
        bgcolor='rgba(255, 255, 255, 0.5)' # Sfondo del radar semitrasparente chiaro
      ),
      template=template,
      showlegend=True,
      legend=dict(font=dict(color='black'))
    )
    
    return fig

# def create_team_radar(df, team_names):
#     """
#     Crea un radar plot comparativo con sfondi colorati per le categorie.
#     VERSIONE FINALE: Usa Barpolar per lo sfondo e posiziona correttamente le etichette.
#     """
    
#     categories = {
#         'Attacking': ['Goals', 'Shooting Accuracy', 'Successful Crosses & Corners', 'Goal Conversion'],
#         'Possession': ['Possession Percentage', 'Passing Accuracy', 'Successful Dribbles', 'Successful Passes Opposition Half'],
#         'Defending': ['Tackles Won', 'Interceptions', 'Recoveries', 'Clean Sheets', 'Goals Conceded']
#     }
#     inverted_metrics = ['Goals Conceded']
    
#     # Costruisci la lista completa delle metriche nell'ordine desiderato
#     radar_metrics = []
#     for cat_metrics in categories.values():
#         radar_metrics.extend(cat_metrics)
    
#     # Normalizzazione dati... (questa parte rimane uguale)
#     df_norm = df.copy()
#     for metric in radar_metrics:
#         if metric in df_norm.columns:
#             min_val, max_val = df_norm[metric].min(), df_norm[metric].max()
#             if (max_val - min_val) > 0:
#                 df_norm[metric] = (df_norm[metric] - min_val) / (max_val - min_val)
#                 if metric in inverted_metrics:
#                     df_norm[metric] = 1 - df_norm[metric]
#             else:
#                 df_norm[metric] = 0.5
#         else:
#             df_norm[metric] = 0.5

#     fig = go.Figure()

#     # --- 1. Aggiungi gli sfondi colorati con Barpolar PRIMA dei dati ---
#     category_colors = {
#         'Attacking': 'rgba(231, 76, 60, 0.2)',
#         'Possession': 'rgba(52, 152, 219, 0.2)',
#         'Defending': 'rgba(46, 139, 87, 0.3)'
#     }
    
#     bar_widths = []
#     bar_colors = []
#     for cat, metrics in categories.items():
#         bar_widths.extend([len(metrics)]) # Una barra per categoria, larga quanto il numero di metriche
#         bar_colors.append(category_colors[cat])

#     fig.add_trace(go.Barpolar(
#         r=[1] * len(categories), # Un valore per ogni categoria
#         # Usiamo i nomi delle categorie come theta per il posizionamento, ma non li mostreremo
#         theta=[cat[0] for cat in categories.values()],
#         width=bar_widths,
#         marker_color=bar_colors,
#         hoverinfo='none',
#         showlegend=False,
#         base=0,
#         opacity=0.4
#     ))

#     # --- 2. Aggiungi le tracce dei dati delle squadre ---
#     for team in team_names:
#         team_data = df_norm[df_norm['equipo'] == team]
#         if team_data.empty: continue
#         team_data = team_data.iloc[0]
        
#         values = [team_data.get(metric, 0.5) for metric in radar_metrics]
        
#         fig.add_trace(go.Scatterpolar(
#             r=values + values[:1], # Chiudi il poligono
#             theta=radar_metrics + [radar_metrics[0]], # Chiudi il poligono
#             fill='toself',
#             name=team,
#             hovertemplate='<b>%{theta}</b><br>Value: %{r:.2f}<extra></extra>'
#         ))

#     # --- 3. Aggiungi le etichette delle categorie con annotazioni posizionate meglio ---
#     annotations = []
#     num_total_metrics = len(radar_metrics)
#     angle_step = 360 / num_total_metrics
    
#     current_angle_idx = 0
#     for category, metrics in categories.items():
#         # Calcola l'angolo medio per la categoria
#         num_cat_metrics = len(metrics)
#         # L'angolo è a metà del settore della categoria
#         angle_deg = (current_angle_idx + num_cat_metrics / 2 - 0.5) * angle_step
        
#         annotations.append(go.layout.Annotation(
#             x=0.5 + 0.55 * np.cos(np.deg2rad(-angle_deg + 90)), # Converti in coordinate cartesiane
#             y=0.5 + 0.55 * np.sin(np.deg2rad(-angle_deg + 90)),
#             xref="paper", yref="paper",
#             text=f"<b>{category.upper()}</b>",
#             showarrow=False,
#             font=dict(size=14, color=category_colors[category].replace('0.2', '1.0').replace('0.3', '1.0'))
#         ))
#         current_angle_idx += num_cat_metrics
        
#     fig.update_layout(
#       polar=dict(
#         radialaxis=dict(
#             visible=True, 
#             range=[0, 1],
#             showticklabels=False, 
#             ticks='',
#             gridcolor='rgba(255, 255, 255, 0.2)'
#         ),
#         angularaxis=dict(
#             tickfont=dict(size=11, color='white'),
#             direction="clockwise"
#         ),
#         bgcolor='rgba(46, 52, 57, 0.0)' # Sfondo del radar trasparente per vedere i settori
#       ),
#       showlegend=True,
#       template="plotly_dark",
#       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
#       barmode='overlay',
#       annotations=annotations
#     )
    
#     return fig