# In src/visualization/team_plots.py
import plotly.graph_objects as go
import pandas as pd
from src.utils import formation_layouts 
from src.metrics.sportmonks import LOWER_IS_BETTER, TEAM_RADAR_GROUPS, uses_sportmonks

def create_team_profile_radar(df_for_normalization, primary_team_name, comparison_team_name=None, template="plotly_dark"):
    """
    Creates a comparative radar plot using a robust Barpolar background and a clean layout.
    """
    if uses_sportmonks(df_for_normalization):
        CATEGORIES = TEAM_RADAR_GROUPS
        inverted_columns = LOWER_IS_BETTER
    else:
        CATEGORIES = {
            'Attacking': {'Goals': 'Gls', 'Goal Conversion %': 'Goal_Conversion', 'Total Shots': 'Sh', 'xG per Shot': 'xG_per_Shot'},
            'Possession & Style': {'Possession %': 'Poss', 'Passing Tempo': 'Passing_Tempo', 'Progressions / Touch': 'Progressions_per_Touch', 'Take-On Success %': 'TakeOn_Success_Rate'},
            'Defending': {'Goals Conceded': 'GA', 'Tackles Won': 'TklW', 'Interceptions': 'Int', 'Aerial Duels Won %': 'Aerial_Duels_Won_Perc'}
        }
        inverted_columns = {'GA'}
    radar_metrics_ordered = [metric for category in CATEGORIES.values() for metric in category.keys()]
    
    # Data Normalization
    df_norm = df_for_normalization.copy()
    # print("\n--- DEBUG: GA column type ---")
    # print(df_norm["GA"].dtype)
    # print("Unique GA values:", df_norm["GA"].unique())

    # for display_name, col_name in [item for sublist in [list(v.items()) for v in CATEGORIES.values()] for item in sublist]:
    #     if col_name in df_norm.columns:
    #         df_norm[display_name] = df_norm[col_name].rank(pct=True)
    #         if display_name in inverted_metrics:
    #             df_norm[display_name] = 1 - df_norm[display_name]
    #     else:
    #         df_norm[display_name] = 0.5
    for display_name, col_name in [item for sublist in [list(v.items()) for v in CATEGORIES.values()] for item in sublist]:
        if col_name in df_norm.columns:
            col_data = df_norm[col_name].astype(float)
            if col_data.max() == col_data.min():
                # Avoid division by zero if all values are equal
                df_norm[display_name] = 0.5
            else:
                if col_name in inverted_columns:
                    normed = (col_data.max() - col_data) / (col_data.max() - col_data.min())
                else:
                    normed = (col_data - col_data.min()) / (col_data.max() - col_data.min())
                df_norm[display_name] = normed
        else:
            df_norm[display_name] = 0.5

    # 🔍 DEBUG 1: Visualizza i valori normalizzati per 'Goals Conceded' nella stessa lega del Napoli
    # napoli_league = df_for_normalization[df_for_normalization["Squad"] == "Napoli"]["League"].values[0]
    # print("\n--- DEBUG: Goals Conceded Normalized Values (League:", napoli_league, ") ---")
    # print(df_norm[df_for_normalization["League"] == napoli_league][["Squad", "GA", "Goals Conceded"]].sort_values("GA"))


    fig = go.Figure()

    # --- Add Background Sectors using Barpolar ---
    category_colors = {
        category: (
            'rgba(214, 39, 40, 0.2)' if category == 'Attacking'
            else 'rgba(31, 119, 180, 0.2)' if category.startswith('Possession')
            else 'rgba(44, 160, 44, 0.2)'
        )
        for category in CATEGORIES
    }
    # bar_colors = [color for category, metrics in CATEGORIES.items() for color in [category_colors[category]] * len(metrics)]
    # bar_widths = [360 / len(radar_metrics_ordered)] * len(radar_metrics_ordered)

    bar_widths = []
    bar_colors = []
    for cat, metrics in CATEGORIES.items():
        bar_widths.extend([1] * len(metrics))
        bar_colors.extend([category_colors[cat]] * len(metrics))

    fig.add_trace(go.Barpolar(
        r=[1] * len(radar_metrics_ordered),
        theta=radar_metrics_ordered,
        width=bar_widths,
        marker_color=bar_colors,
        marker_line_width=0,
        hoverinfo='none',
        showlegend=False,
        opacity=0.8
    ))

    # --- Add Team Traces ---
    teams_to_plot = [primary_team_name]
    if comparison_team_name:
        teams_to_plot.append(comparison_team_name)

    for team_name in teams_to_plot:
        team_data = df_norm[df_norm['Squad'] == team_name]
        if not team_data.empty:
            values = [team_data.iloc[0].get(metric, 0.5) for metric in radar_metrics_ordered]
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=radar_metrics_ordered + [radar_metrics_ordered[0]],
                fill='toself',
                name=team_name,
                hovertemplate='<b>%{theta}</b><br>Percentile Rank: %{r:.2f}<extra></extra>'
            ))
        # print(f"\n--- DEBUG: Radar Values for {team_name} ---")
        # for metric, value in zip(radar_metrics_ordered, values):
        #     print(f"{metric}: {value:.2f}")


    # --- Use a cleaner, less conflicting layout configuration ---
    fig.update_layout(
        height=700,
        template=template,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            angularaxis=dict(direction="clockwise")
        ),
        legend=dict(yanchor="top", y=1.1, xanchor="center", x=0.5, orientation="h"),
        title='Team Statistical Profile'
    )
    
    return fig



def _draw_pitch(fig, bg_color='#2E3439', line_color='rgba(255,255,255,0.5)'):
    """Adds football pitch shapes to a Plotly figure."""
    PITCH_WIDTH_OPTA, PITCH_HEIGHT_OPTA = 100, 100
    
    # Bordi e linea di metà campo
    fig.add_shape(type="rect", x0=0, y0=0, x1=PITCH_WIDTH_OPTA, y1=PITCH_HEIGHT_OPTA, line=dict(color=line_color, width=2))
    fig.add_shape(type="line", x0=50, y0=0, x1=50, y1=PITCH_HEIGHT_OPTA, line=dict(color=line_color, width=2))
    
    # Cerchio di centrocampo
    fig.add_shape(type="circle", x0=40.5, y0=40.5, x1=59.5, y1=59.5, line=dict(color=line_color, width=2))
    
    # Aree di rigore
    fig.add_shape(type="rect", x0=0, y0=21.1, x1=16.5, y1=78.9, line=dict(color=line_color, width=2))
    fig.add_shape(type="rect", x0=100, y0=21.1, x1=83.5, y1=78.9, line=dict(color=line_color, width=2))
    
    # Aree piccole
    fig.add_shape(type="rect", x0=0, y0=36.8, x1=5.5, y1=63.2, line=dict(color=line_color, width=2))
    fig.add_shape(type="rect", x0=100, y0=36.8, x1=94.5, y1=63.2, line=dict(color=line_color, width=2))
    
    fig.update_layout(
        xaxis=dict(range=[-2, 102], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-2, 102], showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        showlegend=False,
    )
    return fig

# --- FUNZIONE DI PLOT FORMAZIONE AGGIORNATA ---
def plot_most_used_formation_plotly(formation_id, team_color='skyblue', template="plotly_dark", height=400):
    """
    Creates a Plotly figure of a football pitch with dots representing a formation.
    Accepts a height parameter for flexible sizing.
    """

    # ... (il corpo della funzione è quasi identico, cambia solo l'uso di 'height')
    fig = go.Figure()
    fig = _draw_pitch(fig)

    coords = []
    if formation_id in formation_layouts.FORMATION_COORDINATES:
        formation_map = formation_layouts.FORMATION_COORDINATES[formation_id]
        for pos_num in range(1, 12):
            coord = formation_map.get(pos_num)
            if coord:
                coords.append(coord)

    if coords:
        x_coords, y_coords = zip(*coords)
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(color=team_color, size=20, line=dict(width=2, color='white')),
            hoverinfo='none'
        ))

    fig.update_layout(
        height=height, # <--- Usa il parametro height
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig
