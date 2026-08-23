# In un nuovo file, es: src/visualization/formation_plotly.py
import logging
logger = logging.getLogger(__name__)

import plotly.graph_objects as go
import pandas as pd

from src.utils.formation_layouts import get_formation_layout_coords, get_formation_name, FORMATION_COORDINATES
from .buildup_plotly import draw_plotly_pitch
from src.visualization.plotly_branding import (
    MATCH_COMPARE_HEIGHT,
    MATCH_PITCH_MUTED,
    MATCH_PITCH_TEXT,
    MATCH_WARNING,
    add_attacking_direction,
    add_zero_state,
    apply_match_pitch_layout,
    get_team_palette,
)
from src.visualization import pitch_plots

# Colori e costanti (prendili dal tuo config.py)
BG_COLOR = 'rgba(46, 52, 57, 1)' # #2E3439 in RGBA per Plotly
LINE_COLOR = 'rgba(211, 211, 211, 0.7)' # Grigio chiaro per le linee
TEXT_COLOR = 'white'
PITCH_WIDTH_OPTA = 100
PITCH_HEIGHT_OPTA = 100

# def prepare_formation_time_series(df_events):
#     """
#     Restituisce:
#       times: list di secondi (float) o minuti
#       frames: list di dict {player_name: (x, y), ...}
#       formations: list dei formation_id attivi
#     """

#     # 1. Filtra solo gli eventi di interesse:
#     df = df_events[df_events['type_name'].isin(['Player on', 'Player off', 'Formation change'])].copy()
#     df['time_sec'] = df['timeMin'] * 60 + df['timeSec']
#     df.sort_values('time_sec', inplace=True)

#     # 2. Inizializza:
#     current_players = {}  # {player_id: (name, x, y)}
#     current_formation = None

#     times, frames, formations = [], [], []

#     # 3. Cicla su ogni "momento chiave":
#     for t, grp in df.groupby('time_sec'):
#         # aggiorna formazione se presente
#         fc = grp.loc[grp['type_name'] == 'Formation change', 'Team formation']
#         if not fc.empty:
#             current_formation = fc.iloc[-1]

#         # sostituzioni
#         for ev in grp.itertuples():
#             if ev.type_name == 'Player on':
#                 current_players[ev.playerId] = (ev.playerName, ev.x, ev.y)
#             elif ev.type_name == 'Player off':
#                 current_players.pop(ev.playerId, None)

#         # snapshot
#         # alla posizione corrente, salviamo solo chi c'è
#         frames.append({pid: (name, x, y)
#                        for pid, (name, x, y) in current_players.items()})
#         times.append(t)
#         formations.append(current_formation)
#         print(f"At {t}s: {len(current_players)} players, Formation ID: {current_formation}")
    
#     print(f"Prepared {len(times)} time points with {len(frames)} frames and {len(set(formations))} unique formations.")

#     return times, frames, formations

def prepare_formation_timeline(df, team_name):
    df_team = df[df["team_name"] == team_name].copy()

    # Assicura i tipi giusti
    df_team["timeMin"] = pd.to_numeric(df_team["timeMin"], errors="coerce")
    df_team["Mapped Position Number"] = pd.to_numeric(df_team["Mapped Position Number"], errors="coerce")
    df_team["Team formation"] = df_team["Team formation"].ffill().bfill()

    timeline = []
    current_players = {}
    current_formation = None

    # 1. Giocatori iniziali
    starters = df_team[df_team["Is Starter"] == True]
    for _, row in starters.iterrows():
        pid = row["playerId"]
        current_players[pid] = {
            "playerName": row["playerName"],
            "mapped_position": int(row["Mapped Position Number"])
        }
    current_formation = starters["Team formation"].iloc[0] if not starters.empty else None
    timeline.append({
        "minute": 0,
        "formation_id": current_formation,
        "players": current_players.copy()
    })

    # 2. Cambi
    subs = df_team[df_team["typeId"] == 19].sort_values("timeMin")
    for _, row in subs.iterrows():
        pid = row["playerId"]
        name = row["playerName"]
        
        # Usa Mapped Position se presente, altrimenti Formation slot
        if int(row["Mapped Position Number"]) > 0:
            pos = int(row["Mapped Position Number"])
        elif pd.notna(row.get("Formation slot")):
            pos = int(row["Formation slot"])
        else:
            continue  # ignora se non c'è posizione valida
        
        minute = int(row["timeMin"])
        formation = row["Team formation"] if not pd.isna(row["Team formation"]) else current_formation

        current_players[pid] = {
            "playerName": name,
            "mapped_position": pos
        }
        current_formation = formation
        timeline.append({
            "minute": minute,
            "formation_id": current_formation,
            "players": current_players.copy()
        })

    return timeline

# def prepare_formation_time_series(df_events, team_id):
#     """
#     Costruisce una timeline dinamica della formazione in base ai cambi (Player On) e ai cambi di modulo.
#     """
#     timeline = []
#     current_players = {}
#     current_formation_id = None

#     # 1. Starting XI
#     starters_df = df_events[(df_events["team_name"] == team_id) & (df_events["Is Starter"] == True)]
#     for _, row in starters_df.iterrows():
#         player_id = row["playerId"]
#         player_name = row["playerName"]
#         mapped_position = row.get("Mapped Position Number", 0)
#         current_players[player_id] = {
#             "playerName": player_name,
#             "mapped_position": int(mapped_position)
#         }

#     # Aggiungiamo snapshot iniziale al minuto 0
#     timeline.append({
#         "time": 0,
#         "formation_id": None,
#         "players": current_players.copy()
#     })

#     # 2. Ordina gli eventi cronologicamente
#     df_sorted = df_events[df_events["team_name"] == team_id].sort_values("timeMin")

#     for _, row in df_sorted.iterrows():
#         type_id = row["typeId"]
#         minute = row["timeMin"]
#         player_id = row["playerId"]
#         player_name = row["playerName"]

#         if type_id == 19:  # Player On
#             mapped_position = row.get("Formation slot", 0)
#             current_players[player_id] = {
#                 "playerName": player_name,
#                 "mapped_position": int(mapped_position)
#             }
#             timeline.append({
#                 "time": minute,
#                 "formation_id": current_formation_id,
#                 "players": current_players.copy()
#             })

#         elif type_id == 40:  # Formation change
#             current_formation_id = int(row.get("Team formation", 0))
#             timeline.append({
#                 "time": minute,
#                 "formation_id": current_formation_id,
#                 "players": current_players.copy()
#             })
#     print(f"timeline : {timeline}")
#     return timeline


def create_frames_from_timeline(timeline, team_color="#1f77b4"):
    frames = []

    for snapshot in timeline:
        minute = int(snapshot["minute"])
        # formation_id = snapshot.get("formation_id")
        players = snapshot.get("players", {})

        # Skip se la formazione non è valida o non presente nei layout
        formation_id = snapshot.get("formation_id")
        if not formation_id:
            continue

        formation_id = int(formation_id)  # 👈 cast for dict key matching
        if formation_id not in FORMATION_COORDINATES:
            logger.warning(f"[DEBUG] Formation ID {formation_id} not found in FORMATION_COORDINATES.")
            continue

        coords = FORMATION_COORDINATES[formation_id]
        x, y, names = [], [], []

        for player_id, info in players.items():
            pos = info.get("mapped_position")

            # Skip se posizione non valida o non mappata
            if not pos or int(pos) not in coords:
                continue

            pos_index = int(pos)
            x.append(coords[pos_index][0])
            y.append(coords[pos_index][1])
            names.append(info.get("playerName", "Unknown"))

        # Costruisce il frame per il minuto corrente
        frame = go.Frame(
            name=str(minute),
            data=[
                go.Scatter(
                    x=x,
                    y=y,
                    mode='markers+text',
                    text=names,
                    textposition='top center',
                    marker=dict(size=20, color=team_color),
                    hoverinfo='text'
                )
            ]
        )
        frames.append(frame)

    logger.info(f"[DEBUG] Created {len(frames)} frames from timeline with {len(timeline)} snapshots.")
    return frames



def _draw_pitch(fig):
    """Aggiunge le forme del campo da calcio a una figura Plotly."""
    fig.add_shape(type="rect", x0=0, y0=0, x1=PITCH_WIDTH_OPTA, y1=PITCH_HEIGHT_OPTA, line=dict(color=LINE_COLOR, width=2), fillcolor="rgba(0,0,0,0)")
    fig.add_shape(type="line", x0=50, y0=0, x1=50, y1=PITCH_HEIGHT_OPTA, line=dict(color=LINE_COLOR, width=2))
    fig.add_shape(type="circle", x0=41.5, y0=41.5, x1=58.5, y1=58.5, line=dict(color=LINE_COLOR, width=2), xref="x", yref="y")
    # Aree di rigore
    fig.add_shape(type="rect", x0=0, y0=21.1, x1=16.5, y1=78.9, line=dict(color=LINE_COLOR, width=2))
    fig.add_shape(type="rect", x0=100, y0=21.1, x1=83.5, y1=78.9, line=dict(color=LINE_COLOR, width=2))
    # Aree piccole
    fig.add_shape(type="rect", x0=0, y0=36.8, x1=5.5, y1=63.2, line=dict(color=LINE_COLOR, width=2))
    fig.add_shape(type="rect", x0=100, y0=36.8, x1=94.5, y1=63.2, line=dict(color=LINE_COLOR, width=2))
    # Archi di rigore
    fig.add_shape(type="path", path="M 16.5,34.9 C 22.5,42 22.5,58 16.5,65.1", line_color=LINE_COLOR)
    fig.add_shape(type="path", path="M 83.5,34.9 C 77.5,42 77.5,58 83.5,65.1", line_color=LINE_COLOR)
    
    # Aggiorna il layout per rimuovere assi, griglia e impostare le dimensioni
    fig.update_layout(
        xaxis=dict(range=[-2, 102], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-2, 102], showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        showlegend=False,
        height=600,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig


# In src/visualization/formation_plotly.py
# ... (import e funzioni helper rimangono uguali)

def plot_formation_interactive(df_processed, match_info):
    """
    Crea un grafico interattivo delle formazioni con Plotly,
    inclusa una timeline per i cambi di formazione e le sostituzioni.
    """
    # 1. Estrai dati di base
    HTEAM_NAME = match_info.get('hteamName', 'Home')
    ATEAM_NAME = match_info.get('ateamName', 'Away')
    HCOL = match_info.get('hColor') or 'tomato'
    ACOL = match_info.get('aColor') or 'skyblue'

    # 2. Prepara i dati per la timeline (eventi chiave)
    timeline_events_df = df_processed[
        df_processed['typeId'].isin([18, 19, 34])
    ].sort_values(['timeMin', 'timeSec', 'eventId']).reset_index(drop=True)

    # Crea la figura di base e il campo
    fig = go.Figure()
    fig = _draw_pitch(fig)

    # 3. Dati iniziali
    starters = df_processed[df_processed['Is Starter'] == True].drop_duplicates(subset=['playerName', 'team_name'], keep='first')
    player_data_df = df_processed.drop_duplicates(subset=['playerName', 'team_name'], keep='first').set_index('playerName')

    # Trova le formazioni iniziali in modo robusto
    lineup_events = df_processed[df_processed['typeId'] == 34]
    home_start_event = lineup_events[(lineup_events['team_name'] == HTEAM_NAME) & (lineup_events['timeMin'] == 0)].iloc[0]
    away_start_event = lineup_events[(lineup_events['team_name'] == ATEAM_NAME) & (lineup_events['timeMin'] == 0)].iloc[0]
    
    # Estrai l'ID formazione dagli eventi specifici
    home_formation_id = int(home_start_event.get('Team formation', 0))
    away_formation_id = int(away_start_event.get('Team formation', 0))

    # Dizionario per tracciare lo stato attuale dei giocatori e delle formazioni
    game_state = {
        'minute': 0,
        'home_formation': home_formation_id,
        'away_formation': away_formation_id,
        'on_pitch': {
            p['playerName']: int(p['Mapped Position Number'])
            for _, p in starters.iterrows()
            if pd.notna(p['Mapped Position Number'])
        }
    }
    
    # 4. Aggiungi TUTTI i giocatori possibili al grafico (nascosti)
    all_players_list = player_data_df.reset_index().to_dict('records')
    for player in all_players_list:
        is_home = player['team_name'] == HTEAM_NAME
        team_color = HCOL if is_home else ACOL
        
        hover_text = (f"<b>{player['playerName']}</b><br>#{player.get('Mapped Jersey Number', '?')}<br>Role: {player.get('positional_role', 'N/A')}")
        jersey_num = str(player.get('Mapped Jersey Number', ''))

        fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', name=player['playerName'], marker=dict(color=team_color, size=25, line=dict(width=2, color=TEXT_COLOR)),
                                hoverinfo='text', hovertext=hover_text, visible=False))
        fig.add_trace(go.Scatter(x=[0], y=[0], mode='text', text=[jersey_num], textfont=dict(color=TEXT_COLOR, size=10, family="Arial, sans-serif"),
                                hoverinfo='none', visible=False))
        
    # 5. Configura lo slider
    steps = []
    # Primo step (minuto 0)
    home_name = get_formation_name(game_state['home_formation'])
    away_name = get_formation_name(game_state['away_formation'])
    title = f"Start: {HTEAM_NAME} ({home_name}) vs {ATEAM_NAME} ({away_name})"
    
    # Calcola visibilità e posizioni iniziali
    initial_visibility = [False] * len(fig.data)
    initial_positions_x = [[] for _ in fig.data]
    initial_positions_y = [[] for _ in fig.data]

    for i, player in enumerate(all_players_list):
        player_name = player['playerName']
        if player_name in game_state['on_pitch']:
            marker_idx, text_idx = i * 2, i * 2 + 1
            initial_visibility[marker_idx] = True
            initial_visibility[text_idx] = True
            
            is_away = player['team_name'] == ATEAM_NAME
            form_id = game_state['away_formation'] if is_away else game_state['home_formation']
            pos_num = game_state['on_pitch'][player_name]
            
            x, y = get_formation_layout_coords(form_id, pos_num)
            if x is not None and y is not None:
                if is_away: x, y = 100 - x, 100 - y
                initial_positions_x[marker_idx] = [x]
                initial_positions_y[marker_idx] = [y]
                initial_positions_x[text_idx] = [x]
                initial_positions_y[text_idx] = [y]

    # Aggiungi lo step iniziale
    steps.append(dict(method='restyle', args=[{'visible': initial_visibility, 'x': initial_positions_x, 'y': initial_positions_y}, {'title.text': title} ], label="0'"))

    # 6. Processa gli eventi per creare gli altri steps
    last_sub_off_info = {} # {timeMin: {'pos': posNum, 'team': teamName}}
    
    for _, event in timeline_events_df.iterrows():
        time_min = event['timeMin']
        event_team = event['team_name']
        
        if event['typeId'] == 18: # Player Off
            player_name_off = event['playerName']
            if player_name_off in game_state['on_pitch']:
                pos_off = game_state['on_pitch'].pop(player_name_off)
                last_sub_off_info[time_min] = {'pos': pos_off, 'team': event_team}
        
        elif event['typeId'] == 19: # Player On
            player_name_on = event['playerName']
            sub_info = last_sub_off_info.get(time_min, {'pos': 0, 'team': event_team})
            game_state['on_pitch'][player_name_on] = sub_info['pos']
            
        elif event['typeId'] == 34: # Formation Change
            new_form_id = int(event['Team formation'])
            if event_team == HTEAM_NAME: game_state['home_formation'] = new_form_id
            else: game_state['away_formation'] = new_form_id

        # Ricostruisci lo stato del grafico per questo step
        step_visibility = [False] * len(fig.data)
        step_positions_x = [[] for _ in fig.data]
        step_positions_y = [[] for _ in fig.data]
        
        for i, player in enumerate(all_players_list):
            player_name = player['playerName']
            if player_name in game_state['on_pitch']:
                marker_idx, text_idx = i * 2, i * 2 + 1
                step_visibility[marker_idx] = True
                step_visibility[text_idx] = True

                is_away = player['team_name'] == ATEAM_NAME
                form_id = game_state['away_formation'] if is_away else game_state['home_formation']
                pos_num = game_state['on_pitch'][player_name]
                
                x, y = get_formation_layout_coords(form_id, pos_num)
                if x is not None and y is not None:
                    if is_away: x, y = 100 - x, 100 - y
                    step_positions_x[marker_idx] = [x]
                    step_positions_y[marker_idx] = [y]
                    step_positions_x[text_idx] = [x]
                    step_positions_y[text_idx] = [y]

        home_name = get_formation_name(game_state['home_formation'])
        away_name = get_formation_name(game_state['away_formation'])
        title = f"Min {time_min}': {HTEAM_NAME} ({home_name}) vs {ATEAM_NAME} ({away_name})"

        steps.append(dict(method='restyle', args=[{'visible': step_visibility, 'x': step_positions_x, 'y': step_positions_y}, {'title.text': title}], label=f"{time_min}'"))
    
    # 7. Applica slider e titolo finale
    fig.update_layout(
        title=dict(text=f"Start: {HTEAM_NAME} ({get_formation_name(home_formation_id)}) vs {ATEAM_NAME} ({get_formation_name(away_formation_id)})", x=0.5, font=dict(color=TEXT_COLOR)),
        sliders=[dict(active=0, currentvalue={"prefix": "Timeline: "}, pad={"t": 50}, steps=steps)]
    )
    
    return fig

def plot_formation_interactive_with_timeline(timeline, team_color="#1f77b4"):
    """
    Plotta una formazione dinamica (frame-by-frame) con slider temporale.
    """
    fig = go.Figure()
    fig = _draw_pitch(fig)

    # Frame iniziale
    initial_frame = None
    for snapshot in timeline:
        logger.debug(f"[DEBUG] Processing snapshot: {snapshot}")
        formation_id = snapshot.get("formation_id")
        players = snapshot.get("players", {})
        logger.debug(f"[DEBUG] Formation ID: {formation_id}, Players: {players}")
        if not formation_id or formation_id not in FORMATION_COORDINATES:
            continue
        coords = FORMATION_COORDINATES[formation_id]
        x, y, names = [], [], []
        for player_id, info in players.items():
            pos = info.get("mapped_position")
            if not pos or int(pos) > len(coords):
                continue
            pos_index = int(pos)
            x.append(coords[pos_index][0])
            y.append(coords[pos_index][1])
            names.append(info.get("playerName", "Unknown"))
        initial_frame = dict(x=x, y=y, text=names)
        break

    if initial_frame:
        fig.add_trace(go.Scatter(
            x=initial_frame["x"],
            y=initial_frame["y"],
            mode='markers+text',
            marker=dict(size=20, color=team_color),
            text=initial_frame["text"],
            textposition='top center',
            hoverinfo='text',
            name="Players"
        ))

    # Aggiunge tutti i frame
    frames = create_frames_from_timeline(timeline, team_color)
    fig.frames = frames

    # Slider e layout
    fig.update_layout(
        title="Dynamic Formation Timeline",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 100]),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 100], scaleanchor="x", scaleratio=0.68),
        plot_bgcolor="white",
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.15,
            x=1.05,
            xanchor="right",
            yanchor="top",
            buttons=[dict(label="Play", method="animate", args=[None, {"frame": {"duration": 1000, "redraw": True}, "fromcurrent": True}]),
                     dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0}, "mode": "immediate", "transition": {"duration": 0}}])]
        )],
        sliders=[dict(
            steps=[dict(method="animate", args=[[f.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}], label=f.name)
                   for f in frames],
            transition={"duration": 0},
            x=0.1,
            xanchor="left",
            y=0,
            yanchor="top",
            len=0.9
        )],
        margin=dict(t=40, b=40, l=10, r=10)
    )
    return fig

def plot_mean_positions_plotly(df_all_touches, df_player_agg, team_color, is_away=False):
    """
    Creates an interactive mean positions plot, combining a heatmap of all touches
    with player average positions. Includes a properly placed average line label.
    """
    fig = go.Figure()
    fig = draw_plotly_pitch(fig)

    # Match Analysis coordinates are already team-relative.
    del is_away
    fig.update_xaxes(range=[0, 100])
    fig.update_yaxes(range=[0, 100])
    
    # 1. Heatmap di tutti i tocchi
    if not df_all_touches.empty:
        fig.add_trace(go.Histogram2dContour(
            x=df_all_touches['x'], y=df_all_touches['y'],
            colorscale=[[0, 'rgba(0,0,0,0)'], [1, team_color]],
            showscale=False, contours=dict(coloring='fill', showlines=False),
            name='Team Touches', hoverinfo='none', opacity=0.5
        ))
    
    # 2. Posizione media dei giocatori
    if not df_player_agg.empty:
        df_player_agg['marker_size'] = 45

        for i, row in df_player_agg.iterrows():
            jersey_text = str(int(row['Mapped Jersey Number'])) if pd.notna(row['Mapped Jersey Number']) else ''
            hover_text = f"<b>{row['playerName']}</b><br>Total Touches: {row['action_count']}"

            is_starter = row['Is Starter']
            
            marker_symbol = 'circle' if is_starter else 'diamond'
            # Converte hex in rgba per i sostituti per aggiungere opacità
            node_color = team_color if is_starter else f"rgba({int(team_color[1:3], 16)}, {int(team_color[3:5], 16)}, {int(team_color[5:7], 16)}, 0.6)"
            
            text_color = 'white' if team_color.lower() in ['red', 'blue', 'green', 'black', 'purple', 'tomato', 'skyblue'] else 'black'
            
            fig.add_trace(go.Scatter(
                x=[row['median_x']], y=[row['median_y']],
                mode='markers+text',
                marker=dict(
                    color=node_color, # <-- Colore del nodo dinamico
                    size=row['marker_size'],
                    symbol=marker_symbol,
                    line=dict(color='white' if is_starter else 'yellow', width=2)
                ),
                text=jersey_text,
                textfont=dict(color=text_color, size=12, weight='bold'),
                hovertext=hover_text, hoverinfo='text', name=row['playerName']
            ))

    # 3. Linea media del baricentro della squadra e etichetta
    # avg_line_x = df_player_agg[df_player_agg['Mapped Jersey Number'] != 1]['median_x'].mean()
    avg_line_x = df_player_agg['median_x'].mean()
    if pd.notna(avg_line_x):
        fig.add_shape(
            type='line', x0=avg_line_x, y0=-5, x1=avg_line_x, y1=105,
            line=dict(color='#6f8797', width=2, dash='dash')
        )
        
        pitch_length_meters = 105.0
        # Il calcolo dei metri usa sempre la coordinata originale, non quella invertita per il plot
        avg_line_meters = avg_line_x * (pitch_length_meters / 100.0)

        # The label follows the same left-to-right coordinate system.
        x_paper_coord = avg_line_x / 100.0

        fig.add_annotation(
            x=x_paper_coord, y=1.05,
            xref="paper", yref="paper",
            text=f"<b>Avg. Line: {avg_line_meters:.1f}m</b>",
            showarrow=False,
            font=dict(color="#16324a", size=12, family="Inter, Arial"),
            bgcolor="rgba(255, 255, 255, 0.94)",
            bordercolor="#cbdde6", borderwidth=1, borderpad=5
        )

    fig.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial", color="#27445b"),
        margin=dict(l=12, r=12, t=54, b=12),
        height=500,
        xaxis=dict(showgrid=False, zeroline=False, visible=False, fixedrange=True),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, fixedrange=True, scaleanchor="x", scaleratio=0.68)
    )

    add_attacking_direction(fig, dark=False)

    return fig

# PLOT-02 — synchronized interactive formation timeline
# ---------------------------------------------------------------------------

FORMATION_TIMELINE_HEIGHT = 510

_FORMATION_EVENT_CODES = {
    "starting_xi": "XI",
    "goal": "G",
    "substitution": "SUB",
    "formation_change": "FORM",
    "dismissal": "RC",
}


def _timeline_flag_is_true(value):
    """Return True only for explicit truthy Opta qualifier values."""
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {
        "1", "true", "yes", "y", "red", "rc",
    }


def _timeline_row_flag(row, aliases):
    return any(
        column in row.index
        and _timeline_flag_is_true(row.get(column))
        for column in aliases
    )


def _timeline_safe_int(value, default=None):
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _timeline_id(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            pass
    return text or None


def _timeline_event_seconds(row):
    minute = _timeline_safe_int(row.get("timeMin"), 0)
    second = _timeline_safe_int(row.get("timeSec"), 0)
    if minute is None:
        return None
    return int(max(minute, 0) * 60 + max(second or 0, 0))


def _timeline_time_label(seconds):
    minute, second = divmod(int(max(seconds or 0, 0)), 60)
    return (
        f"{minute}′ {second:02d}″"
        if second
        else f"{minute}′"
    )


def _timeline_formation_id(value, default=None):
    value = _timeline_safe_int(value, default)
    return default if value is None else int(value)


def _timeline_formation_name(formation_id):
    if formation_id is None:
        return "Unknown shape"
    try:
        name = get_formation_name(int(formation_id))
    except (TypeError, ValueError, KeyError):
        name = None
    return str(name) if name else f"Formation {formation_id}"


def _timeline_extract_player_positions(event):
    """Extract {player_id: Opta slot} from type 34 / 40 rows."""
    players_raw = event.get("Involved")
    positions_raw = event.get("Team player formation")
    if (
        players_raw is None
        or positions_raw is None
        or pd.isna(players_raw)
        or pd.isna(positions_raw)
    ):
        return {}

    player_ids = [
        _timeline_id(value)
        for value in str(players_raw).split(",")
    ]
    positions = [
        _timeline_safe_int(value)
        for value in str(positions_raw).split(",")
    ]
    if len(player_ids) != len(positions):
        return {}

    return {
        player_id: int(position)
        for player_id, position in zip(player_ids, positions)
        if player_id and position is not None and position > 0
    }


def _timeline_copy_state(state):
    return {
        "formation_id": _timeline_formation_id(
            state.get("formation_id")
        ),
        "players": {
            str(player_id): int(position)
            for player_id, position in (
                state.get("players", {}) or {}
            ).items()
            if player_id is not None and position is not None
        },
    }


def _timeline_player_data(df):
    player_map = {}
    if df is None or df.empty or "playerId" not in df.columns:
        return player_map

    for _, row in df.iterrows():
        player_id = _timeline_id(row.get("playerId"))
        if not player_id:
            continue

        current = player_map.setdefault(
            player_id,
            {"name": "Unknown", "jersey": "?"},
        )

        name = row.get("playerName")
        if name is not None and not pd.isna(name):
            name = str(name).strip()
            if name:
                current["name"] = name

        jersey = _timeline_safe_int(
            row.get("Mapped Jersey Number")
        )
        if jersey is not None:
            current["jersey"] = str(jersey)

    return player_map


def _timeline_player_label(player_map, player_id, fallback_name=None):
    player_id = _timeline_id(player_id)
    info = (player_map or {}).get(player_id, {})
    name = info.get("name")
    if not name or name == "Unknown":
        name = (
            str(fallback_name)
            if fallback_name is not None
            and not pd.isna(fallback_name)
            else "Unknown"
        )
    jersey = info.get("jersey", "?")
    return (
        f"#{jersey} {name}"
        if jersey and jersey != "?"
        else str(name)
    )


def _timeline_team_side(
    row,
    *,
    home_id,
    away_id,
    home_team,
    away_team,
):
    contestant_id = _timeline_id(row.get("contestantId"))
    if contestant_id and contestant_id == home_id:
        return "home"
    if contestant_id and contestant_id == away_id:
        return "away"

    team_name = row.get("team_name")
    if team_name == home_team:
        return "home"
    if team_name == away_team:
        return "away"
    return None


def _timeline_is_dismissal(row):
    return (
        _timeline_row_flag(row, ("Red card", "Red Card"))
        or _timeline_row_flag(
            row,
            ("Second yellow", "Second Yellow", "Second yellow card"),
        )
    )


def _timeline_dismissal_label(row):
    if _timeline_row_flag(
        row,
        ("Second yellow", "Second Yellow", "Second yellow card"),
    ):
        return "Second yellow"
    return "Red card"


def _timeline_is_own_goal(row):
    return _timeline_row_flag(
        row,
        ("Own goal", "Own Goal"),
    )


def _timeline_resolve_sub_on(df, sub_off):
    """
    Pair typeId 18 (Player Off) with typeId 19 (Player On).
    related_eventId is authoritative; same-time matching is fallback only.
    """
    event_id = _timeline_id(sub_off.get("eventId"))
    player_on = df[
        pd.to_numeric(df["typeId"], errors="coerce").eq(19)
    ].copy()

    if player_on.empty:
        return None

    if event_id and "related_eventId" in player_on.columns:
        linked = player_on.loc[
            player_on["related_eventId"].map(_timeline_id).eq(event_id)
        ]
        if not linked.empty:
            return linked.sort_values(
                "event_sequence_index",
                kind="stable",
            ).iloc[0]

    contestant_id = _timeline_id(sub_off.get("contestantId"))
    candidates = player_on.copy()
    if contestant_id and "contestantId" in candidates.columns:
        candidates = candidates[
            candidates["contestantId"].map(_timeline_id).eq(
                contestant_id
            )
        ]

    off_time = _timeline_event_seconds(sub_off)
    if off_time is not None and not candidates.empty:
        candidate_times = candidates.apply(
            _timeline_event_seconds,
            axis=1,
        )
        delta = candidate_times.sub(off_time).abs()
        nearby = candidates.loc[delta.le(2)]
        if not nearby.empty:
            return nearby.sort_values(
                "event_sequence_index",
                kind="stable",
            ).iloc[0]

    return None


def _timeline_event_code(events):
    counts = {}
    for event in events or []:
        code = _FORMATION_EVENT_CODES.get(event.get("kind"))
        if code:
            counts[code] = counts.get(code, 0) + 1

    labels = []
    for code in ("XI", "G", "SUB", "FORM", "RC"):
        count = counts.get(code, 0)
        if count:
            labels.append(f"{code}×{count}" if count > 1 else code)
    return "·".join(labels)


def _timeline_event_payload(
    kind,
    team,
    description,
    *,
    row=None,
    seconds=None,
):
    """Store readable exact event time inside a minute-level match moment."""
    exact_seconds = seconds

    if exact_seconds is None and row is not None:
        exact_seconds = _timeline_event_seconds(row)

    return {
        "kind": kind,
        "team": team,
        "description": description,
        "event_time_label": (
            _timeline_time_label(exact_seconds)
            if exact_seconds is not None
            else ""
        ),
    }


def _timeline_snapshot(
    *,
    seconds,
    score_home,
    score_away,
    home_state,
    away_state,
    home_highlights,
    away_highlights,
    events,
):
    home_copy = _timeline_copy_state(home_state)
    away_copy = _timeline_copy_state(away_state)

    return {
        "time_seconds": int(seconds),
        "time_label": _timeline_time_label(seconds),
        "score_home": int(score_home),
        "score_away": int(score_away),
        "score": f"{int(score_home)} – {int(score_away)}",
        "events": list(events or []),
        "home_state": home_copy,
        "away_state": away_copy,
        "home_highlights": sorted(
            str(value) for value in (home_highlights or set())
        ),
        "away_highlights": sorted(
            str(value) for value in (away_highlights or set())
        ),
        "home_formation_name": _timeline_formation_name(
            home_copy.get("formation_id")
        ),
        "away_formation_name": _timeline_formation_name(
            away_copy.get("formation_id")
        ),
    }


def build_formation_timeline_model(df_processed, match_info):
    """
    Build one compact synchronized timeline for Home and Away.

    Moments include Starting XI, goals, substitutions, formation changes and
    dismissals. Ordinary yellow cards are intentionally excluded.
    """
    if df_processed is None or df_processed.empty:
        return {
            "home_team": "Home",
            "away_team": "Away",
            "home_id": None,
            "away_id": None,
            "player_data": {},
            "moments": [],
            "match_end_seconds": 0,
        }

    df = df_processed.copy()
    if "event_sequence_index" not in df.columns:
        df = df.reset_index().rename(
            columns={"index": "event_sequence_index"}
        )

    sequence_values = pd.to_numeric(
        df["event_sequence_index"],
        errors="coerce",
    )
    fallback_sequence = pd.Series(
        range(len(df)),
        index=df.index,
        dtype=float,
    )
    df["event_sequence_index"] = sequence_values.fillna(
        fallback_sequence
    )

    type_ids = pd.to_numeric(df["typeId"], errors="coerce")

    start_events = df.loc[
        type_ids.eq(34)
    ].sort_values(
        "event_sequence_index",
        kind="stable",
    )

    if len(start_events) < 2:
        raise ValueError(
            "Could not find starting formation events for both teams."
        )

    match_info = match_info or {}
    home_team = match_info.get("hteamName") or "Home"
    away_team = match_info.get("ateamName") or "Away"

    team_series = (
        start_events["team_name"]
        if "team_name" in start_events.columns
        else pd.Series(index=start_events.index, dtype=object)
    )

    home_candidates = start_events[team_series.eq(home_team)]
    away_candidates = start_events[team_series.eq(away_team)]

    home_start = (
        home_candidates.iloc[0]
        if not home_candidates.empty
        else start_events.iloc[0]
    )

    if not away_candidates.empty:
        away_start = away_candidates.iloc[0]
    else:
        remaining = start_events.drop(
            index=home_start.name,
            errors="ignore",
        )
        away_start = (
            remaining.iloc[0]
            if not remaining.empty
            else start_events.iloc[1]
        )

    if home_team == "Home":
        value = home_start.get("team_name")
        if value is not None and not pd.isna(value):
            home_team = str(value)

    if away_team == "Away":
        value = away_start.get("team_name")
        if value is not None and not pd.isna(value):
            away_team = str(value)

    home_id = _timeline_id(home_start.get("contestantId"))
    away_id = _timeline_id(away_start.get("contestantId"))

    home_state = {
        "formation_id": _timeline_formation_id(
            home_start.get("Team formation")
        ),
        "players": _timeline_extract_player_positions(home_start),
    }
    away_state = {
        "formation_id": _timeline_formation_id(
            away_start.get("Team formation")
        ),
        "players": _timeline_extract_player_positions(away_start),
    }

    player_map = _timeline_player_data(df)
    score_home = 0
    score_away = 0

    moments = [
        _timeline_snapshot(
            seconds=0,
            score_home=0,
            score_away=0,
            home_state=home_state,
            away_state=away_state,
            home_highlights=set(),
            away_highlights=set(),
            events=[
                _timeline_event_payload(
                    "starting_xi",
                    "both",
                    "Starting XI and opening team structures.",
                    seconds=0,
                )
            ],
        )
    ]

    relevant = df.loc[
        type_ids.isin([16, 17, 18, 40])
    ].copy()

    if not relevant.empty:
        relevant["_timeline_seconds"] = relevant.apply(
            _timeline_event_seconds,
            axis=1,
        )
        relevant = relevant[
            relevant["_timeline_seconds"].notna()
        ].sort_values(
            ["_timeline_seconds", "event_sequence_index"],
            kind="stable",
        )

        # The analytical timeline works at football-minute resolution.
        # Multiple feed events a few seconds apart during the same stoppage
        # therefore become one navigable match moment, while each event keeps
        # its exact timestamp in the detail panel.
        relevant["_timeline_minute"] = (
            relevant["_timeline_seconds"]
            .floordiv(60)
            .astype(int)
            .mul(60)
        )

        grouped_relevant = relevant.groupby(
            "_timeline_minute",
            sort=True,
        )
    else:
        grouped_relevant = []

    for seconds, group in grouped_relevant:
        seconds = int(seconds)
        home_highlights = set()
        away_highlights = set()
        moment_events = []

        for _, event in group.iterrows():
            type_id = _timeline_safe_int(event.get("typeId"))

            side = _timeline_team_side(
                event,
                home_id=home_id,
                away_id=away_id,
                home_team=home_team,
                away_team=away_team,
            )

            team_name = (
                home_team
                if side == "home"
                else away_team
                if side == "away"
                else str(event.get("team_name", "Unknown team"))
            )

            state = (
                home_state
                if side == "home"
                else away_state
                if side == "away"
                else None
            )

            highlight_set = (
                home_highlights
                if side == "home"
                else away_highlights
                if side == "away"
                else set()
            )

            if type_id == 16:
                own_goal = _timeline_is_own_goal(event)
                scoring_side = side
                if own_goal:
                    scoring_side = (
                        "away"
                        if side == "home"
                        else "home"
                        if side == "away"
                        else None
                    )

                if scoring_side == "home":
                    score_home += 1
                elif scoring_side == "away":
                    score_away += 1

                scorer = _timeline_player_label(
                    player_map,
                    event.get("playerId"),
                    event.get("playerName"),
                )
                description = (
                    f"Own goal · {scorer} ({team_name})"
                    if own_goal
                    else f"Goal · {scorer} ({team_name})"
                )
                moment_events.append(
                    _timeline_event_payload(
                        "goal",
                        side,
                        description,
                        row=event,
                    )
                )

            elif type_id == 18:
                sub_on = _timeline_resolve_sub_on(df, event)
                player_off_id = _timeline_id(
                    event.get("playerId")
                )
                player_on_id = (
                    _timeline_id(sub_on.get("playerId"))
                    if sub_on is not None
                    else None
                )

                if state is not None:
                    position = (
                        state["players"].pop(player_off_id, None)
                        if player_off_id
                        else None
                    )
                    if player_on_id and position is not None:
                        state["players"][player_on_id] = int(position)
                        highlight_set.add(player_on_id)

                player_off = _timeline_player_label(
                    player_map,
                    player_off_id,
                    event.get("playerName"),
                )

                if sub_on is not None:
                    player_on = _timeline_player_label(
                        player_map,
                        player_on_id,
                        sub_on.get("playerName"),
                    )
                    description = (
                        f"{team_name} · {player_on} on for {player_off}"
                    )
                else:
                    description = (
                        f"{team_name} · {player_off} off "
                        "(incoming player link unavailable)"
                    )

                moment_events.append(
                    _timeline_event_payload(
                        "substitution",
                        side,
                        description,
                        row=event,
                    )
                )

            elif type_id == 40:
                if state is None:
                    continue

                previous_players = dict(state.get("players", {}))
                new_formation_id = _timeline_formation_id(
                    event.get("Team formation"),
                    state.get("formation_id"),
                )
                extracted_players = _timeline_extract_player_positions(
                    event
                )

                if new_formation_id is not None:
                    state["formation_id"] = new_formation_id

                if extracted_players:
                    state["players"] = extracted_players
                    for player_id, position in extracted_players.items():
                        if previous_players.get(player_id) != position:
                            highlight_set.add(player_id)

                formation_name = _timeline_formation_name(
                    state.get("formation_id")
                )

                moment_events.append(
                    _timeline_event_payload(
                        "formation_change",
                        side,
                        (
                            f"{team_name} · shape change to "
                            f"{formation_name}"
                        ),
                        row=event,
                    )
                )

            elif type_id == 17 and _timeline_is_dismissal(event):
                player_id = _timeline_id(event.get("playerId"))

                if state is not None and player_id:
                    state["players"].pop(player_id, None)

                player = _timeline_player_label(
                    player_map,
                    player_id,
                    event.get("playerName"),
                )
                card_label = _timeline_dismissal_label(event)
                moment_events.append(
                    _timeline_event_payload(
                        "dismissal",
                        side,
                        (
                            f"{card_label} · {player} ({team_name})"
                        ),
                        row=event,
                    )
                )

        # A group containing only normal yellow cards is ignored.
        if not moment_events:
            continue

        snapshot = _timeline_snapshot(
            seconds=seconds,
            score_home=score_home,
            score_away=score_away,
            home_state=home_state,
            away_state=away_state,
            home_highlights=home_highlights,
            away_highlights=away_highlights,
            events=moment_events,
        )

        if seconds == 0:
            snapshot["events"] = moments[0]["events"] + snapshot["events"]
            moments[0] = snapshot
        else:
            moments.append(snapshot)

    all_times = df.apply(_timeline_event_seconds, axis=1)
    valid_times = [
        int(value)
        for value in all_times.tolist()
        if value is not None and not pd.isna(value)
    ]
    last_moment = moments[-1]["time_seconds"] if moments else 0

    return {
        "home_team": str(home_team),
        "away_team": str(away_team),
        "home_id": home_id,
        "away_id": away_id,
        "player_data": player_map,
        "moments": moments,
        "match_end_seconds": int(
            max(valid_times + [last_moment, 1])
        ),
    }


def get_formation_timeline_moment(model, selected_time):
    moments = (model or {}).get("moments", []) or []
    if not moments:
        return None

    try:
        target = float(selected_time)
    except (TypeError, ValueError):
        target = float(moments[0]["time_seconds"])

    return min(
        moments,
        key=lambda moment: abs(
            float(moment.get("time_seconds", 0)) - target
        ),
    )


def build_formation_slider_marks(model):
    # Every analytical moment remains selectable. Rail text is deliberately
    # sparse: one semantic label per moment, with a fixed priority so mixed
    # moments never become strings such as "SUB·FORM".
    moments = (model or {}).get("moments", []) or []

    priority = {
        "dismissal": (4, "RC"),
        "goal": (3, "G"),
        "formation_change": (2, "FORM"),
        "starting_xi": (1, "XI"),
    }

    marks = {}
    labelled = []

    for moment in moments:
        seconds = int(moment.get("time_seconds", 0))
        events = moment.get("events", []) or []
        kinds = {
            event.get("kind")
            for event in events
        }

        best_kind = None
        best_priority = 0
        best_code = ""

        for kind in kinds:
            kind_priority, code = priority.get(
                kind,
                (0, ""),
            )

            if kind_priority > best_priority:
                best_kind = kind
                best_priority = kind_priority
                best_code = code

        marks[seconds] = {
            "label": "",
            "style": {
                "fontWeight": "700",
            },
        }

        if best_kind:
            labelled.append(
                {
                    "seconds": seconds,
                    "priority": best_priority,
                    "code": best_code,
                }
            )

    # Labels closer than two football minutes are hard to read on a compact
    # 90-minute rail. Keep every dot, but show only the strongest nearby label.
    # If priorities tie, keep the later moment because it reflects the most
    # recent tactical state.
    min_gap_seconds = 120
    visible = []

    for candidate in labelled:
        if (
            not visible
            or candidate["seconds"] - visible[-1]["seconds"]
            >= min_gap_seconds
        ):
            visible.append(candidate)
            continue

        previous = visible[-1]

        replace_previous = (
            candidate["priority"] > previous["priority"]
            or (
                candidate["priority"] == previous["priority"]
                and candidate["seconds"] > previous["seconds"]
            )
        )

        if replace_previous:
            visible[-1] = candidate

    for item in visible:
        seconds = item["seconds"]
        marks[seconds]["label"] = (
            f"{seconds // 60}′ {item['code']}"
        )

    return marks


def plot_formation_timeline_state(
    state,
    player_data_map,
    *,
    is_away=False,
    highlighted_players=None,
):
    """
    Render one team state using the shared Match Plot Design System.

    Home and Away deliberately share the canonical team-relative
    left-to-right orientation for direct shape comparison.
    """
    palette = get_team_palette(is_away=is_away)
    fig = go.Figure()

    pitch_shapes = pitch_plots.get_plotly_pitch_shapes(
        "rgba(255,255,255,0.24)",
        "rgba(255,255,255,0.82)",
    )

    state = state or {}
    formation_id = _timeline_formation_id(
        state.get("formation_id")
    )
    highlighted = {
        str(value)
        for value in (highlighted_players or [])
    }

    player_ids = []
    x_values = []
    y_values = []
    jerseys = []
    surnames = []
    customdata = []

    for player_id, position in (
        state.get("players", {}) or {}
    ).items():
        player_id = str(player_id)
        try:
            x, y = get_formation_layout_coords(
                formation_id,
                int(position),
            )
        except (TypeError, ValueError, KeyError):
            x, y = None, None

        if x is None or y is None:
            continue

        info = (player_data_map or {}).get(player_id, {})
        name = str(info.get("name", "Unknown"))
        jersey = str(info.get("jersey", "?"))
        surname = name.split()[-1] if name else "Unknown"

        player_ids.append(player_id)
        x_values.append(float(x))
        y_values.append(float(y))
        jerseys.append(jersey)
        surnames.append(surname)
        customdata.append([name, jersey, int(position)])

    if x_values:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers+text",
                marker=dict(
                    size=34,
                    color=palette["primary"],
                    line=dict(
                        color="rgba(255,255,255,0.92)",
                        width=1.8,
                    ),
                ),
                text=jerseys,
                textposition="middle center",
                textfont=dict(
                    color="#ffffff",
                    size=11,
                    family="Inter, Arial, sans-serif",
                ),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b>"
                    "<br>Jersey: %{customdata[1]}"
                    "<br>Formation slot: %{customdata[2]}"
                    "<extra></extra>"
                ),
                showlegend=False,
                name="Players",
            )
        )

        # High-contrast player labels are easier to scan than free-floating
        # white text, especially when two lines are vertically compact.
        for x, y, surname in zip(
            x_values,
            y_values,
            surnames,
        ):
            label = str(surname)
            if len(label) > 14:
                label = label[:13] + "…"

            place_above = y < 14

            fig.add_annotation(
                x=x,
                y=y,
                text=f"<b>{label}</b>",
                showarrow=False,
                xanchor="center",
                yanchor=(
                    "bottom"
                    if place_above
                    else "top"
                ),
                yshift=(
                    23
                    if place_above
                    else -23
                ),
                font=dict(
                    family="Inter, Arial, sans-serif",
                    color="#ffffff",
                    size=10.5,
                ),
                bgcolor="rgba(16,47,69,0.92)",
                bordercolor="rgba(255,255,255,0.20)",
                borderwidth=1,
                borderpad=2,
                opacity=0.98,
            )

        highlight_x = []
        highlight_y = []
        for player_id, x, y in zip(
            player_ids,
            x_values,
            y_values,
        ):
            if player_id in highlighted:
                highlight_x.append(x)
                highlight_y.append(y)

        if highlight_x:
            fig.add_trace(
                go.Scatter(
                    x=highlight_x,
                    y=highlight_y,
                    mode="markers",
                    marker=dict(
                        size=42,
                        color="rgba(0,0,0,0)",
                        line=dict(
                            color=MATCH_WARNING,
                            width=4,
                        ),
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                    name="Changed player",
                )
            )
    else:
        add_zero_state(
            fig,
            "Formation positions unavailable",
        )

    fig.add_annotation(
        x=3,
        y=97,
        text=f"<b>{_timeline_formation_name(formation_id)}</b>",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(
            family="Inter, Arial, sans-serif",
            color=MATCH_PITCH_TEXT,
            size=13,
        ),
        bgcolor="rgba(16,47,69,0.72)",
        borderpad=4,
    )

    apply_match_pitch_layout(
        fig,
        pitch_shapes=pitch_shapes,
        height=FORMATION_TIMELINE_HEIGHT,
        showlegend=False,
        header=False,
    )
    add_attacking_direction(fig, dark=True)

    return fig
