# In un nuovo file, es: src/visualization/formation_plotly.py
import plotly.graph_objects as go
import pandas as pd

from src.utils.formation_layouts import get_formation_layout_coords, get_formation_name, FORMATION_COORDINATES
from .buildup_plotly import draw_plotly_pitch

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
            print(f"[DEBUG] Formation ID {formation_id} not found in FORMATION_COORDINATES.")
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

    print(f"[DEBUG] Created {len(frames)} frames from timeline with {len(timeline)} snapshots.")
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
        print(f"[DEBUG] Processing snapshot: {snapshot}")
        formation_id = snapshot.get("formation_id")
        players = snapshot.get("players", {})
        print(f"[DEBUG] Formation ID: {formation_id}, Players: {players}")
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

    # Gestione assi (corretta con inversione per away)
    if is_away:
        fig.update_xaxes(range=[100, 0])
        fig.update_yaxes(range=[100, 0])
    else:
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
            line=dict(color='dimgrey', width=3, dash='dash')
        )
        
        pitch_length_meters = 105.0
        # Il calcolo dei metri usa sempre la coordinata originale, non quella invertita per il plot
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

