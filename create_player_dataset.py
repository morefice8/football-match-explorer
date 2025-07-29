# scripts/create_player_dataset.py
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import pycountry
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.player_helpers import LEAGUES, TEAM_NAME_EXCEPTIONS, COUNTRY_CODE_MAP, build_team_name_map, find_official_team_info, get_all_available_seasons, get_league_matchday_info

DATA_PATH = os.path.join("data", "fbref", "player_top5_europe")
OUTPUT_PATH = os.path.join("data", "processed")

# --- NUOVA FUNZIONE HELPER ---
def get_avg_minutes_from_logs(player_folder, season):
    """
    Calcola i minuti medi per partita giocata leggendo i matchlog di un giocatore.
    Restituisce anche il totale delle partite giocate.
    """
    matchlog_path = os.path.join(DATA_PATH, player_folder, season, "player_matchlogs.csv")
    if not os.path.exists(matchlog_path):
        return 0, 0

    try:
        df_logs = pd.read_csv(matchlog_path)
        # Considera solo le partite in cui il giocatore è sceso in campo (Min > 0)
        df_played = df_logs[pd.to_numeric(df_logs['Min'], errors='coerce').fillna(0) > 0]
        
        if df_played.empty:
            return 0, 0
            
        matches_played = len(df_played)
        minutes_played = int(df_played['Min'].sum())
        
        avg_mins = (minutes_played / matches_played) if matches_played > 0 else 0
        return avg_mins, matches_played
    except Exception:
        return 0, 0

def generate_season_dataframe(season):
    print(f"--- Inizio processamento per la stagione {season} ---")
    
    league_matchdays = get_league_matchday_info(season)
    all_players_data = []
    player_folders = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
    total_players = len(player_folders)
    print(f"Trovati {total_players} giocatori da analizzare.")

    for i, player_folder in enumerate(player_folders):
        if (i + 1) % 100 == 0:
            print(f"  ...processando giocatore {i+1}/{total_players}")
            
        player_name = player_folder.replace('_', ' ')
        season_path = os.path.join(DATA_PATH, player_folder, season)
        profile_file = os.path.join(DATA_PATH, player_folder, "profile.json")
        if not os.path.isdir(season_path) or not os.path.exists(profile_file): continue
        
        try:
            with open(profile_file, 'r', encoding='utf-8') as f: profile_data = json.load(f)
            
            # ... (logica di caricamento e merge dei CSV invariata) ...
            csv_files = { "std": "standard_stats.csv", "shoot": "shooting.csv", "pass": "passing.csv", "def": "defensive_actions.csv", "gca": "goal_and_shot_creation.csv", "gk": "goalkeeping.csv", "possession": "possession.csv", "misc": "miscellaneous_stats.csv", "adv_gk": "advanced_goalkeeping.csv"}
            player_dfs = {}
            for key, filename in csv_files.items():
                file_path = os.path.join(season_path, filename)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    df = df.loc[:, ~df.columns.duplicated()]
                    if key == 'possession': df.rename(columns={'1/3': 'Carries_Into_Final_Third'}, inplace=True)
                    if key == 'pass': df.rename(columns={'1/3': 'Passes_Into_Final_Third'}, inplace=True)
                    if key == 'adv_gk': df.rename(columns={'Att': 'Crosses_Faced'}, inplace=True)
                    if key == 'misc': df.rename(columns={'Won': 'Aerial_Won', 'Lost': 'Aerial_Lost'}, inplace=True)
                    if key == 'def': df.rename(columns={'Lost': 'Challenges_Lost'}, inplace=True)
                    player_dfs[key] = df
            
            if 'std' not in player_dfs: continue
            merge_keys = ['Season', 'Comp', 'Squad', 'Country', 'Age']
            for df_key in player_dfs:
                for key in merge_keys:
                    if key in player_dfs[df_key].columns:
                        if key == 'Age': player_dfs[df_key][key] = pd.to_numeric(player_dfs[df_key][key], errors='coerce')
                        else: player_dfs[df_key][key] = player_dfs[df_key][key].astype(str)
            df_merged = player_dfs['std']
            for key in player_dfs:
                if key != 'std':
                    df_to_merge = player_dfs[key]
                    common_cols = list(set(df_merged.columns) & set(df_to_merge.columns) - set(merge_keys))
                    df_merged = pd.merge(df_merged, df_to_merge.drop(columns=common_cols, errors='ignore'), on=merge_keys, how='left')
            df_season_stats = df_merged[df_merged['Season'] == str(season)].copy()
            if df_season_stats.empty: continue
            raw_numeric_cols = ['Min', '90s', 'Gls', 'Ast', 'xG', 'G-PK', 'SoT', 'SCA', 'PrgP', 'PrgC', 'Touches', 'Cmp', 'Att', 'Passes_Into_Final_Third', 'Carries_Into_Final_Third', 'Tkl+Int', 'TklW', 'Challenges_Lost', 'Clr', 'Blocks', 'PSxG', 'GA', 'Saves', 'SoTA', 'Stp', 'Crosses_Faced', '#OPA', 'Aerial_Won', 'Aerial_Lost']
            for col in raw_numeric_cols:
                if col in df_season_stats.columns:
                    df_season_stats[col] = pd.to_numeric(df_season_stats[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
                else: 
                    df_season_stats[col] = 0
            agg_dict = {col: 'sum' for col in raw_numeric_cols}
            season_summary = df_season_stats.agg({k: v for k, v in agg_dict.items() if k in df_season_stats.columns}).to_dict()

            # --- MODIFICA: Recupera e aggiungi Avg_Min_Per_Match e MP ---
            avg_mins, matches_played = get_avg_minutes_from_logs(player_folder, season)
            season_summary['Avg_Min_Per_Match'] = avg_mins
            season_summary['MP'] = matches_played # Sovrascrive MP con il valore corretto dai log

            season_summary['Player'] = player_name
            season_summary['Pos'] = profile_data.get('position', 'N/A')
            team_map = build_team_name_map()
            club_name_json = profile_data.get('club', {}).get('name', 'N/A')
            official_club_name, official_league_name = find_official_team_info(club_name_json, team_map)
            season_summary['Club'] = official_club_name
            season_summary['League'] = official_league_name
            
            try:
                birth_date = datetime.strptime(profile_data["birth_date"], "%Y-%m-%d")
                season_summary['Age'] = datetime.now().year - birth_date.year - ((datetime.now().month, datetime.now().day) < (birth_date.month, birth_date.day))
            except: 
                season_summary['Age'] = np.nan
            
            # ... (logica per nazionalità invariata) ...
            nationality_name = profile_data.get('national_team', {}).get('name')
            nationality_code = None
            national_team_link = profile_data.get('national_team', {}).get('link')
            if national_team_link:
                try:
                    three_letter_code = national_team_link.split('/')[-2].upper()
                    if three_letter_code in COUNTRY_CODE_MAP: nationality_code = COUNTRY_CODE_MAP[three_letter_code]
                    else:
                        country = pycountry.countries.get(alpha_3=three_letter_code)
                        if country: nationality_code = country.alpha_2.lower()
                except: pass
            if not nationality_name or not nationality_code:
                birth_place = profile_data.get('birth_place')
                if birth_place and ',' in birth_place:
                    try:
                        country_name_from_birth = birth_place.split(',')[-1].strip()
                        country_info = pycountry.countries.search_fuzzy(country_name_from_birth)[0]
                        if not nationality_name: nationality_name = country_info.name
                        if not nationality_code: nationality_code = country_info.alpha_2.lower()
                    except: pass
            season_summary['Nationality'] = nationality_name if nationality_name else "N/A"
            season_summary['Nationality_Code'] = nationality_code
            
            all_players_data.append(season_summary)
        except Exception as e:
            print(f"  -> Errore per {player_folder}: {e}")
            
    if not all_players_data: 
        print(f"Nessun dato trovato per la stagione {season}.")
        return

    df_full = pd.DataFrame(all_players_data)

    df_full['League_MP'] = df_full['League'].map(league_matchdays).fillna(0)
    
    # ... (calcolo metriche per 90 invariato) ...
    with np.errstate(divide='ignore', invalid='ignore'):
        df_full['G_minus_xG_per_90'] = ((df_full['Gls'] - df_full['xG']) / df_full['90s']).replace(np.inf, 0).fillna(0)
        df_full['Passes_F3_per_90'] = (df_full['Passes_Into_Final_Third'] / df_full['90s']).replace(np.inf, 0).fillna(0)
        df_full['Carries_F3_per_90'] = (df_full['Carries_Into_Final_Third'] / df_full['90s']).replace(np.inf, 0).fillna(0)
        df_full['Tkl+Int_per_90'] = (df_full['Tkl+Int'] / df_full['90s']).replace(np.inf, 0).fillna(0)
        df_full['Aerial_Duels_perc'] = (df_full['Aerial_Won'] / (df_full['Aerial_Won'] + df_full['Aerial_Lost']) * 100).replace(np.inf, 0).fillna(0)
        df_full['Def_Duel_Win_perc'] = (df_full['TklW'] / (df_full['TklW'] + df_full['Challenges_Lost']) * 100).replace(np.inf, 0).fillna(0)
        df_full['Clr_per_90'] = (df_full['Clr'] / df_full['90s']).replace(np.inf, 0).fillna(0)
        df_full['Blocks_per_90'] = (df_full['Blocks'] / df_full['90s']).replace(np.inf, 0).fillna(0)
        df_full['PrgP_per_90'] = (df_full['PrgP'] / df_full['90s']).replace(np.inf, 0).fillna(0)
        df_full['SCA90'] = (df_full['SCA'] / df_full['90s']).replace(np.inf, 0).fillna(0)
        df_full['SoT/90'] = (df_full['SoT'] / df_full['90s']).replace(np.inf, 0).fillna(0)
        df_full['PSxG+/-'] = (df_full['PSxG'] - df_full['GA']).fillna(0)
        df_full['Save%'] = (df_full['Saves'] / df_full['SoTA'] * 100).replace(np.inf, 0).fillna(0)
        df_full['Stp%'] = (df_full['Stp'] / df_full['Crosses_Faced'] * 100).replace(np.inf, 0).fillna(0)
        df_full['#OPA/90'] = (df_full['#OPA'] / df_full['90s']).replace(np.inf, 0).fillna(0)

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    parquet_file = os.path.join(OUTPUT_PATH, f"player_stats_{season}.parquet")
    csv_file = os.path.join(OUTPUT_PATH, f"player_stats_{season}.csv")
    
    df_full.to_parquet(parquet_file, index=False)
    df_full.to_csv(csv_file, index=False)
    
    print(f"--- Salvataggio completato! Dati salvati in: {parquet_file} e {csv_file} ---")

if __name__ == "__main__":
    available_seasons = get_all_available_seasons(DATA_PATH)
    if available_seasons:
        for season in available_seasons:
            generate_season_dataframe(season)
    else:
        print("Nessuna stagione trovata nei dati grezzi.")