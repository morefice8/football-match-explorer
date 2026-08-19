# src/utils/player_helpers.py
import os
import pandas as pd
import pycountry
from src.metrics.sportmonks import SPORTMONKS_TOOLTIPS
from src.utils.league_config import LEAGUES

# --- COSTANTI CONDIVISE ---
TEAM_NAME_EXCEPTIONS = {
    "manchester_united": "manchester_utd", "newcastle_united": "newcastle_utd",
    "manchester_city": "manchester_city", "Manchester City": "manchester_city",
    "tottenham_hotspur": "tottenham",
    "wolverhampton_wanderers": "wolves", "nottingham_forest": "nott'ham_forest",
    "paris_saint-germain": "paris_s-g", "eintracht_frankfurt": "eint_frankfurt",
    "brighton_&_hove_albion": "brighton", "internazionale": "inter", "paris_sg": "paris_s-g"
}
COUNTRY_CODE_MAP = {
    'ENG': 'gb-eng', 'SCO': 'gb-sct', 'WAL': 'gb-wls', 'NIR': 'gb-nir', 'GUI': 'gn', 
    'KVX': 'xk', 'CGO': 'cg', 'DRC': 'cd', 'DEU': 'de', 'GER': 'de', 'ALG': 'dz', 'DZA': 'dz'
}

NATIONALITY_FLAG_OVERRIDES = {
    'england': 'gb-eng',
    'scotland': 'gb-sct',
    'wales': 'gb-wls',
    'northern ireland': 'gb-nir',
}

METRIC_TOOLTIPS = {
    "Gls": "Total Goals Scored: The total number of goals scored by the player in all seasonal competitions. The primary metric for evaluating a finisher.",
    "Ast": "Total Assists: The total number of assists provided. Measures a player's ability to create direct scoring opportunities for teammates.",
    "G_minus_xG_per_90": "Goals minus Expected Goals per 90: Measures a player's finishing ability. A high positive value indicates that the player scores more goals than the quality of his chances would suggest, demonstrating composure and shooting ability.",
    "SCA90": "Shot-Creating Actions per 90: The two attacking actions (passes, dribbling, fouls suffered) that directly lead to a shot. Indicates a player's creative involvement.",
    "SoT/90": "Shots on Target per 90: Total shots on target per 90 minutes. Indicates The frequency and accuracy with which a player engages the opposing goalkeeper.",
    "Passes_F3_per_90": "Passes into Final Third per 90: Number of completed passes that enter the attacking third. Measures the player's ability to advance the play.",
    "PrgP_per_90": "Progressive Passes per 90: Passes that significantly advance the ball towards the opponent's goal. An index of verticality and vision.",
    "Carries_F3_per_90": "Carries into Final Third per 90: Number of times the player carries the ball into the attacking third. Measures the player's ability to progress with the ball.",
    "Tkl+Int_per_90": "Tackles + Interceptions per 90: Sum of the main defensive actions. A general indicator of a player's defensive workload.",
    "Aerial_Duels_perc": "Aerial Duels Won %: Percentage of aerial duels won. A key metric for central defenders and physical attackers.",
    "Clr_per_90": "Clearances per 90: Defensive clearances per 90 minutes. Indicates a player's ability to clear an immediate threat from their own penalty area.",
    "Blocks_per_90": "Blocks per 90: Sum of shots and passes blocked per 90 minutes. Measures the opposition's ability to block and read their passing/shooting lines.",
    "Save%": "Save Percentage: Percentage of shots on target saved. The key metric for evaluating a goalkeeper's reactivity.",
    "PSxG+/-": "Post-Shot xG minus Goals Against: Difference between expected goals 'post-shot' and goals conceded. A positive value indicates that the goalkeeper saved 'impossible' goals, saving better than average.",
    "Stp%": "Crosses Stopped %: Percentage of opponents' crosses into the penalty area that are intercepted by the goalkeeper. Measures the goalkeeper's dominance of the penalty area on high balls.",
    "#OPA/90": "Sweeper Actions per 90: The goalkeeper's defensive actions outside his own penalty area per 90 minutes. Indicates his proactivity and ability to play as a sweeper."
}

# Sportmonks-native alternatives used when FBref-only definitions are absent.
METRIC_TOOLTIPS.update({
    "Chances_Created_per_90": "Chances created per 90 minutes, as provided by Sportmonks.",
    "Key_Passes_per_90": "Passes that directly lead to a shot, per 90 minutes.",
    "Successful_Dribbles_per_90": "Successful dribbles per 90 minutes.",
    "Saves_per_90": "Goalkeeper saves per 90 minutes.",
    "GA_per_90": "Goals conceded by the goalkeeper per 90 minutes.",
})
METRIC_TOOLTIPS.update(SPORTMONKS_TOOLTIPS)

# --- FUNZIONI DI SUPPORTO CONDIVISE ---
def get_nationality_flag_code(country_code, nationality=None):
    """Return the flag-icon-css code for a Sportmonks nationality."""
    nationality_key = str(nationality or '').strip().casefold()
    if nationality_key in NATIONALITY_FLAG_OVERRIDES:
        return NATIONALITY_FLAG_OVERRIDES[nationality_key]

    raw_code = str(country_code or '').strip()
    if not raw_code or raw_code.casefold() == 'nan':
        return None

    return COUNTRY_CODE_MAP.get(raw_code.upper(), raw_code.lower())


def build_team_name_map():
    team_map = {}
    for league_folder, league_info in LEAGUES.items():
        teams_dir_path = os.path.join("data", "fbref", league_folder, f"{league_folder}_teams")
        if os.path.isdir(teams_dir_path):
            for team_folder in os.listdir(teams_dir_path):
                team_map[team_folder.lower()] = {'folder_name': team_folder, 'league_name': league_info['name']}
    return team_map

def find_official_team_info(club_name_from_json, team_map):
    if not club_name_from_json or not isinstance(club_name_from_json, str): return 'N/A', 'N/A'
    simple_name = club_name_from_json.lower().replace(' ', '_')
    if simple_name in TEAM_NAME_EXCEPTIONS:
        lookup_key = TEAM_NAME_EXCEPTIONS[simple_name]
        if lookup_key in team_map:
            info = team_map[lookup_key]
            return info['folder_name'].replace('_', ' '), info['league_name']
    for key, info in team_map.items():
        if simple_name in key:
            return info['folder_name'].replace('_', ' '), info['league_name']
    return club_name_from_json, "Unknown League"

def get_all_available_seasons(data_path):
    all_seasons = set()
    if not os.path.isdir(data_path): return []
    player_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    for player_folder in player_folders:
        try:
            seasons = [s for s in os.listdir(os.path.join(data_path, player_folder)) if os.path.isdir(os.path.join(data_path, player_folder, s)) and '-' in s]
            all_seasons.update(seasons)
        except FileNotFoundError: continue
    return sorted(list(all_seasons), reverse=True)

def get_league_matchday_info(season):
    """
    Legge i file standard delle squadre per determinare il numero massimo di partite (giornate) giocate
    in ogni lega per una data stagione.
    """
    league_matchdays = {}
    print("-> Recupero informazioni sulle giornate giocate per ogni lega...")
    for league_folder, league_info in LEAGUES.items():
        try:
            stats_path = os.path.join("data", "fbref", league_folder, f"{league_folder}_stats", season, "standard.csv")
            if os.path.exists(stats_path):
                df_league = pd.read_csv(stats_path)
                # La colonna 'MP' (Matches Played) ci dice le giornate
                if 'MP' in df_league.columns:
                    max_mp = pd.to_numeric(df_league['MP'], errors='coerce').max()
                    league_matchdays[league_info['name']] = max_mp
        except Exception as e:
            print(f"  -> Attenzione: impossibile leggere i dati delle giornate per {league_info['name']}. Errore: {e}")
    
    print(f"-> Giornate giocate: {league_matchdays}")
    return league_matchdays
