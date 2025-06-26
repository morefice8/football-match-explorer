from datetime import datetime

# --- Constants ---
AUTHOR = '@Michele_OREFICE'
DEFAULT_HCOL = '#ff4b44' # Default Home Team Color
DEFAULT_ACOL = '#00a0de' # Default Away Team Color
VIOLET = '#a369ff'
GREEN = '#69f900'
BG_COLOR = '#ffffff'    # Default Background Color for plots
LINE_COLOR = '#000000' # Default Line Color for plots
UNSUCCESSFUL_COLOR = '#000000' # Default color for unsuccessful events
CARRY_COLOR = '#ffb366' # Default color for carries
SHOT_TYPES = ['Goal', 'Miss', 'Attempt Saved', 'Post'] # Use names if type_name is reliable

# In config.py or at the top of app.py
TEAM_NAME_TO_LOGO_CODE = {
    "Bournemouth": "BOU",
    "Arsenal": "ARS",
    "Aston Villa": "AVL", # Or whatever your code is
    "Brentford": "BRE",
    "Brighton": "BHA", # Or "BRI"
    "Burnley": "BUR",
    "Chelsea": "CHE",
    "Crystal Palace": "CRY", # Or "CPA"
    "Everton": "EVE",
    "Fulham": "FUL",
    "Ipswich": "IPS", # Or "IPW"
    "Leicester": "LEI",
    "Liverpool": "LIV",
    "Luton Town": "LUT",
    "Man City": "MCI", # Or "MAC"
    "Man Utd": "MUN", # Or "MAU"
    "Newcastle": "NEW",
    "Nottm Forest": "NFO", # Or "NOT"
    "Sheffield United": "SHE", # Or "SHU"
    "Southampton": "SOU", # Or "SFC"
    "Tottenham": "TOT",
    "West Ham": "WHU",
    "Wolves": "WOL"
    # Add all teams meticulously matching your data's team names and your desired codes
}

# Assuming all these are English league teams for the "ENG_" prefix
LOGO_PREFIX = "ENG_"
LOGO_EXTENSION = ".png"
DEFAULT_LOGO_PATH = "/assets/logos/_default_badge.png" # Make sure _default_badge.png exists

# --- File Paths (relative to project root) ---
# It's good practice to manage paths here
# Consider using os.path.join or pathlib for better path construction
OPTA_EVENTS_XLSX = "data/mappings/OptaEvents.xlsx"
OPTA_QUALIFIERS_JSON = "data/mappings/optaQualifierCodes.json"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
MAPPINGS_DIR = "data/mappings"
SAVE_PLOTS = True

# --- Default Analysis Parameters ---
# Define default Opta typeIds for different action categories
DEFAULT_PASS_TYPE_ID = 1
DEFAULT_SHOT_TYPES = ['Goal', 'Miss', 'Attempt Saved', 'Post'] # Use names if type_name is reliable
# Define a standard set of Defensive Actions for PPDA / Pressure plots
# DEFAULT_DEFENSIVE_ACTION_IDS = [4, 7, 45] # Default: Foul, Tackle, Challenge
# Optional: Define separate defaults if needed
DEFAULT_PPDA_DEF_ACTION_IDS = [4, 7, 8, 45, 74] # 4 = Foul, 7 = Tackle, 8 = Interception, 45 = Challenge, 74 = Blocked Pass
# DEFAULT_PRESSURE_MAP_IDS = [4, 7, 45]

# Default Thresholds for PPDA
DEFAULT_PPDA_PASS_ZONE_THRESH = 60.0 # Passing in opponent's first 60%
DEFAULT_PPDA_DEF_ZONE_THRESH = 40.0 # Pressing in opponent's final 40%

# Default Pitch Dimensions (if needed elsewhere)
DEFAULT_PITCH_LENGTH_METERS = 105.0
DEFAULT_PITCH_WIDTH_METERS = 68.0

# Default Settings for Buildup Analysis
LOSS_TYPES = ['Pass', 'Take On', 'Error', 'Aerial', 'Dispossessed']
RECOVERY_EVENT_TYPES = ["Interception", "Tackle", "Ball Recovery", "Error"]
KEYWORDS_PRIORITY_FOR_OFFENSIVE_TRANSITIONS = ["goal", "shot", "chance", "lost", "offside", "foul", "out"]
TRIGGER_TYPES_FOR_BUILDUPS = ['Out', 'Foul', 'Card', 'Miss', 'Offside provoked', 'Save', 'Claim', 'Keeper pick-up', 'Ball recovery', 'Corner Awarded', 'Attempt Saved'] # Triggers that start a buildup


# def extract_match_info(opta_data):
#     """Extracts basic match information from the loaded Opta JSON data."""
#     match_info_dict = opta_data.get('matchInfo', {})
#     contestants = match_info_dict.get('contestant', [])
#     competition = match_info_dict.get('competition', {})
#     date_str = match_info_dict.get('date', '')

#     info = {
#         'hteamName': contestants[0]['name'] if len(contestants) > 0 else None,
#         'hteamShortName': contestants[0]['code'] if len(contestants) > 0 else None,
#         'ateamName': contestants[1]['name'] if len(contestants) > 1 else None,
#         'ateamShortName': contestants[1]['code'] if len(contestants) > 1 else None,
#         'gw': match_info_dict.get('week'), # Gameweek
#         'league': competition.get('name'),
#         'date_iso': date_str, # Original ISO format date string
#         'date_formatted': None # Formatted date (e.g., 11 March 2024)
#     }

#     if date_str:
#         try:
#             # Remove 'Z' if present (indicates UTC) and parse
#             # Ensure the format string matches the input date format exactly
#             date_obj = datetime.strptime(date_str.replace('Z', ''), "%Y-%m-%d")
#             info['date_formatted'] = date_obj.strftime("%d %B %Y") # e.g., 11 March 2024
#         except ValueError:
#             print(f"Warning: Could not parse date string: {date_str}")
#             # You might want to log this error or handle it differently

#     return info

def extract_match_info(opta_data):
    """
    Extracts match information from Opta JSON data.
    """
    info = {}

    # --- Extract from matchInfo ---
    match_info_block = opta_data.get('matchInfo', {})
    contestants = match_info_block.get('contestant', [])
    competition_block = match_info_block.get('competition', {})
    date_str_matchinfo = match_info_block.get('date', '')

    # Home Team
    if len(contestants) > 0 and contestants[0]:
        info['hteamName'] = contestants[0].get('name', 'Home Full') # Full name
        info['hteamDisplayName'] = contestants[0].get('shortName', info['hteamName']) # shortName for display
        info['hteamCode'] = contestants[0].get('code', 'HOM') # code for logo
    else:
        info['hteamName'] = 'Home Full'
        info['hteamDisplayName'] = 'Home'
        info['hteamCode'] = 'HOM'

    # Away Team
    if len(contestants) > 1 and contestants[1]:
        info['ateamName'] = contestants[1].get('name', 'Away Full') # Full name
        info['ateamDisplayName'] = contestants[1].get('shortName', info['ateamName']) # shortName for display
        info['ateamCode'] = contestants[1].get('code', 'AWY') # code for logo
    else:
        info['ateamName'] = 'Away Full'
        info['ateamDisplayName'] = 'Away'
        info['ateamCode'] = 'AWY'
    
    info['gw'] = match_info_block.get('week')
    info['competitionName'] = competition_block.get('name', 'Unknown Competition')

    # --- Extract from liveData (for scores and potentially more accurate date) ---
    live_data_block = opta_data.get("liveData", {})
    match_details_block = live_data_block.get("matchDetails", {})
    
    scores_block = match_details_block.get("scores", {})
    total_scores = scores_block.get("total", {})
    info['home_score'] = total_scores.get("home")
    info['away_score'] = total_scores.get("away")

    info['matchStatus'] = match_details_block.get('matchStatus')
    info['winner'] = match_details_block.get('winner')

    # --- Date Handling ---
    info['date_formatted'] = "Unknown Date"
    date_str_livedata = None
    periods = match_details_block.get("period", [])
    if periods and isinstance(periods, list) and len(periods) > 0 and periods[0]: # Added check for periods[0]
        first_period_start_str = periods[0].get("start")
        if first_period_start_str:
            date_str_livedata = first_period_start_str

    final_date_str_to_parse = date_str_livedata if date_str_livedata else date_str_matchinfo
    info['date_iso'] = final_date_str_to_parse

    if final_date_str_to_parse:
        try:
            cleaned_date_str = final_date_str_to_parse.replace('Z', '')
            dt_object = None
            if '.' in cleaned_date_str:
                dt_object = datetime.strptime(cleaned_date_str, "%Y-%m-%dT%H:%M:%S.%f")
            else:
                dt_object = datetime.strptime(cleaned_date_str, "%Y-%m-%dT%H:%M:%S")
            info['date_formatted'] = dt_object.strftime("%d %B %Y")
        except ValueError:
            try:
                dt_object = datetime.strptime(final_date_str_to_parse.split("T")[0], "%Y-%m-%d")
                info['date_formatted'] = dt_object.strftime("%d %B %Y")
            except ValueError:
                print(f"Warning: Could not parse date string: {final_date_str_to_parse}")
                info['date_formatted'] = final_date_str_to_parse.split("T")[0] if final_date_str_to_parse else "Unknown Date"

    # --- Match ID ---
    opta_game_block = opta_data.get('game', {}) # F9 style game block
    # Use matchInfo.id if available (common for F24), else F9 game.id, else fallback
    info['matchId'] = str(match_info_block.get('id', opta_game_block.get('id', 'N/A')))


    # Fallbacks for essential names if they ended up None
    if not info.get('hteamName'): info['hteamName'] = 'Home Full'
    if not info.get('hteamDisplayName'): info['hteamDisplayName'] = 'Home'
    if not info.get('hteamCode'): info['hteamCode'] = 'HOM'
    
    if not info.get('ateamName'): info['ateamName'] = 'Away Full'
    if not info.get('ateamDisplayName'): info['ateamDisplayName'] = 'Away'
    if not info.get('ateamCode'): info['ateamCode'] = 'AWY'
    
    if not info.get('competitionName'): info['competitionName'] = 'Competition'

    return info