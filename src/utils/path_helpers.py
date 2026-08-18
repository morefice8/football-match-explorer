# --- START OF FILE src/utils/path_helpers.py ---
import os
from src.utils.league_config import LEAGUES, LEAGUE_NAME_TO_FOLDER

# Definiamo le costanti qui, in modo che siano centralizzate
DATA_PATH = os.path.join("data", "fbref")
def get_team_logo_path(league_name, team_name):
    """Generates the correct URL path for a team's logo."""
    league_folder = LEAGUE_NAME_TO_FOLDER.get(league_name)
    if not league_folder: return ""
    teams_folder_name = f"{league_folder}_teams"
    team_folder_name = team_name.replace(' ', '_')
    return f"/data/fbref/{league_folder}/{teams_folder_name}/{team_folder_name}/logo.png"

def get_player_photo_path(player_name):
    player_folder = str(player_name).replace(' ', '_')
    base_player_folder = os.path.join(DATA_PATH, "player_top5_europe", player_folder)
    local_path = os.path.join(base_player_folder, f"{player_folder}.png")

    if os.path.exists(local_path):
        return f"/{local_path.replace(os.sep, '/')}"  # Percorso URL valido
    else:
        # Fallback immagine default (es. avatar generico)
        return "/assets/avatar_placeholder.png" 

def get_league_logo_path(league_name):
    """Generates the URL path for a league's logo."""
    league_folder = LEAGUE_NAME_TO_FOLDER.get(league_name)
    if not league_folder: return ""
    return f"/data/fbref/{league_folder}/{league_folder}.png"
# --- END OF FILE src/utils/path_helpers.py ---
