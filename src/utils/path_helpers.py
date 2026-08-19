# --- START OF FILE src/utils/path_helpers.py ---
import csv
import os
from functools import lru_cache
from pathlib import Path

from src.utils.league_config import LEAGUES, LEAGUE_NAME_TO_FOLDER

# Definiamo le costanti qui, in modo che siano centralizzate
DATA_PATH = os.path.join("data", "fbref")


@lru_cache(maxsize=512)
def _get_sportmonks_team_logo(league_name, team_name):
    """Resolve a cached Sportmonks team image without requiring parquet support."""
    processed_root = Path("data") / "sportmonks" / "processed"
    if not processed_root.exists():
        return ""

    season_folders = sorted(
        (path for path in processed_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    target_league = str(league_name or "").strip().casefold()
    target_team = str(team_name or "").strip().casefold()

    for season_folder in season_folders:
        teams_csv = season_folder / "teams.csv"
        if not teams_csv.exists():
            continue
        try:
            with teams_csv.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    row_league = str(row.get("league", "")).strip().casefold()
                    row_team = str(row.get("team_name", "")).strip().casefold()
                    if row_league == target_league and row_team == target_team:
                        return str(row.get("image_path") or "")
        except (OSError, csv.Error):
            continue
    return ""


def get_team_logo_path(league_name, team_name):
    """Return a local FBref badge when available, then a Sportmonks fallback."""
    league_folder = LEAGUE_NAME_TO_FOLDER.get(league_name)
    if league_folder:
        teams_folder_name = f"{league_folder}_teams"
        team_folder_name = str(team_name).replace(' ', '_')
        local_path = Path(DATA_PATH) / league_folder / teams_folder_name / team_folder_name / "logo.png"
        if local_path.exists():
            return f"/{local_path.as_posix()}"

    return _get_sportmonks_team_logo(league_name, team_name)

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
