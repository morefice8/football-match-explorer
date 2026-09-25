# --- START OF FILE src/utils/path_helpers.py ---
import csv
import os
from functools import lru_cache
from pathlib import Path

from src.config import DEFAULT_LOGO_PATH
from src.utils.league_config import LEAGUES, LEAGUE_NAME_TO_FOLDER

# Definiamo le costanti qui, in modo che siano centralizzate
DATA_PATH = os.path.join("data", "fbref")
SPORTMONKS_TEAM_ALIASES = {
    "milan": "ac milan",
}


def _normalize_team_name(team_name):
    target_team = str(team_name or "").strip().casefold()
    return SPORTMONKS_TEAM_ALIASES.get(target_team, target_team)


def _find_local_fbref_logo(league_folder, team_name):
    teams_folder_name = f"{league_folder}_teams"
    team_folder_name = str(team_name).replace(' ', '_')
    local_path = Path(DATA_PATH) / league_folder / teams_folder_name / team_folder_name / "logo.png"
    if local_path.exists():
        return f"/{local_path.as_posix()}"
    return ""


def _find_local_fbref_logo_any_league(team_name):
    """Check every supported domestic league's local badge folder for this team.

    Used when the match's own competition isn't one of the five domestic
    leagues this app scrapes (e.g. a Champions League fixture) -- the team
    may still have a badge filed under its actual domestic league.
    """
    for league_folder in LEAGUE_NAME_TO_FOLDER.values():
        hit = _find_local_fbref_logo(league_folder, team_name)
        if hit:
            return hit
    return ""


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
    target_team = _normalize_team_name(team_name)

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


@lru_cache(maxsize=512)
def _get_sportmonks_team_logo_by_name(team_name):
    """Resolve a cached Sportmonks team image by team name alone, any league.

    Same data source as ``_get_sportmonks_team_logo``, but without the league
    filter -- a team keeps its Sportmonks badge regardless of which
    competition the current match belongs to, so a Champions League fixture
    can still resolve Inter's or Arsenal's badge via their domestic league
    entry in the same teams.csv.
    """
    processed_root = Path("data") / "sportmonks" / "processed"
    if not processed_root.exists():
        return ""

    season_folders = sorted(
        (path for path in processed_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    target_team = _normalize_team_name(team_name)

    for season_folder in season_folders:
        teams_csv = season_folder / "teams.csv"
        if not teams_csv.exists():
            continue
        try:
            with teams_csv.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    row_team = str(row.get("team_name", "")).strip().casefold()
                    if row_team == target_team:
                        return str(row.get("image_path") or "")
        except (OSError, csv.Error):
            continue
    return ""


def get_team_logo_path(league_name, team_name):
    """Return a team badge: local FBref badge, then Sportmonks, by competition first.

    ``league_name`` is the match's own competition (e.g. "Serie A" or "UEFA
    Champions League"). When it's one of the five domestic leagues this app
    has badge data for, that's tried first. Otherwise -- or if the team
    wasn't found there -- we search every supported league by team name
    alone, since a team's badge is filed under its domestic league
    regardless of which competition it's currently playing in. If nothing
    resolves anywhere, we fall back to the generic default badge rather than
    an empty/broken image.
    """
    league_folder = LEAGUE_NAME_TO_FOLDER.get(league_name)
    if league_folder:
        local_path = _find_local_fbref_logo(league_folder, team_name)
        if local_path:
            return local_path

        sportmonks_hit = _get_sportmonks_team_logo(league_name, team_name)
        if sportmonks_hit:
            return sportmonks_hit

    cross_league_hit = (
        _find_local_fbref_logo_any_league(team_name)
        or _get_sportmonks_team_logo_by_name(team_name)
    )
    if cross_league_hit:
        return cross_league_hit

    return DEFAULT_LOGO_PATH

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
