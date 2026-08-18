"""Shared UI configuration for the five supported domestic leagues."""

from __future__ import annotations

from src.providers.sportmonks.config import LEAGUES as SPORTMONKS_LEAGUES


def sportmonks_league_logo(league_id: int) -> str:
    """Build the public CDN URL used by Sportmonks ``image_path`` fields."""
    return f"https://cdn.sportmonks.com/images/soccer/leagues/{league_id % 32}/{league_id}.png"


LEAGUE_SHORT_NAMES = {
    "premier-league": "PL",
    "la-liga": "LL",
    "bundesliga": "BL",
    "serie-a": "SA",
    "ligue-1": "L1",
}


LEAGUES = {
    slug: {
        "name": league.name,
        "league_id": league.league_id,
        "logo": sportmonks_league_logo(league.league_id),
        "fallback": LEAGUE_SHORT_NAMES[slug],
    }
    for slug, league in SPORTMONKS_LEAGUES.items()
}

LEAGUE_NAME_TO_FOLDER = {value["name"]: slug for slug, value in LEAGUES.items()}
