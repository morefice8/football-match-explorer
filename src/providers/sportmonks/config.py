"""Stable configuration and statistic identifiers for Sportmonks API v3."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class LeagueConfig:
    slug: str
    name: str
    league_id: int
    country: str


LEAGUES: dict[str, LeagueConfig] = {
    "premier-league": LeagueConfig("premier-league", "Premier League", 8, "England"),
    "la-liga": LeagueConfig("la-liga", "La Liga", 564, "Spain"),
    "bundesliga": LeagueConfig("bundesliga", "Bundesliga", 82, "Germany"),
    "serie-a": LeagueConfig("serie-a", "Serie A", 384, "Italy"),
    "ligue-1": LeagueConfig("ligue-1", "Ligue 1", 301, "France"),
}


POSITION_CODES = {
    24: "GK",
    25: "DF",
    26: "MF",
    27: "FW",
}


# Sportmonks lineup type IDs. formation_field is also checked because it is
# more defensive against future API changes.
STARTING_LINEUP_TYPE_ID = 11
BENCH_LINEUP_TYPE_ID = 12


STAT_COLUMNS: dict[int, str] = {
    34: "corners",
    40: "captain",
    41: "shots_off_target",
    42: "shots_total",
    43: "attacks",
    44: "dangerous_attacks",
    45: "possession_percentage",
    46: "ball_safe",
    49: "shots_inside_box",
    50: "shots_outside_box",
    51: "offsides",
    52: "goals",
    53: "goal_kicks",
    54: "goal_attempts",
    55: "free_kicks",
    56: "fouls",
    57: "saves",
    58: "shots_blocked",
    59: "substitutions",
    60: "throw_ins",
    62: "long_passes",
    64: "hit_woodwork",
    65: "successful_headers",
    78: "tackles",
    79: "assists",
    80: "passes",
    81: "successful_passes",
    82: "successful_passes_percentage",
    84: "yellow_cards",
    86: "shots_on_target",
    87: "injuries",
    88: "goals_conceded",
    94: "dispossessed",
    95: "offsides_provoked",
    96: "fouls_drawn",
    97: "blocked_shots",
    98: "total_crosses",
    99: "accurate_crosses",
    100: "interceptions",
    101: "clearances",
    104: "saves_inside_box",
    105: "total_duels",
    106: "duels_won",
    107: "aerials_won",
    108: "dribble_attempts",
    109: "successful_dribbles",
    110: "dribbled_past",
    116: "accurate_passes",
    117: "key_passes",
    118: "rating",
    119: "minutes_played",
    120: "touches",
    121: "turnovers",
    122: "long_balls",
    123: "long_balls_won",
    124: "through_balls",
    125: "through_balls_won",
    571: "error_lead_to_goal",
    580: "big_chances_created",
    581: "big_chances_missed",
    582: "clearance_off_line",
    583: "last_man_tackle",
    584: "good_high_claim",
    1490: "man_of_match",
    1491: "duels_lost",
    1533: "successful_crosses_percentage",
    1535: "goalkeeper_goals_conceded",
    1584: "accurate_passes_percentage",
    1605: "successful_dribbles_percentage",
    5304: "expected_goals",
    5305: "expected_goals_on_target",
    7939: "expected_points",
    7940: "expected_goals_penalties",
    7941: "expected_goals_free_kicks",
    7942: "expected_goals_corners",
    7943: "expected_non_penalty_goals",
    7944: "expected_goals_set_play",
    7945: "expected_goals_open_play",
    9684: "expected_goals_difference",
    9685: "shooting_performance",
    9686: "expected_goals_prevented",
    9687: "expected_goals_against",
    9706: "chances_created",
    27264: "successful_long_passes",
    27265: "successful_long_passes_percentage",
    27266: "aerials_lost",
    27267: "tackles_won",
    27268: "tackles_won_percentage",
    27269: "passes_final_third",
    27270: "long_balls_won_percentage",
    27271: "ball_recoveries",
    27272: "backward_passes",
    27273: "possession_lost",
    27274: "aerials",
    27275: "aerials_won_percentage",
    27276: "duels_won_percentage",
    48997: "error_lead_to_shot",
    117172: "cumulative_minutes_played",
}


ADDITIVE_PLAYER_COLUMNS = {
    "shots_off_target",
    "shots_total",
    "goals",
    "fouls",
    "saves",
    "shots_blocked",
    "hit_woodwork",
    "assists",
    "passes",
    "yellow_cards",
    "shots_on_target",
    "goals_conceded",
    "dispossessed",
    "offsides_provoked",
    "fouls_drawn",
    "blocked_shots",
    "total_crosses",
    "accurate_crosses",
    "interceptions",
    "clearances",
    "saves_inside_box",
    "total_duels",
    "duels_won",
    "aerials_won",
    "dribble_attempts",
    "successful_dribbles",
    "dribbled_past",
    "accurate_passes",
    "key_passes",
    "minutes_played",
    "touches",
    "turnovers",
    "long_balls",
    "long_balls_won",
    "through_balls",
    "through_balls_won",
    "error_lead_to_goal",
    "big_chances_created",
    "big_chances_missed",
    "clearance_off_line",
    "last_man_tackle",
    "good_high_claim",
    "man_of_match",
    "duels_lost",
    "goalkeeper_goals_conceded",
    "expected_goals",
    "expected_goals_on_target",
    "shooting_performance",
    "chances_created",
    "aerials_lost",
    "tackles_won",
    "passes_final_third",
    "ball_recoveries",
    "backward_passes",
    "possession_lost",
    "aerials",
    "error_lead_to_shot",
}


def slugify(value: Any) -> str:
    text = str(value or "").casefold().strip()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def statistic_column(type_id: Any, type_name: Any = None) -> str:
    try:
        numeric_id = int(type_id)
    except (TypeError, ValueError):
        numeric_id = 0
    if numeric_id in STAT_COLUMNS:
        return STAT_COLUMNS[numeric_id]
    suffix = slugify(type_name) or "unknown"
    return f"stat_{numeric_id}_{suffix}"


def season_name_matches(value: Any, start_year: int, end_year: int) -> bool:
    normalised = re.sub(r"[^0-9]+", " ", str(value or "")).strip()
    return normalised in {
        f"{start_year} {end_year}",
        f"{start_year} {str(end_year)[-2:]}",
        str(start_year),
    }
