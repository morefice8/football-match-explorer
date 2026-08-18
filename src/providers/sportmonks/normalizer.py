"""Convert Sportmonks API responses into stable application datasets."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import POSITION_CODES, STARTING_LINEUP_TYPE_ID, statistic_column


def response_data(payload: Any) -> Any:
    """Return the useful part of a Sportmonks response or an empty value."""
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value if math.isfinite(float(value)) else None
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("%", "").strip()
        try:
            number = float(cleaned)
            return int(number) if number.is_integer() else number
        except ValueError:
            return None
    return None


def metric_value(detail: dict[str, Any]) -> float | int | None:
    """Extract the most useful scalar from fixture or season statistic data."""
    value: Any = detail.get("data", detail.get("value"))
    if isinstance(value, dict):
        # Fixture values normally use ``value``. Season totals use ``total``
        # or ``count``. Average is deliberately the last choice.
        for key in ("value", "total", "count", "goals", "average"):
            if key in value:
                scalar = _number(value[key])
                if scalar is not None:
                    return scalar
        return None
    return _number(value)


def _type_info(detail: dict[str, Any]) -> tuple[int | None, str | None, str | None, str | None]:
    type_data = detail.get("type") if isinstance(detail.get("type"), dict) else {}
    raw_id = detail.get("type_id", type_data.get("id"))
    try:
        type_id = int(raw_id)
    except (TypeError, ValueError):
        type_id = None
    return (
        type_id,
        type_data.get("name"),
        type_data.get("code"),
        type_data.get("stat_group"),
    )


def _metrics(
    details: Iterable[dict[str, Any]],
    *,
    catalog: list[dict[str, Any]],
    level: str,
) -> dict[str, float | int]:
    row: dict[str, float | int] = {}
    for detail in details:
        if not isinstance(detail, dict):
            continue
        type_id, name, code, stat_group = _type_info(detail)
        column = statistic_column(type_id, name)
        value = metric_value(detail)
        if value is not None:
            row[column] = value
        catalog.append(
            {
                "level": level,
                "type_id": type_id,
                "column": column,
                "name": name,
                "code": code,
                "stat_group": stat_group,
            }
        )
    return row


def _date_age(date_of_birth: Any, season_end_year: int) -> float:
    if not date_of_birth:
        return np.nan
    try:
        born = date.fromisoformat(str(date_of_birth)[:10])
        reference = date(season_end_year, 6, 30)
        return reference.year - born.year - ((reference.month, reference.day) < (born.month, born.day))
    except (TypeError, ValueError):
        return np.nan


def _player_record(player: dict[str, Any], season_end_year: int) -> dict[str, Any]:
    nationality = player.get("nationality") if isinstance(player.get("nationality"), dict) else {}
    position = player.get("position") if isinstance(player.get("position"), dict) else {}
    position_id = player.get("position_id") or position.get("id")
    return {
        "player_id": player.get("id"),
        "player_name": player.get("display_name") or player.get("name") or player.get("common_name"),
        "common_name": player.get("common_name"),
        "first_name": player.get("firstname"),
        "last_name": player.get("lastname"),
        "date_of_birth": player.get("date_of_birth"),
        "age": _date_age(player.get("date_of_birth"), season_end_year),
        "position_id": position_id,
        "position": POSITION_CODES.get(position_id, position.get("name")),
        "detailed_position_id": player.get("detailed_position_id"),
        "nationality_id": player.get("nationality_id") or nationality.get("id"),
        "nationality": nationality.get("name"),
        "nationality_code": (nationality.get("iso2") or nationality.get("code") or "").lower() or None,
        "height": player.get("height"),
        "weight": player.get("weight"),
        "image_path": player.get("image_path"),
    }


class SeasonNormalizer:
    """Incrementally normalise one or more leagues from cached API payloads."""

    def __init__(self, season_name: str):
        self.season_name = season_name
        try:
            self.season_end_year = int(season_name.split("-")[1])
        except (IndexError, ValueError):
            self.season_end_year = datetime.utcnow().year

        self.leagues: list[dict[str, Any]] = []
        self.teams: list[dict[str, Any]] = []
        self.players: list[dict[str, Any]] = []
        self.squads: list[dict[str, Any]] = []
        self.fixtures: list[dict[str, Any]] = []
        self.standings: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.team_matchlogs: list[dict[str, Any]] = []
        self.player_matchlogs: list[dict[str, Any]] = []
        self.team_season_api: list[dict[str, Any]] = []
        self.player_season_api: list[dict[str, Any]] = []
        self.metric_catalog: list[dict[str, Any]] = []

    def add_league(self, league: dict[str, Any]) -> None:
        self.leagues.append({**league, "season": self.season_name})

    def add_teams(self, payload: Any, league: dict[str, Any]) -> None:
        for team in _as_list(response_data(payload)):
            if not isinstance(team, dict):
                continue
            self.teams.append(
                {
                    "league_id": league["league_id"],
                    "league": league["league"],
                    "season_id": league["season_id"],
                    "season": self.season_name,
                    "team_id": team.get("id"),
                    "team_name": team.get("name"),
                    "short_code": team.get("short_code"),
                    "country_id": team.get("country_id"),
                    "venue_id": team.get("venue_id"),
                    "founded": team.get("founded"),
                    "image_path": team.get("image_path"),
                }
            )

    def add_squad(self, payload: Any, league: dict[str, Any], team: dict[str, Any]) -> None:
        for item in _as_list(response_data(payload)):
            if not isinstance(item, dict):
                continue
            player = item.get("player") if isinstance(item.get("player"), dict) else {}
            player_row = _player_record(player, self.season_end_year)
            if not player_row.get("player_id"):
                player_row["player_id"] = item.get("player_id")
                player_row["player_name"] = item.get("player_name")
            self.players.append(player_row)
            details = _metrics(
                _as_list(item.get("details")), catalog=self.metric_catalog, level="player_season_squad"
            )
            squad_row = {
                "league_id": league["league_id"],
                "league": league["league"],
                "season_id": league["season_id"],
                "season": self.season_name,
                "team_id": item.get("team_id") or team.get("team_id"),
                "team_name": team.get("team_name"),
                "player_id": item.get("player_id") or player_row.get("player_id"),
                "player_name": player_row.get("player_name"),
                "position_id": item.get("position_id") or player_row.get("position_id"),
                "position": POSITION_CODES.get(item.get("position_id")) or player_row.get("position"),
                "jersey_number": item.get("jersey_number"),
                **details,
            }
            self.squads.append(squad_row)

    def add_team_season_stats(self, payload: Any, league: dict[str, Any]) -> None:
        team = response_data(payload)
        if not isinstance(team, dict):
            return
        for statistic in _as_list(team.get("statistics")):
            if not isinstance(statistic, dict):
                continue
            row = {
                "league_id": league["league_id"],
                "league": league["league"],
                "season_id": league["season_id"],
                "season": self.season_name,
                "team_id": statistic.get("team_id") or team.get("id"),
                "team_name": team.get("name"),
            }
            row.update(
                _metrics(
                    _as_list(statistic.get("details")),
                    catalog=self.metric_catalog,
                    level="team_season_api",
                )
            )
            self.team_season_api.append(row)

    def add_player_season_stats(self, payload: Any, league: dict[str, Any]) -> None:
        player = response_data(payload)
        if not isinstance(player, dict):
            return
        player_row = _player_record(player, self.season_end_year)
        self.players.append(player_row)
        for statistic in _as_list(player.get("statistics")):
            if not isinstance(statistic, dict):
                continue
            row = {
                "league_id": league["league_id"],
                "league": league["league"],
                "season_id": league["season_id"],
                "season": self.season_name,
                "team_id": statistic.get("team_id"),
                "player_id": statistic.get("player_id") or player.get("id"),
                "player_name": player_row.get("player_name"),
                "position_id": statistic.get("position_id") or player.get("position_id"),
                "position": POSITION_CODES.get(statistic.get("position_id") or player.get("position_id")),
                "jersey_number": statistic.get("jersey_number"),
            }
            row.update(
                _metrics(
                    _as_list(statistic.get("details")),
                    catalog=self.metric_catalog,
                    level="player_season_api",
                )
            )
            self.player_season_api.append(row)

    def add_standings(self, payload: Any, league: dict[str, Any]) -> None:
        def walk(value: Any) -> Iterable[dict[str, Any]]:
            if isinstance(value, list):
                for child in value:
                    yield from walk(child)
            elif isinstance(value, dict):
                if value.get("participant_id") is not None and value.get("position") is not None:
                    yield value
                else:
                    for child in value.values():
                        if isinstance(child, (dict, list)):
                            yield from walk(child)

        seen: set[tuple[Any, Any]] = set()
        for item in walk(response_data(payload)):
            key = (item.get("participant_id"), item.get("position"))
            if key in seen:
                continue
            seen.add(key)
            participant = item.get("participant") if isinstance(item.get("participant"), dict) else {}
            row = {
                "league_id": league["league_id"],
                "league": league["league"],
                "season_id": league["season_id"],
                "season": self.season_name,
                "team_id": item.get("participant_id"),
                "team_name": participant.get("name"),
                "position": item.get("position"),
                "points": item.get("points"),
                "result": item.get("result"),
            }
            row.update(
                _metrics(_as_list(item.get("details")), catalog=self.metric_catalog, level="standing")
            )
            self.standings.append(row)

    @staticmethod
    def _score_map(scores: list[dict[str, Any]]) -> dict[Any, Any]:
        current = [score for score in scores if str(score.get("description", "")).upper() == "CURRENT"]
        chosen = current or scores
        result: dict[Any, Any] = {}
        for score in chosen:
            participant_id = score.get("participant_id")
            score_data = score.get("score") if isinstance(score.get("score"), dict) else {}
            goals = _number(score_data.get("goals"))
            if participant_id is not None and goals is not None:
                result[participant_id] = goals
        return result

    def add_fixture(self, payload: Any, league: dict[str, Any]) -> None:
        fixture = response_data(payload)
        if not isinstance(fixture, dict):
            return

        participants = [p for p in _as_list(fixture.get("participants")) if isinstance(p, dict)]
        participant_by_id = {p.get("id"): p for p in participants}
        location_by_id = {
            p.get("id"): (p.get("meta") or {}).get("location")
            for p in participants
            if isinstance(p.get("meta"), dict)
        }
        home = next((p for p in participants if location_by_id.get(p.get("id")) == "home"), {})
        away = next((p for p in participants if location_by_id.get(p.get("id")) == "away"), {})
        state = fixture.get("state") if isinstance(fixture.get("state"), dict) else {}
        score_map = self._score_map([s for s in _as_list(fixture.get("scores")) if isinstance(s, dict)])
        fixture_id = fixture.get("id")
        has_team_detail = bool(_as_list(fixture.get("statistics")) or _as_list(fixture.get("xgfixture")))

        base = {
            "league_id": league["league_id"],
            "league": league["league"],
            "season_id": league["season_id"],
            "season": self.season_name,
            "fixture_id": fixture_id,
            "starting_at": fixture.get("starting_at"),
            "round_id": fixture.get("round_id"),
            "stage_id": fixture.get("stage_id"),
            "venue_id": fixture.get("venue_id"),
            "state_id": fixture.get("state_id"),
            "state": state.get("short_name") or state.get("state") or state.get("developer_name"),
            "result_info": fixture.get("result_info"),
            "home_team_id": home.get("id"),
            "home_team": home.get("name"),
            "away_team_id": away.get("id"),
            "away_team": away.get("name"),
            "home_score": score_map.get(home.get("id")),
            "away_score": score_map.get(away.get("id")),
        }
        self.fixtures.append(base)

        by_team: dict[Any, dict[str, Any]] = defaultdict(dict)
        for detail in _as_list(fixture.get("statistics")) + _as_list(fixture.get("xgfixture")):
            if not isinstance(detail, dict):
                continue
            team_id = detail.get("participant_id")
            by_team[team_id].update(
                _metrics([detail], catalog=self.metric_catalog, level="team_fixture")
            )

        # Some team-level fields (notably passes into the final third and
        # tackles won) are available only inside lineup details. Roll them up
        # once and use them only when Sportmonks did not return a team value.
        lineup_rollup_columns = {
            "assists", "accurate_passes", "key_passes", "passes_final_third",
            "touches", "tackles_won", "aerials_won", "aerials_lost",
            "error_lead_to_shot", "clearances", "blocked_shots",
            "chances_created", "dribble_attempts", "successful_dribbles",
        }
        lineup_details: dict[Any, dict[str, Any]] = {}
        lineup_rollups: dict[Any, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for lineup in _as_list(fixture.get("lineups")):
            if not isinstance(lineup, dict):
                continue
            key = lineup.get("id") or (lineup.get("team_id"), lineup.get("player_id"))
            details = _metrics(
                _as_list(lineup.get("details")), catalog=self.metric_catalog, level="player_fixture"
            )
            lineup_details[key] = details
            team_id = lineup.get("team_id")
            for column in lineup_rollup_columns:
                value = _number(details.get(column))
                if value is not None:
                    lineup_rollups[team_id][column] += float(value)
        for team_id, rollup in lineup_rollups.items():
            for column, value in rollup.items():
                by_team[team_id].setdefault(column, value)
            if "successful_passes" not in by_team[team_id] and "accurate_passes" in rollup:
                by_team[team_id]["successful_passes"] = rollup["accurate_passes"]

        formations: dict[Any, Any] = {}
        for formation in _as_list(fixture.get("formations")):
            if isinstance(formation, dict):
                team_id = formation.get("participant_id") or formation.get("team_id")
                formations[team_id] = formation.get("formation") or formation.get("formation_name")

        fixture_team_rows: list[dict[str, Any]] = []
        for team_id, participant in participant_by_id.items():
            opponent = next((p for p in participants if p.get("id") != team_id), {})
            gf = score_map.get(team_id)
            ga = score_map.get(opponent.get("id"))
            result = None
            if gf is not None and ga is not None:
                result = "W" if gf > ga else "L" if gf < ga else "D"
            row = {
                **base,
                "team_id": team_id,
                "team_name": participant.get("name"),
                "location": location_by_id.get(team_id),
                "opponent_id": opponent.get("id"),
                "opponent": opponent.get("name"),
                "goals_for": gf,
                "goals_against": ga,
                "result": result,
                "formation": formations.get(team_id),
                "has_detail": has_team_detail,
                **by_team.get(team_id, {}),
            }
            fixture_team_rows.append(row)

        team_lookup = {row["team_id"]: row for row in fixture_team_rows}
        for row in fixture_team_rows:
            opponent_row = team_lookup.get(row["opponent_id"], {})
            row["shots_against"] = opponent_row.get("shots_total")
            row["opponent_passes"] = opponent_row.get("passes")
            row["opponent_final_third_passes"] = opponent_row.get("passes_final_third")
            row["expected_goals_against"] = opponent_row.get("expected_goals")
            row["expected_goals_on_target_against"] = opponent_row.get("expected_goals_on_target")
            row["expected_goals_set_play_against"] = opponent_row.get("expected_goals_set_play")
            row["expected_goals_corners_against"] = opponent_row.get("expected_goals_corners")
            self.team_matchlogs.append(row)

        player_xg: dict[Any, dict[str, Any]] = defaultdict(dict)
        for lineup in _as_list(fixture.get("lineups")):
            if not isinstance(lineup, dict):
                continue
            for detail in _as_list(lineup.get("xglineup")):
                if isinstance(detail, dict):
                    player_xg[lineup.get("player_id")].update(
                        _metrics([detail], catalog=self.metric_catalog, level="player_fixture_xg")
                    )

        for lineup in _as_list(fixture.get("lineups")):
            if not isinstance(lineup, dict):
                continue
            player = lineup.get("player") if isinstance(lineup.get("player"), dict) else {}
            player_row = _player_record(player, self.season_end_year)
            if not player_row.get("player_id"):
                player_row["player_id"] = lineup.get("player_id")
                player_row["player_name"] = lineup.get("player_name")
            self.players.append(player_row)
            team_id = lineup.get("team_id")
            team_row = team_lookup.get(team_id, {})
            lineup_key = lineup.get("id") or (lineup.get("team_id"), lineup.get("player_id"))
            details = dict(lineup_details.get(lineup_key, {}))
            details.update(player_xg.get(lineup.get("player_id"), {}))
            formation_field = lineup.get("formation_field")
            starter = lineup.get("type_id") == STARTING_LINEUP_TYPE_ID or bool(formation_field)
            position_id = lineup.get("position_id") or player_row.get("position_id")
            opponent_xgot = None
            if POSITION_CODES.get(position_id) == "GK":
                raw_xgot = _number(team_row.get("expected_goals_on_target_against"))
                minutes = _number(details.get("minutes_played"))
                if raw_xgot is not None and minutes is not None:
                    opponent_xgot = raw_xgot * min(minutes, 90) / 90
            self.player_matchlogs.append(
                {
                    **base,
                    "team_id": team_id,
                    "team_name": participant_by_id.get(team_id, {}).get("name"),
                    "opponent_id": team_row.get("opponent_id"),
                    "opponent": team_row.get("opponent"),
                    "location": team_row.get("location"),
                    "result": team_row.get("result"),
                    "player_id": lineup.get("player_id"),
                    "player_name": lineup.get("player_name") or player_row.get("player_name"),
                    "position_id": position_id,
                    "position": POSITION_CODES.get(position_id) or player_row.get("position"),
                    "jersey_number": lineup.get("jersey_number"),
                    "starter": starter,
                    "formation_field": formation_field,
                    "formation_position": lineup.get("formation_position"),
                    "team_expected_goals": team_row.get("expected_goals"),
                    "opponent_expected_goals_on_target": opponent_xgot,
                    **details,
                }
            )

        for event in _as_list(fixture.get("events")):
            if not isinstance(event, dict):
                continue
            type_data = event.get("type") if isinstance(event.get("type"), dict) else {}
            self.events.append(
                {
                    **{key: base[key] for key in ("league_id", "league", "season_id", "season", "fixture_id", "starting_at")},
                    "event_id": event.get("id"),
                    "team_id": event.get("participant_id"),
                    "team_name": participant_by_id.get(event.get("participant_id"), {}).get("name"),
                    "type_id": event.get("type_id"),
                    "event_type": type_data.get("name") or type_data.get("developer_name"),
                    "player_id": event.get("player_id"),
                    "player_name": event.get("player_name"),
                    "related_player_id": event.get("related_player_id"),
                    "related_player_name": event.get("related_player_name"),
                    "minute": event.get("minute"),
                    "extra_minute": event.get("extra_minute"),
                    "result": event.get("result"),
                    "info": event.get("info"),
                    "addition": event.get("addition"),
                    "on_bench": event.get("on_bench"),
                }
            )

    @staticmethod
    def _frame(rows: list[dict[str, Any]], dedupe: list[str] | None = None) -> pd.DataFrame:
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        if dedupe:
            keys = [key for key in dedupe if key in frame.columns]
            if keys:
                frame = frame.drop_duplicates(keys, keep="last")
        return frame.reset_index(drop=True)

    @staticmethod
    def _coalesce_players(rows: list[dict[str, Any]]) -> pd.DataFrame:
        """Deduplicate players without losing squad-only nationality metadata."""
        frame = pd.DataFrame(rows)
        if frame.empty or "player_id" not in frame:
            return frame

        def last_present(series: pd.Series) -> Any:
            present = series.dropna()
            present = present[present.astype(str).str.len() > 0]
            return present.iloc[-1] if not present.empty else np.nan

        value_columns = [column for column in frame.columns if column != "player_id"]
        return (
            frame.groupby("player_id", dropna=False, as_index=False)[value_columns]
            .agg(last_present)
            .reset_index(drop=True)
        )

    @staticmethod
    def _safe_divide(numerator: pd.Series, denominator: pd.Series, multiplier: float = 1.0) -> pd.Series:
        result = multiplier * pd.to_numeric(numerator, errors="coerce") / pd.to_numeric(denominator, errors="coerce")
        return result.replace([np.inf, -np.inf], np.nan).fillna(0)

    def _player_aggregate(self, matchlogs: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
        if matchlogs.empty:
            return pd.DataFrame()
        frame = matchlogs.copy()
        if "minutes_played" not in frame:
            frame["minutes_played"] = 0
        frame["minutes_played"] = pd.to_numeric(frame["minutes_played"], errors="coerce").fillna(0)
        frame["appearance"] = (frame["minutes_played"] > 0).astype(int)
        frame["start"] = (frame["starter"].fillna(False) & (frame["minutes_played"] > 0)).astype(int)

        non_additive = {
            "captain", "rating", "successful_passes_percentage", "accurate_passes_percentage",
            "successful_dribbles_percentage", "successful_crosses_percentage", "tackles_won_percentage",
            "aerials_won_percentage", "duels_won_percentage", "cumulative_minutes_played",
        }
        id_columns = {
            "league_id", "season_id", "fixture_id", "round_id", "stage_id", "venue_id", "state_id",
            "home_team_id", "away_team_id", "home_score", "away_score", "team_id", "opponent_id",
            "player_id", "position_id", "jersey_number", "formation_position",
        }
        numeric_candidates = [
            column for column in frame.select_dtypes(include=[np.number, "bool"]).columns
            if column not in id_columns and column not in non_additive
        ]
        group_keys = ["league_id", "league", "season_id", "season", "player_id"]
        aggregated = frame.groupby(group_keys, dropna=False)[numeric_candidates].sum(min_count=1).reset_index()

        dominant = (
            frame.groupby(group_keys + ["team_id", "team_name", "position"], dropna=False)["minutes_played"]
            .sum()
            .reset_index()
            .sort_values("minutes_played", ascending=False)
            .drop_duplicates(group_keys)
            .drop(columns="minutes_played")
        )
        aggregated = aggregated.merge(dominant, on=group_keys, how="left")
        if not players.empty:
            player_meta = players.drop_duplicates("player_id", keep="last")
            metadata = [
                column for column in (
                    "player_id", "player_name", "common_name", "first_name", "last_name", "date_of_birth", "age",
                    "nationality_id", "nationality", "nationality_code", "height", "weight", "image_path",
                ) if column in player_meta.columns
            ]
            aggregated = aggregated.merge(player_meta[metadata], on="player_id", how="left")
        name_from_log = frame.groupby(group_keys, dropna=False)["player_name"].last().reset_index()
        aggregated = aggregated.merge(name_from_log, on=group_keys, how="left", suffixes=("", "_log"))
        if "player_name_log" in aggregated:
            if "player_name" in aggregated:
                aggregated["player_name"] = aggregated["player_name"].fillna(aggregated["player_name_log"])
            else:
                aggregated["player_name"] = aggregated["player_name_log"]
            aggregated.drop(columns="player_name_log", inplace=True)
        aggregated["90s"] = pd.to_numeric(aggregated.get("minutes_played", 0), errors="coerce").fillna(0) / 90
        return aggregated

    def _player_compatibility(self, player_stats: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
        if player_stats.empty:
            return pd.DataFrame()
        df = player_stats.copy()
        league_mp = team_stats.groupby("League")["MP"].max().to_dict() if not team_stats.empty else {}

        def col(name: str) -> pd.Series:
            return pd.to_numeric(df[name], errors="coerce").fillna(0) if name in df else pd.Series(0.0, index=df.index)

        out = pd.DataFrame(index=df.index)
        out["Player"] = df.get("player_name")
        out["Pos"] = df.get("position")
        out["Age"] = df.get("age")
        out["Club"] = df.get("team_name")
        out["League"] = df.get("league")
        out["Nationality"] = df.get("nationality")
        out["Nationality_Code"] = df.get("nationality_code")
        out["image_path"] = df.get("image_path")
        out["MP"] = col("appearance")
        out["Starts"] = col("start")
        out["Min"] = col("minutes_played")
        out["90s"] = col("90s")
        out["Avg_Min_Per_Match"] = self._safe_divide(out["Min"], out["MP"])
        out["League_MP"] = out["League"].map(league_mp).fillna(0)
        out["Gls"] = col("goals")
        out["Ast"] = col("assists")
        out["xG"] = col("expected_goals")
        out["xGoT"] = col("expected_goals_on_target")
        out["Sh"] = col("shots_total")
        out["SoT"] = col("shots_on_target")
        out["Chances_Created"] = col("chances_created")
        out["Key_Passes"] = col("key_passes")
        out["Passes_Into_Final_Third"] = col("passes_final_third")
        out["Passes"] = col("passes")
        out["Accurate_Passes"] = col("accurate_passes")
        out["Dribble_Attempts"] = col("dribble_attempts")
        out["Successful_Dribbles"] = col("successful_dribbles")
        out["Tkl"] = col("tackles")
        out["TklW"] = col("tackles_won")
        out["Int"] = col("interceptions")
        out["Tkl+Int"] = out["TklW"] + out["Int"]
        out["Clr"] = col("clearances")
        out["Blocks"] = col("blocked_shots")
        out["Aerial_Won"] = col("aerials_won")
        out["Aerial_Lost"] = col("aerials_lost")
        out["Saves"] = col("saves")
        out["GA"] = col("goalkeeper_goals_conceded")
        out.loc[out["GA"] == 0, "GA"] = col("goals_conceded")
        goalkeeper = out["Pos"].astype(str).str.contains("GK", na=False)
        out.loc[~goalkeeper, ["Saves", "GA"]] = 0
        out["SoTA"] = out["Saves"] + out["GA"]
        out["PSxG"] = col("opponent_expected_goals_on_target")
        out.loc[~goalkeeper, "PSxG"] = 0
        out["Gls_per_90"] = self._safe_divide(out["Gls"], out["90s"])
        out["Ast_per_90"] = self._safe_divide(out["Ast"], out["90s"])
        out["xG_per_90"] = self._safe_divide(out["xG"], out["90s"])
        out["Sh_per_90"] = self._safe_divide(out["Sh"], out["90s"])
        out["xG_per_Shot"] = self._safe_divide(out["xG"], out["Sh"])
        out["G_minus_xG_per_90"] = self._safe_divide(out["Gls"] - out["xG"], out["90s"])
        out["Chances_Created_per_90"] = self._safe_divide(out["Chances_Created"], out["90s"])
        out["Key_Passes_per_90"] = self._safe_divide(out["Key_Passes"], out["90s"])
        out["Passes_F3_per_90"] = self._safe_divide(out["Passes_Into_Final_Third"], out["90s"])
        out["Successful_Dribbles_per_90"] = self._safe_divide(col("successful_dribbles"), out["90s"])
        out["Pass_Completion_Perc"] = self._safe_divide(out["Accurate_Passes"], out["Passes"], 100)
        out["Dribble_Success_Perc"] = self._safe_divide(out["Successful_Dribbles"], out["Dribble_Attempts"], 100)
        out["TklW_per_90"] = self._safe_divide(out["TklW"], out["90s"])
        out["Int_per_90"] = self._safe_divide(out["Int"], out["90s"])
        out["Tkl+Int_per_90"] = self._safe_divide(out["Tkl+Int"], out["90s"])
        out["Aerial_Duels_perc"] = self._safe_divide(out["Aerial_Won"], out["Aerial_Won"] + out["Aerial_Lost"], 100)
        out["Clr_per_90"] = self._safe_divide(out["Clr"], out["90s"])
        out["Blocks_per_90"] = self._safe_divide(out["Blocks"], out["90s"])
        out["SoT/90"] = self._safe_divide(out["SoT"], out["90s"])
        out["Saves_per_90"] = self._safe_divide(out["Saves"], out["90s"])
        out["GA_per_90"] = self._safe_divide(out["GA"], out["90s"])
        out["Save%"] = self._safe_divide(out["Saves"], out["SoTA"], 100)
        out["PSxG+/-"] = out["PSxG"] - out["GA"]
        # Sportmonks names this input xG on Target (xGoT), not post-shot xG.
        # Keep the historical PSxG aliases for backwards compatibility while
        # exposing accurately named fields to all new UI definitions.
        out["xGoT_Faced"] = out["PSxG"]
        out["xGoT_minus_GA"] = out["xGoT_Faced"] - out["GA"]
        out["xGoT_minus_GA_per_90"] = self._safe_divide(out["xGoT_minus_GA"], out["90s"])
        # These FBref definitions have no equivalent in the audited API plan.
        for unsupported in ("SCA90", "PrgP_per_90", "Carries_F3_per_90", "Stp%", "#OPA/90"):
            out[unsupported] = np.nan
        out["Data_Source"] = "Sportmonks"
        out["sportmonks_player_id"] = df.get("player_id")
        out["sportmonks_team_id"] = df.get("team_id")
        out["sportmonks_league_id"] = df.get("league_id")
        if not team_stats.empty and {"sportmonks_team_id", "image_path"}.issubset(team_stats.columns):
            team_images = (
                team_stats.dropna(subset=["sportmonks_team_id"])
                .drop_duplicates("sportmonks_team_id")
                .set_index("sportmonks_team_id")["image_path"]
                .to_dict()
            )
            out["team_image_path"] = out["sportmonks_team_id"].map(team_images)
        else:
            out["team_image_path"] = None
        return out

    def _team_aggregate(self, matchlogs: pd.DataFrame) -> pd.DataFrame:
        if matchlogs.empty:
            return pd.DataFrame()
        frame = matchlogs.copy()
        if "has_detail" in frame:
            frame = frame[frame["has_detail"].fillna(False).astype(bool)].copy()
        if frame.empty:
            return pd.DataFrame()
        frame["played"] = frame["goals_for"].notna().astype(int)
        frame["win"] = (frame["result"] == "W").astype(int)
        frame["draw"] = (frame["result"] == "D").astype(int)
        frame["loss"] = (frame["result"] == "L").astype(int)
        id_columns = {
            "league_id", "season_id", "fixture_id", "round_id", "stage_id", "venue_id", "state_id",
            "home_team_id", "away_team_id", "team_id", "opponent_id", "home_score", "away_score",
        }
        non_additive = {"has_detail", "possession_percentage", "successful_passes_percentage"}
        numeric = [
            column for column in frame.select_dtypes(include=[np.number, "bool"]).columns
            if column not in id_columns and column not in non_additive
        ]
        keys = ["league_id", "league", "season_id", "season", "team_id", "team_name"]
        agg = frame.groupby(keys, dropna=False)[numeric].sum(min_count=1).reset_index()

        def col(name: str) -> pd.Series:
            return pd.to_numeric(agg[name], errors="coerce").fillna(0) if name in agg else pd.Series(0.0, index=agg.index)

        out = pd.DataFrame(index=agg.index)
        out["sportmonks_league_id"] = agg["league_id"]
        out["sportmonks_team_id"] = agg["team_id"]
        out["Squad"] = agg["team_name"]
        out["League"] = agg["league"]
        out["Season"] = agg["season"]
        out["MP"] = col("played")
        out["W"] = col("win")
        out["D"] = col("draw")
        out["L"] = col("loss")
        out["Pts"] = (3 * out["W"]) + out["D"]
        out["Pts_per_MP"] = self._safe_divide(out["Pts"], out["MP"])
        out["90s"] = out["MP"]
        mappings = {
            "Gls": "goals_for", "GA": "goals_against", "xG": "expected_goals",
            "xGA": "expected_goals_against", "npxG": "expected_non_penalty_goals",
            "xGoT": "expected_goals_on_target", "Shooting_Performance": "shooting_performance",
            "Set_Piece_xG": "expected_goals_set_play", "Corner_xG": "expected_goals_corners",
            "Set_Piece_xGA": "expected_goals_set_play_against",
            "Poss": "possession_percentage", "Sh": "shots_total", "SoT": "shots_on_target",
            "Shots_Against": "shots_against", "Opp_Passes": "opponent_passes",
            "Opp_FinalThirdPasses": "opponent_final_third_passes",
            "Ast": "assists", "Att": "passes",
            "Cmp": "successful_passes", "KP": "key_passes", "PassesFinal3rd": "passes_final_third",
            "Touches": "touches", "AttTakeOn": "dribble_attempts", "SuccTakeOn": "successful_dribbles",
            "Tkl": "tackles", "TklW": "tackles_won", "Int": "interceptions", "Fls": "fouls",
            "Crs": "total_crosses", "Corners": "corners", "Chances_Created": "chances_created",
            "AerialsWon": "aerials_won", "AerialsLost": "aerials_lost", "Errors": "error_lead_to_shot",
        }
        for target, source in mappings.items():
            out[target] = col(source)
        # Possession is a percentage and must be averaged, not summed.
        if "possession_percentage" in frame:
            possession = frame.groupby(keys, dropna=False)["possession_percentage"].mean().reset_index(drop=True)
            out["Poss"] = pd.to_numeric(possession, errors="coerce").fillna(0)
        out["Goal_Conversion"] = self._safe_divide(out["Gls"], out["Sh"], 100)
        out["Goal_Conversion_Perc"] = out["Goal_Conversion"]
        out["GD"] = out["Gls"] - out["GA"]
        out["Passing_Tempo"] = self._safe_divide(out["Att"], out["Touches"])
        out["Pass_Completion_Perc"] = self._safe_divide(out["Cmp"], out["Att"], 100)
        out["Aerial_Duels_Won_Perc"] = self._safe_divide(out["AerialsWon"], out["AerialsWon"] + out["AerialsLost"], 100)
        out["Defensive_Actions"] = out["TklW"] + out["Int"]
        out["Shots_Conceded_per_DA"] = self._safe_divide(out["Shots_Against"], out["Defensive_Actions"])
        out["xG_per_Shot"] = self._safe_divide(out["xG"], out["Sh"])
        out["xG_Overperformance"] = out["Gls"] - out["xG"]
        out["Gls_per_90"] = self._safe_divide(out["Gls"], out["90s"])
        out["GA_per_90"] = self._safe_divide(out["GA"], out["90s"])
        out["xG_per_90"] = self._safe_divide(out["xG"], out["90s"])
        out["npxG_per_90"] = self._safe_divide(out["npxG"], out["90s"])
        out["xGA_per_90"] = self._safe_divide(out["xGA"], out["90s"])
        out["Shooting_Performance_per_90"] = self._safe_divide(out["Shooting_Performance"], out["90s"])
        out["Sh_per_90"] = self._safe_divide(out["Sh"], out["90s"])
        out["Shots_Against_per_90"] = self._safe_divide(out["Shots_Against"], out["90s"])
        out["xGA_per_Shot_Against"] = self._safe_divide(out["xGA"], out["Shots_Against"])
        out["KP_per_90"] = self._safe_divide(out["KP"], out["90s"])
        out["FinalThird_per_90"] = self._safe_divide(out["PassesFinal3rd"], out["90s"])
        out["FinalThird_per_100_Passes"] = self._safe_divide(out["PassesFinal3rd"], out["Att"], 100)
        out["Chances_Created_per_90"] = self._safe_divide(out["Chances_Created"], out["90s"])
        out["Progressions_per_Touch"] = self._safe_divide(out["PassesFinal3rd"], out["Touches"])
        out["Cross_Touch_Ratio"] = self._safe_divide(out["Crs"], out["Touches"])
        out["TakeOn_Success_Rate"] = self._safe_divide(out["SuccTakeOn"], out["AttTakeOn"])
        out["TakeOn_Success_Perc"] = self._safe_divide(out["SuccTakeOn"], out["AttTakeOn"], 100)
        out["Tkl_Int_per_90"] = self._safe_divide(out["Defensive_Actions"], out["90s"])
        out["Defensive_Actions_per_90"] = out["Tkl_Int_per_90"]
        out["Pressure_Activity_per_100_Opp_Passes"] = self._safe_divide(
            out["Tkl"] + out["Int"] + out["Fls"], out["Opp_Passes"], 100
        )
        global_defensive_actions = out["Tkl"] + out["Int"] + out["Fls"]
        out["Global_PPDA_Proxy"] = (
            pd.to_numeric(out["Opp_Passes"], errors="coerce")
            / global_defensive_actions.replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        out.loc[out["Opp_Passes"] <= 0, "Global_PPDA_Proxy"] = np.nan
        out["Errors_per_GA"] = self._safe_divide(out["Errors"], out["GA"])
        out["Errors_per_90"] = self._safe_divide(out["Errors"], out["90s"])
        out["Set_Piece_xG_per_90"] = self._safe_divide(out["Set_Piece_xG"], out["90s"])
        out["Corner_xG_per_Corner"] = self._safe_divide(out["Corner_xG"], out["Corners"])
        out["Set_Piece_xGA_per_90"] = self._safe_divide(out["Set_Piece_xGA"], out["90s"])
        out["Set_Piece_xG_Difference_per_90"] = self._safe_divide(
            out["Set_Piece_xG"] - out["Set_Piece_xGA"], out["90s"]
        )
        out["Ast_xAG_ratio"] = np.nan
        out["Data_Source"] = "Sportmonks"
        return out

    @staticmethod
    def _save_frame(frame: pd.DataFrame, directory: Path, name: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_csv(directory / f"{name}.csv", index=False)
        try:
            frame.to_parquet(directory / f"{name}.parquet", index=False)
        except ImportError as exc:
            raise RuntimeError("Parquet support is missing. Install pyarrow and rerun the importer.") from exc

    def frames(self) -> dict[str, pd.DataFrame]:
        players = self._coalesce_players(self.players)
        teams = self._frame(self.teams, ["league_id", "season_id", "team_id"])
        fixtures = self._frame(self.fixtures, ["fixture_id"])
        team_matchlogs = self._frame(self.team_matchlogs, ["fixture_id", "team_id"])
        player_matchlogs = self._frame(self.player_matchlogs, ["fixture_id", "player_id", "team_id"])
        team_stats = self._team_aggregate(team_matchlogs)
        if not team_stats.empty and not teams.empty:
            team_meta = teams[["team_id", "image_path"]].drop_duplicates("team_id")
            team_stats = team_stats.merge(
                team_meta,
                left_on="sportmonks_team_id",
                right_on="team_id",
                how="left",
            ).drop(columns="team_id")
        player_stats = self._player_aggregate(player_matchlogs, players)
        player_compat = self._player_compatibility(player_stats, team_stats)
        catalog = self._frame(self.metric_catalog, ["level", "type_id", "column"])
        return {
            "leagues": self._frame(self.leagues, ["league_id", "season_id"]),
            "teams": teams,
            "players": players,
            "squads": self._frame(self.squads, ["league_id", "season_id", "team_id", "player_id"]),
            "fixtures": fixtures,
            "standings": self._frame(self.standings, ["league_id", "season_id", "team_id"]),
            "events": self._frame(self.events, ["event_id"]),
            "team_matchlogs": team_matchlogs,
            "player_matchlogs": player_matchlogs,
            "team_season_api": self._frame(self.team_season_api, ["league_id", "season_id", "team_id"]),
            "player_season_api": self._frame(self.player_season_api, ["league_id", "season_id", "team_id", "player_id"]),
            "metric_catalog": catalog,
            "team_stats": team_stats,
            "player_stats": player_stats,
            "player_stats_compat": player_compat,
        }

    def save(self, project_root: Path) -> dict[str, pd.DataFrame]:
        frames = self.frames()
        canonical = project_root / "data" / "sportmonks" / "processed" / self.season_name
        selected_league_ids = {
            row.get("league_id") for row in self.leagues if row.get("league_id") is not None
        }
        dedupe_keys = {
            "leagues": ["league_id", "season_id"],
            "teams": ["league_id", "season_id", "team_id"],
            "players": ["player_id"],
            "squads": ["league_id", "season_id", "team_id", "player_id"],
            "fixtures": ["fixture_id"],
            "standings": ["league_id", "season_id", "team_id"],
            "events": ["event_id"],
            "team_matchlogs": ["fixture_id", "team_id"],
            "player_matchlogs": ["fixture_id", "player_id", "team_id"],
            "team_season_api": ["league_id", "season_id", "team_id"],
            "player_season_api": ["league_id", "season_id", "team_id", "player_id"],
            "metric_catalog": ["level", "type_id", "column"],
            "team_stats": ["sportmonks_league_id", "sportmonks_team_id"],
            "player_stats": ["league_id", "season_id", "player_id"],
            "player_stats_compat": ["sportmonks_league_id", "sportmonks_player_id"],
        }

        def merge_previous(name: str, frame: pd.DataFrame, path: Path) -> pd.DataFrame:
            if not path.exists():
                return frame
            try:
                previous = pd.read_parquet(path)
            except Exception:
                return frame
            if previous.empty:
                return frame
            league_column = "league_id" if "league_id" in previous.columns else (
                "sportmonks_league_id" if "sportmonks_league_id" in previous.columns else None
            )
            if league_column and selected_league_ids:
                previous = previous[~previous[league_column].isin(selected_league_ids)]
            elif name not in {"players", "metric_catalog"}:
                # Do not mix a legacy compatibility dataset with Sportmonks rows.
                return frame
            combined = pd.concat([previous, frame], ignore_index=True, sort=False)
            keys = [key for key in dedupe_keys.get(name, []) if key in combined.columns]
            return combined.drop_duplicates(keys, keep="last") if keys else combined.drop_duplicates()

        merged_frames: dict[str, pd.DataFrame] = {}
        for name, frame in frames.items():
            if name.endswith("_compat"):
                continue
            frame = merge_previous(name, frame, canonical / f"{name}.parquet")
            merged_frames[name] = frame
            self._save_frame(frame, canonical, name)

        compatibility = project_root / "data" / "processed"
        player_compat = merge_previous(
            "player_stats_compat",
            frames["player_stats_compat"],
            compatibility / f"player_stats_{self.season_name}.parquet",
        )
        team_compat = merge_previous(
            "team_stats",
            frames["team_stats"],
            compatibility / f"team_stats_{self.season_name}.parquet",
        )
        self._save_frame(player_compat, compatibility, f"player_stats_{self.season_name}")
        self._save_frame(team_compat, compatibility, f"team_stats_{self.season_name}")
        merged_frames["player_stats_compat"] = player_compat
        merged_frames["team_stats"] = team_compat
        return merged_frames
