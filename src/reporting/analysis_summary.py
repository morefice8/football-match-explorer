"""Compact AI-oriented tactical summary for the Match Analysis Pack.

REPORT-19 deliberately consumes only the already-built neutral report bundle and
static figure catalog. It does not calculate new football metrics and it never
copies complete event tables into the summary.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.reporting.bundle import normalize_for_json
from src.reporting.selectors import rank_players_by_metric_family


ANALYSIS_SUMMARY_SCHEMA_VERSION = "1.0"
ANALYSIS_SUMMARY_MAX_BYTES = 1024 * 1024
MAX_RANKING_ROWS = 10
MAX_CROSS_ROUTES = 8
MAX_REPRESENTATIVE_SEQUENCE_EVENTS = 120
ANALYSIS_SUMMARY_SCHEMA_FILE = (
    Path(__file__).resolve().parent
    / "schemas"
    / "analysis-summary.schema.json"
)


def _section(bundle, section_id: str):
    try:
        return bundle.section(section_id)
    except Exception:
        return None


def _section_data(bundle, section_id: str) -> Any:
    section = _section(bundle, section_id)
    if section is None:
        return {}
    return getattr(section, "data", None) or {}


def _status_value(section) -> str | None:
    if section is None:
        return None
    status = getattr(section, "status", None)
    return getattr(status, "value", status)


def _teams(bundle) -> tuple[str, ...]:
    return tuple(str(team) for team in (getattr(bundle, "teams", ()) or ()))


def _as_frame(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
        if (
            not isinstance(frame.index, pd.RangeIndex)
            or frame.index.name is not None
        ):
            raw_names = (
                list(frame.index.names)
                if isinstance(frame.index, pd.MultiIndex)
                else [frame.index.name]
            )
            names = []
            used = {str(column) for column in frame.columns}
            for position, raw in enumerate(raw_names):
                base = str(raw).strip() if raw not in (None, "") else (
                    "item" if len(raw_names) == 1 else f"index_level_{position}"
                )
                name = base
                suffix = 1
                while name in used or name in names:
                    name = (
                        f"index_{base}"
                        if suffix == 1
                        else f"index_{base}_{suffix}"
                    )
                    suffix += 1
                names.append(name)
                used.add(name)
            frame = frame.reset_index(names=names)
        return frame
    if isinstance(value, pd.Series):
        return value.to_frame().T.reset_index(drop=True)
    if isinstance(value, Mapping):
        return pd.DataFrame([dict(value)])
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        if value and all(isinstance(item, Mapping) for item in value):
            return pd.DataFrame([dict(item) for item in value])
    return pd.DataFrame()


def _column(frame: pd.DataFrame, *aliases: str) -> Any | None:
    lookup = {
        str(column).strip().casefold(): column
        for column in frame.columns
    }
    for alias in aliases:
        found = lookup.get(str(alias).strip().casefold())
        if found is not None:
            return found
    return None


def _first(mapping: Mapping[str, Any] | None, *aliases: str) -> Any:
    if not isinstance(mapping, Mapping):
        return None
    by_name = {
        str(key).strip().casefold(): value
        for key, value in mapping.items()
    }
    for alias in aliases:
        value = by_name.get(str(alias).strip().casefold())
        if value not in (None, ""):
            try:
                if pd.isna(value):
                    continue
            except (TypeError, ValueError):
                pass
            return value
    return None


def _native(value: Any) -> Any:
    return normalize_for_json(value)


def _safe_number(value: Any) -> int | float | None:
    value = _native(value)
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _snake_key(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text or "value"


def _compact_mapping(
    value: Any,
    *,
    depth: int = 0,
    max_depth: int = 4,
) -> dict[str, Any]:
    """Keep compact scalar summary data; reject table/event-shaped payloads."""
    if not isinstance(value, Mapping) or depth > max_depth:
        return {}

    result: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = _snake_key(raw_key)

        if isinstance(raw_value, (pd.DataFrame, pd.Series)):
            continue

        if isinstance(raw_value, Mapping):
            nested = _compact_mapping(
                raw_value,
                depth=depth + 1,
                max_depth=max_depth,
            )
            if nested:
                result[key] = nested
            continue

        if isinstance(raw_value, Sequence) and not isinstance(
            raw_value, (str, bytes, bytearray)
        ):
            if len(raw_value) <= 40 and all(
                not isinstance(item, (Mapping, pd.DataFrame, pd.Series))
                for item in raw_value
            ):
                result[key] = _native(list(raw_value))
            continue

        result[key] = _native(raw_value)

    return result


def _metadata_value(info: Mapping[str, Any], *aliases: str) -> Any:
    return _first(info, *aliases)


def _match_summary(bundle) -> dict[str, Any]:
    teams = _teams(bundle)
    info = getattr(bundle, "match_info", {}) or {}
    scope = getattr(bundle, "scope", None)
    scope_value = getattr(scope, "value", scope)

    return {
        "home_team": teams[0] if len(teams) > 0 else None,
        "away_team": teams[1] if len(teams) > 1 else None,
        "competition": _native(
            _metadata_value(
                info,
                "competition",
                "competitionName",
                "competition_name",
                "tournament",
            )
        ),
        "date": _native(
            _metadata_value(
                info,
                "date",
                "match_date",
                "matchDate",
                "game_date",
                "date_iso",
                "date_formatted",
            )
        ),
        "match_id": _native(
            _metadata_value(
                info,
                "match_id",
                "matchId",
                "game_id",
                "gameId",
            )
        ),
        "venue": _native(
            _metadata_value(
                info,
                "venue",
                "venueName",
                "stadium",
            )
        ),
        "scope": str(scope_value) if scope_value is not None else None,
        "source_signature": _native(
            getattr(bundle, "source_signature", None)
        ),
    }


def _goal_rows(bundle) -> list[dict[str, Any]]:
    data = _section_data(bundle, "overview")
    if not isinstance(data, Mapping):
        return []

    origins = [
        item
        for item in (data.get("goal_origins", []) or [])
        if isinstance(item, Mapping)
    ]
    scorers = [
        item
        for item in (data.get("scorers", []) or [])
        if isinstance(item, Mapping)
    ]

    by_event = {
        str(_first(item, "goal_event_id", "eventId", "id")): item
        for item in scorers
        if _first(item, "goal_event_id", "eventId", "id") not in (None, "")
    }

    source = origins if origins else scorers
    rows = []

    for item in source:
        event_id = _first(
            item,
            "goal_event_id",
            "eventId",
            "id",
            "optaEventId",
        )
        scorer_row = by_event.get(str(event_id), {})
        scorer = (
            _first(scorer_row, "scorer", "playerName", "player_name")
            or _first(item, "scorer", "playerName", "player_name")
        )
        team = (
            _first(scorer_row, "team_name", "teamName")
            or _first(item, "team_name", "teamName")
        )
        assist = _first(
            item,
            "official_assist",
            "assist",
            "assist_player",
        )
        shot_creating = _first(
            item,
            "shot_creating_pass",
            "shotCreatingPass",
        )

        creator_type = None
        creator = None
        if assist not in (None, ""):
            creator_type = "assist"
            creator = assist
        elif shot_creating not in (None, ""):
            creator_type = "shot_creating_pass"
            creator = shot_creating

        rows.append(
            {
                "event_id": _native(event_id),
                "team": _native(team),
                "scorer": _native(scorer),
                "minute": _safe_number(
                    _first(item, "minute", "timeMin")
                    or _first(scorer_row, "minute", "timeMin")
                ),
                "second": _safe_number(
                    _first(item, "second", "timeSec")
                    or _first(scorer_row, "second", "timeSec")
                ),
                "possession_origin": _native(
                    _first(item, "possession_origin")
                ),
                "origin_detail": _native(
                    _first(item, "origin_detail")
                ),
                "attack_type": _native(
                    _first(item, "attack_type")
                ),
                "decisive_mechanism": _native(
                    _first(item, "decisive_mechanism")
                ),
                "creator_type": creator_type,
                "creator": _native(creator),
                "possession_duration_seconds": _safe_number(
                    _first(item, "possession_duration_seconds")
                ),
                "pass_count": _safe_number(
                    _first(item, "pass_count")
                ),
                "analysis_module": _native(
                    _first(
                        item,
                        "analysis_module",
                        "analytic_module",
                        "module",
                    )
                ),
            }
        )

    return rows


def _score_and_goals(bundle) -> dict[str, Any]:
    data = _section_data(bundle, "overview")
    result = data.get("result", {}) if isinstance(data, Mapping) else {}
    result = result if isinstance(result, Mapping) else {}

    return {
        "home_score": _safe_number(result.get("home_score")),
        "away_score": _safe_number(result.get("away_score")),
        "goals": _goal_rows(bundle),
    }


def _data_quality(bundle) -> dict[str, Any]:
    data = _section_data(bundle, "overview")
    coverage = (
        data.get("data_coverage", {})
        if isinstance(data, Mapping)
        else {}
    )
    coverage = coverage if isinstance(coverage, Mapping) else {}

    receiver = coverage.get("receiver", {}) or {}
    coordinates = coverage.get("coordinates", {}) or {}
    outcome = coverage.get("outcome", {}) or {}
    qualifiers = coverage.get("qualifiers", {}) or {}
    carries = coverage.get("final_third_carries", {}) or {}

    carry_rows = []
    for team in _teams(bundle):
        payload = carries.get(team, {}) if isinstance(carries, Mapping) else {}
        carry_rows.append(
            {
                "team": team,
                **_compact_mapping(payload if isinstance(payload, Mapping) else {}),
            }
        )

    return {
        "event_rows": _safe_number(coverage.get("event_rows")),
        "receiver": _compact_mapping(
            receiver if isinstance(receiver, Mapping) else {}
        ),
        "coordinates": _compact_mapping(
            coordinates if isinstance(coordinates, Mapping) else {}
        ),
        "outcome": _compact_mapping(
            outcome if isinstance(outcome, Mapping) else {}
        ),
        "unmapped_qualifiers": {
            "count": _safe_number(
                qualifiers.get("unmapped_count")
                if isinstance(qualifiers, Mapping)
                else None
            ),
            "ids": _native(
                list(qualifiers.get("unmapped_ids", ()) or ())
                if isinstance(qualifiers, Mapping)
                else []
            ),
        },
        "final_third_carries": carry_rows,
    }


def _team_comparison(bundle) -> list[dict[str, Any]]:
    data = _section_data(bundle, "overview")
    profile = data.get("game_profile", {}) if isinstance(data, Mapping) else {}
    profile = profile if isinstance(profile, Mapping) else {}

    fields = (
        "passes",
        "successful_passes",
        "pass_completion_pct",
        "shots",
        "shots_on_target",
        "progressive_passes",
        "final_third_entries",
        "crosses",
        "cross_retention_pct",
        "ball_recoveries",
    )

    rows = []
    for index, team in enumerate(_teams(bundle)):
        source = profile.get(team, {}) if isinstance(profile, Mapping) else {}
        source = source if isinstance(source, Mapping) else {}
        row = {
            "side": "home" if index == 0 else "away",
            "team": team,
        }
        for field in fields:
            row[field] = _native(source.get(field))
        rows.append(row)
    return rows


def _formations(bundle) -> dict[str, Any]:
    data = _section_data(bundle, "formation-timeline")
    if not isinstance(data, Mapping):
        data = {}

    moments = [
        item
        for item in (data.get("moments", []) or [])
        if isinstance(item, Mapping)
    ]

    compact = []
    for moment in moments[:30]:
        event_kinds = []
        for event in (moment.get("events", []) or []):
            if not isinstance(event, Mapping):
                continue
            kind = str(event.get("kind") or "").strip()
            if kind and kind not in event_kinds:
                event_kinds.append(kind)

        time_seconds = _safe_number(moment.get("time_seconds"))
        minute = _safe_number(moment.get("minute"))
        if minute is None and time_seconds is not None:
            minute = int(float(time_seconds) // 60)

        compact.append(
            {
                "time_seconds": time_seconds,
                "time_label": _native(moment.get("time_label")),
                "minute": minute,
                "score": _native(moment.get("score")),
                "home_formation": _native(
                    moment.get("home_formation_name")
                ),
                "away_formation": _native(
                    moment.get("away_formation_name")
                ),
                "change_reasons": event_kinds,
            }
        )

    return {
        "starting": compact[0] if compact else None,
        "final": compact[-1] if compact else None,
        "timeline": compact,
    }


def _sort_frame(
    frame: pd.DataFrame,
    aliases: tuple[str, ...],
) -> pd.DataFrame:
    column = _column(frame, *aliases)
    if column is None or frame.empty:
        return frame
    numeric = pd.to_numeric(frame[column], errors="coerce")
    return (
        frame.assign(_report19_sort=numeric)
        .sort_values(
            "_report19_sort",
            ascending=False,
            na_position="last",
            kind="stable",
        )
        .drop(columns=["_report19_sort"])
    )


def _row_value(row: pd.Series, frame: pd.DataFrame, *aliases: str) -> Any:
    column = _column(frame, *aliases)
    if column is None:
        return None
    return _native(row.get(column))


def _passing(bundle) -> dict[str, Any]:
    data = _section_data(bundle, "pass-network")
    teams_data = data.get("teams", {}) if isinstance(data, Mapping) else {}
    teams_data = teams_data if isinstance(teams_data, Mapping) else {}

    rows = []
    for team in _teams(bundle):
        payload = teams_data.get(team, {}) or {}
        payload = payload if isinstance(payload, Mapping) else {}
        nodes = _sort_frame(
            _as_frame(payload.get("nodes")),
            ("pass_involvement",),
        ).head(MAX_RANKING_ROWS)

        leaders = []
        for _, row in nodes.iterrows():
            leaders.append(
                {
                    "player": _row_value(
                        row, nodes,
                        "playerName", "player_name", "Player", "item",
                    ),
                    "jersey_number": _row_value(
                        row, nodes, "jersey_number", "jersey",
                    ),
                    "passes_sent": _safe_number(
                        _row_value(row, nodes, "pass_sent")
                    ),
                    "passes_received": _safe_number(
                        _row_value(row, nodes, "pass_received")
                    ),
                    "pass_involvement": _safe_number(
                        _row_value(row, nodes, "pass_involvement")
                    ),
                    "minutes": _safe_number(
                        _row_value(row, nodes, "minutes")
                    ),
                }
            )

        rows.append(
            {
                "team": team,
                "network_summary": _compact_mapping(
                    payload.get("summary", {})
                    if isinstance(payload.get("summary"), Mapping)
                    else {}
                ),
                "top_involvement": leaders,
            }
        )

    return {"teams": rows}


def _progression(bundle) -> dict[str, Any]:
    data = _section_data(bundle, "progressive-passes")
    teams_data = data.get("teams", {}) if isinstance(data, Mapping) else {}
    teams_data = teams_data if isinstance(teams_data, Mapping) else {}

    rows = []
    for team in _teams(bundle):
        payload = teams_data.get(team, {}) or {}
        payload = payload if isinstance(payload, Mapping) else {}

        ranking = _sort_frame(
            _as_frame(payload.get("player_ranking")),
            ("Successful", "successful", "Progressive Passes"),
        ).head(MAX_RANKING_ROWS)

        leaders = []
        for _, row in ranking.iterrows():
            leaders.append(
                {
                    "player": _row_value(
                        row, ranking,
                        "Player", "playerName", "player_name", "item",
                    ),
                    "successful": _safe_number(
                        _row_value(row, ranking, "Successful", "successful")
                    ),
                    "attempted": _safe_number(
                        _row_value(row, ranking, "Attempted", "attempted")
                    ),
                    "completion_pct": _safe_number(
                        _row_value(
                            row, ranking,
                            "Completion %", "completion_pct",
                        )
                    ),
                    "progression_m": _safe_number(
                        _row_value(
                            row, ranking,
                            "Progression m", "progression_m",
                        )
                    ),
                }
            )

        summary = payload.get("summary", {})
        rows.append(
            {
                "team": team,
                "summary": _compact_mapping(
                    summary if isinstance(summary, Mapping) else {}
                ),
                "top_passers": leaders,
            }
        )

    return {"teams": rows}


def _final_third_entries(bundle) -> dict[str, Any]:
    data = _section_data(bundle, "final-third-entries")
    teams_data = data.get("teams", {}) if isinstance(data, Mapping) else {}
    teams_data = teams_data if isinstance(teams_data, Mapping) else {}

    rows = []
    for team in _teams(bundle):
        payload = teams_data.get(team, {}) or {}
        stats = payload.get("stats", {}) if isinstance(payload, Mapping) else {}
        rows.append(
            {
                "team": team,
                "summary": _compact_mapping(
                    stats if isinstance(stats, Mapping) else {}
                ),
            }
        )
    return {"teams": rows}


def _crosses(bundle) -> dict[str, Any]:
    data = _section_data(bundle, "cross-flow")
    data = data if isinstance(data, Mapping) else {}
    rows = []

    for team in _teams(bundle):
        payload = data.get(team, {}) or {}
        payload = payload if isinstance(payload, Mapping) else {}
        routes = _sort_frame(
            _as_frame(payload.get("routes")),
            ("Crosses", "crosses", "count"),
        ).head(MAX_CROSS_ROUTES)

        route_rows = []
        for _, row in routes.iterrows():
            route_rows.append(
                {
                    "origin_zone": _row_value(
                        row, routes,
                        "Origin Zone", "origin_zone", "origin",
                    ),
                    "destination_zone": _row_value(
                        row, routes,
                        "Destination Zone", "destination_zone", "destination",
                    ),
                    "crosses": _safe_number(
                        _row_value(row, routes, "Crosses", "crosses", "count")
                    ),
                    "share_pct": _safe_number(
                        _row_value(row, routes, "Share %", "share_pct")
                    ),
                    "completion_pct": _safe_number(
                        _row_value(
                            row, routes,
                            "Completion %", "completion_pct",
                        )
                    ),
                    "retention_pct": _safe_number(
                        _row_value(
                            row, routes,
                            "Retention %", "retention_pct",
                        )
                    ),
                    "shots": _safe_number(
                        _row_value(row, routes, "Shots", "shots")
                    ),
                    "shot_rate_pct": _safe_number(
                        _row_value(
                            row, routes,
                            "Shot Rate %", "shot_rate_pct",
                        )
                    ),
                }
            )

        summary = payload.get("summary", {})
        rows.append(
            {
                "team": team,
                "summary": _compact_mapping(
                    summary if isinstance(summary, Mapping) else {}
                ),
                "top_routes": route_rows,
            }
        )

    return {"teams": rows}


def _sequence_outcome_counts(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    column = _column(
        frame,
        "terminal_outcome",
        "final_outcome",
        "sequence_outcome_type",
    )
    if column is None:
        return []
    counts = (
        frame[column]
        .fillna("unknown")
        .astype(str)
        .value_counts(dropna=False)
    )
    return [
        {"outcome": str(label), "count": int(count)}
        for label, count in counts.head(12).items()
    ]


def _buildup(bundle) -> dict[str, Any]:
    data = _section_data(bundle, "build-up")
    data = data if isinstance(data, Mapping) else {}
    teams_data = data.get("teams", {}) or {}
    teams_data = teams_data if isinstance(teams_data, Mapping) else {}

    rows = []
    for team in _teams(bundle):
        payload = teams_data.get(team, {}) or {}
        payload = payload if isinstance(payload, Mapping) else {}
        summary = _as_frame(payload.get("summary"))

        duration_col = _column(summary, "duration_seconds")
        pass_col = _column(summary, "pass_count")
        rows.append(
            {
                "team": team,
                "sequences": int(len(summary)),
                "avg_duration_seconds": (
                    _safe_number(
                        pd.to_numeric(
                            summary[duration_col],
                            errors="coerce",
                        ).mean()
                    )
                    if duration_col is not None and not summary.empty
                    else None
                ),
                "avg_completed_passes": (
                    _safe_number(
                        pd.to_numeric(
                            summary[pass_col],
                            errors="coerce",
                        ).mean()
                    )
                    if pass_col is not None and not summary.empty
                    else None
                ),
                "outcome_summary": _sequence_outcome_counts(summary),
            }
        )

    comparison = data.get("comparison", {})
    return {
        "teams": rows,
        "comparison": _native(comparison)
        if isinstance(comparison, Mapping)
        else {},
    }


def _defensive_shape(bundle) -> dict[str, Any]:
    data = _section_data(bundle, "defensive-shape")
    data = data if isinstance(data, Mapping) else {}

    def half(key: str) -> list[dict[str, Any]]:
        rows = []
        for team in _teams(bundle):
            payload = data.get(team, {}) or {}
            profile = (
                payload.get(key, {})
                if isinstance(payload, Mapping)
                else {}
            )
            profile = profile if isinstance(profile, Mapping) else {}
            rows.append(
                {
                    "team": team,
                    "action_count": _safe_number(profile.get("action_count")),
                    "block_height_m": _safe_number(profile.get("block_height_m")),
                    "width_m": _safe_number(profile.get("width_m")),
                    "compactness_m": _safe_number(profile.get("compactness_m")),
                    "density_peak_pct": _safe_number(profile.get("density_peak_pct")),
                }
            )
        return rows

    return {
        "first_half": half("first_half"),
        "second_half": half("second_half"),
    }


def _ppda(bundle) -> dict[str, Any]:
    data = _section_data(bundle, "ppda")
    teams_data = data.get("teams", {}) if isinstance(data, Mapping) else {}
    teams_data = teams_data if isinstance(teams_data, Mapping) else {}

    rows = []
    for team in _teams(bundle):
        profile = teams_data.get(team, {}) or {}
        profile = profile if isinstance(profile, Mapping) else {}
        rows.append(
            {
                "team": team,
                "overall": _compact_mapping(
                    profile.get("overall", {})
                    if isinstance(profile.get("overall"), Mapping)
                    else {}
                ),
                "first_half": _compact_mapping(
                    profile.get("first_half", {})
                    if isinstance(profile.get("first_half"), Mapping)
                    else {}
                ),
                "second_half": _compact_mapping(
                    profile.get("second_half", {})
                    if isinstance(profile.get("second_half"), Mapping)
                    else {}
                ),
            }
        )
    return {"teams": rows}


def _category_counts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping):
        return []
    return [
        {"category": str(category), "count": _safe_number(count)}
        for category, count in value.items()
    ]


def _transition_section(bundle, section_id: str) -> dict[str, Any]:
    data = _section_data(bundle, section_id)
    data = data if isinstance(data, Mapping) else {}
    rows = []

    for team in _teams(bundle):
        payload = data.get(team, {}) or {}
        payload = payload if isinstance(payload, Mapping) else {}
        stats = payload.get("stats", {}) or {}
        stats = stats if isinstance(stats, Mapping) else {}
        kpis = payload.get("kpis", {}) or {}
        kpis = kpis if isinstance(kpis, Mapping) else {}

        rows.append(
            {
                "team": team,
                "total": _safe_number(stats.get("total")),
                "kpis": _compact_mapping(kpis),
                "outcomes": _category_counts(stats.get("outcomes")),
                "terminal_outcomes": _category_counts(
                    stats.get("terminal_outcomes")
                ),
                "channels": _category_counts(stats.get("flanks")),
                "types": _category_counts(stats.get("types")),
            }
        )

    return {"teams": rows}


def _restarts(bundle) -> dict[str, Any]:
    data = _section_data(bundle, "restarts")
    data = data if isinstance(data, Mapping) else {}
    rows = []

    for team in _teams(bundle):
        payload = data.get(team, {}) or {}
        payload = payload if isinstance(payload, Mapping) else {}
        records = _as_frame(payload.get("records"))

        player_col = _column(
            records,
            "player_name",
            "playerName",
            "Player",
        )
        restart_col = _column(
            records,
            "restart_type",
            "Action Type",
        )
        development_col = _column(
            records,
            "development_outcome",
            "Development Outcome",
        )

        top_takers = []
        if player_col is not None and not records.empty:
            grouped = []
            for player, group in records.groupby(
                player_col,
                dropna=True,
                sort=False,
            ):
                primary = None
                if restart_col is not None:
                    counts = (
                        group[restart_col]
                        .dropna()
                        .astype(str)
                        .value_counts()
                    )
                    primary = counts.index[0] if not counts.empty else None
                shots = None
                if development_col is not None:
                    shots = int(
                        group[development_col]
                        .fillna("")
                        .astype(str)
                        .str.contains("shot", case=False, regex=False)
                        .sum()
                    )
                grouped.append(
                    {
                        "player": str(player),
                        "restart_count": int(len(group)),
                        "primary_restart": _native(primary),
                        "shots": shots,
                    }
                )
            grouped.sort(
                key=lambda row: (
                    -int(row["restart_count"]),
                    row["player"].casefold(),
                )
            )
            top_takers = grouped[:MAX_RANKING_ROWS]

        rows.append(
            {
                "team": team,
                "summary": _compact_mapping(
                    payload.get("summary", {})
                    if isinstance(payload.get("summary"), Mapping)
                    else {}
                ),
                "top_takers": top_takers,
            }
        )

    return {"teams": rows}


def _player_team_lookup(data: Mapping[str, Any]) -> dict[str, str]:
    lookup = {}
    options = data.get("player_options", {}) if isinstance(data, Mapping) else {}
    if not isinstance(options, Mapping):
        return lookup

    for team, families in options.items():
        if not isinstance(families, Mapping):
            continue
        for values in families.values():
            if not isinstance(values, Sequence) or isinstance(
                values, (str, bytes, bytearray)
            ):
                continue
            for item in values:
                if isinstance(item, Mapping):
                    player = (
                        item.get("player_name")
                        or item.get("playerName")
                    )
                else:
                    player = item
                if player:
                    lookup[str(player)] = str(team)
    return lookup


def _canonical_player_frame(
    source: Any,
    *,
    team_lookup: Mapping[str, str],
) -> pd.DataFrame:
    frame = _as_frame(source)
    if frame.empty:
        return frame

    player_col = _column(
        frame,
        "playerName",
        "player_name",
        "Player",
        "player",
        "item",
    )
    if player_col is None:
        return frame

    if player_col != "playerName":
        frame = frame.rename(columns={player_col: "playerName"})

    team_col = _column(
        frame,
        "team_name",
        "teamName",
        "Team",
        "team",
    )
    inferred = (
        frame["playerName"].astype(str).map(
            {str(k): str(v) for k, v in team_lookup.items()}
        )
    )
    if team_col is None:
        frame["team_name"] = inferred
    else:
        if team_col != "team_name":
            frame = frame.rename(columns={team_col: "team_name"})
        missing = (
            frame["team_name"].isna()
            | frame["team_name"].astype(str).str.strip().eq("")
        )
        frame["team_name"] = frame["team_name"].where(
            ~missing,
            inferred,
        )

    return frame


_PLAYER_FIELDS = {
    "passing": (
        ("offensive_pass_contributions", ("Offensive Pass Contributions", "offensive_pass_contributions")),
        ("progressive_passes", ("Progressive Passes", "progressive_passes")),
        ("passes_into_box", ("Passes into Box", "passes_into_box")),
        ("key_passes", ("Key Passes", "key_passes")),
        ("assists", ("Assists", "assists")),
    ),
    "shooting": (
        ("shots", ("Shot Sequence Shots", "Shots", "shots")),
        ("shot_assists", ("Shot Sequence Shot Assists", "Shot Sequence Assists", "Shot Assists", "shot_assists")),
        ("pre_assists", ("Shot Sequence Pre-Assists", "Pre-Assists", "pre_assists")),
        ("involvements", ("Shot Sequence Involvements", "shot_sequence_involvements")),
    ),
    "defending": (
        ("unique_contributions", ("unique", "Unique Defensive Contributions", "unique_defensive_contributions")),
        ("tackles_won", ("tackles_won", "Tackles Won")),
        ("interceptions", ("interceptions", "Interceptions")),
        ("recoveries", ("recoveries", "Recoveries")),
        ("clearances", ("clearances", "Clearances")),
        ("blocks", ("blocks", "Blocks")),
    ),
}


def _top_players(bundle) -> dict[str, Any]:
    data = _section_data(bundle, "player-highlights")
    data = data if isinstance(data, Mapping) else {}
    team_lookup = _player_team_lookup(data)

    source_by_family = {
        "passing": data.get("player_stats"),
        "shooting": data.get("shot_sequence_ranking"),
        "defending": data.get("defensive_ranking"),
    }

    result = {}
    for family, source in source_by_family.items():
        frame = _canonical_player_frame(
            source,
            team_lookup=team_lookup,
        )
        family_rows = []

        for team in _teams(bundle):
            ranked = (
                rank_players_by_metric_family(
                    frame,
                    team,
                    category=family,
                    limit=MAX_RANKING_ROWS,
                )
                if not frame.empty
                else pd.DataFrame()
            )

            ranking = []
            for _, row in ranked.iterrows():
                item = {
                    "player": _row_value(
                        row,
                        ranked,
                        "playerName",
                        "player_name",
                        "Player",
                        "item",
                    ),
                }
                for target, aliases in _PLAYER_FIELDS[family]:
                    item[target] = _safe_number(
                        _row_value(row, ranked, *aliases)
                    )
                ranking.append(item)

            family_rows.append(
                {
                    "team": team,
                    "ranking": ranking,
                }
            )

        result[family] = family_rows

    return result


_SEQUENCE_IDS = (
    "sequence_id",
    "trigger_sequence_id",
    "buildup_sequence_id",
    "loss_sequence_id",
    "restart_id",
)
_EVENT_ID_ALIASES = (
    "id",
    "optaEventId",
    "opta_event_id",
    "eventId",
    "event_id",
)


def _sequence_id_column(frame: pd.DataFrame) -> Any | None:
    return _column(frame, *_SEQUENCE_IDS)


def _frames_from_sequence_source(source: Any):
    if isinstance(source, pd.DataFrame):
        yield source.copy()
        return

    if isinstance(source, Mapping):
        direct = _as_frame(source)
        if not direct.empty:
            yield direct
        for key in ("events", "rows", "sequence", "data"):
            nested = source.get(key)
            if nested is not None:
                yield from _frames_from_sequence_source(nested)
        return

    if isinstance(source, Sequence) and not isinstance(
        source, (str, bytes, bytearray)
    ):
        if source and all(isinstance(item, Mapping) for item in source):
            combined = _as_frame(source)
            if (
                not combined.empty
                and _sequence_id_column(combined) is not None
            ):
                yield combined
                return
        for item in source:
            yield from _frames_from_sequence_source(item)


def _find_sequence_events(source: Any, selected_id: Any) -> pd.DataFrame:
    selected_text = str(selected_id)
    for frame in _frames_from_sequence_source(source):
        if frame.empty:
            continue
        id_col = _sequence_id_column(frame)
        if id_col is None:
            continue
        mask = frame[id_col].astype(str).eq(selected_text)
        if mask.any():
            return frame.loc[mask].copy()
    return pd.DataFrame()


def _representative_source(
    bundle,
    *,
    section_id: str,
    team: str,
) -> Any:
    data = _section_data(bundle, section_id)
    if not isinstance(data, Mapping):
        return None

    if section_id == "build-up":
        teams_data = data.get("teams", {}) or {}
        payload = (
            teams_data.get(team, {})
            if isinstance(teams_data, Mapping)
            else {}
        )
    else:
        payload = data.get(team, {})

    if not isinstance(payload, Mapping):
        return None
    return payload.get("sequences")


def _representative_receiver_lookup(bundle) -> dict[str, Any]:
    """Reuse canonical receiver resolution without exporting bulk pass rows."""

    data = _section_data(bundle, "pass-network")
    passes = (
        _as_frame(data.get("passes"))
        if isinstance(data, Mapping)
        else pd.DataFrame()
    )
    if passes.empty:
        return {}

    id_col = _column(passes, *_EVENT_ID_ALIASES)
    receiver_col = _column(
        passes,
        "receiver",
        "receiver_name",
        "pass_receiver",
    )
    reliable_col = _column(
        passes,
        "receiver_is_reliable",
    )
    if id_col is None or receiver_col is None:
        return {}

    lookup = {}
    for _, row in passes.iterrows():
        identifier = _native(row.get(id_col))
        receiver = _native(row.get(receiver_col))
        if identifier in (None, "") or receiver in (None, ""):
            continue

        if reliable_col is not None:
            reliable = row.get(reliable_col)
            try:
                if pd.notna(reliable) and not bool(reliable):
                    continue
            except (TypeError, ValueError):
                pass

        lookup[str(identifier)] = receiver

    return lookup


def _event_projection(
    row: pd.Series,
    frame: pd.DataFrame,
    *,
    receiver_lookup: Mapping[str, Any],
) -> dict[str, Any]:
    def value(*aliases: str) -> Any:
        return _row_value(row, frame, *aliases)

    event_id = value(*_EVENT_ID_ALIASES)
    direct_receiver = value(
        "receiver",
        "receiver_name",
        "pass_receiver",
    )
    receiver = (
        direct_receiver
        if direct_receiver not in (None, "")
        else receiver_lookup.get(str(event_id))
        if event_id not in (None, "")
        else None
    )

    return {
        "event_id": event_id,
        "period": _safe_number(value("periodId", "period_id", "period")),
        "minute": _safe_number(value("timeMin", "minute")),
        "second": _safe_number(value("timeSec", "second")),
        "team": value("team_name", "teamName", "team"),
        "player": value("playerName", "player_name", "player"),
        "event_type": value("type_name", "event_type", "type"),
        "outcome": value("outcome", "Outcome"),
        "x": _safe_number(value("x")),
        "y": _safe_number(value("y")),
        "end_x": _safe_number(value("end_x", "endX")),
        "end_y": _safe_number(value("end_y", "endY")),
        "receiver": _native(receiver),
    }


def _representative_sequences(bundle, catalog) -> dict[str, Any]:
    figure_map = {
        "build-up-top-sequence": ("buildup", "build-up"),
        "defensive-transitions-top-sequence": (
            "defensive_transition",
            "defensive-transitions",
        ),
        "offensive-transitions-top-sequence": (
            "offensive_transition",
            "offensive-transitions",
        ),
        "restart-top-sequence": ("restart", "restarts"),
    }

    sequences = []
    event_pool = {}
    receiver_lookup = _representative_receiver_lookup(bundle)

    for artifact in tuple(getattr(catalog, "figures", ()) or ()):
        figure_id = str(getattr(artifact, "id", "") or "")
        if figure_id not in figure_map:
            continue

        selection = getattr(artifact, "selection", None) or {}
        if not isinstance(selection, Mapping):
            continue

        selected_id = selection.get("selected_id")
        team = getattr(artifact, "team_name", None)
        if selected_id in (None, "") or team in (None, ""):
            continue

        module, section_id = figure_map[figure_id]
        frame = _find_sequence_events(
            _representative_source(
                bundle,
                section_id=section_id,
                team=str(team),
            ),
            selected_id,
        )

        full_count = int(len(frame))
        selected_frame = frame.head(
            MAX_REPRESENTATIVE_SEQUENCE_EVENTS
        )
        event_keys = []

        for position, (_, row) in enumerate(
            selected_frame.iterrows()
        ):
            projected = _event_projection(
                row,
                selected_frame,
                receiver_lookup=receiver_lookup,
            )
            event_id = projected.get("event_id")
            if event_id not in (None, ""):
                event_key = str(event_id)
            else:
                event_key = (
                    f"{module}:{team}:{selected_id}:{position}"
                )

            if event_key not in event_pool:
                event_pool[event_key] = {
                    "event_key": event_key,
                    **projected,
                }
            event_keys.append(event_key)

        sequences.append(
            {
                "module": module,
                "team": str(team),
                "sequence_id": str(selected_id),
                "selection_reason": _native(
                    getattr(artifact, "selection_reason", None)
                    or selection.get("selection_reason")
                ),
                "criteria": _native(selection.get("criteria", [])),
                "event_count": full_count,
                "events_included": len(event_keys),
                "truncated": (
                    full_count
                    > MAX_REPRESENTATIVE_SEQUENCE_EVENTS
                ),
                "event_ids": event_keys,
            }
        )

    return {
        "sequences": sequences,
        "events": list(event_pool.values()),
    }


def _generation_status(
    bundle,
    generation_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    generation = (
        generation_manifest.get("generation", {})
        if isinstance(generation_manifest, Mapping)
        else {}
    )
    generation = generation if isinstance(generation, Mapping) else {}

    section_statuses = generation.get("section_statuses")
    if not isinstance(section_statuses, Mapping):
        section_statuses = {
            str(getattr(section, "id", "")): str(
                _status_value(section) or "missing"
            )
            for section in tuple(
                getattr(bundle, "sections", ()) or ()
            )
        }

    return {
        "status": _native(generation.get("status")),
        "pack_schema_version": _native(
            generation.get("pack_schema_version")
        ),
        "analysis_summary_schema_version": (
            ANALYSIS_SUMMARY_SCHEMA_VERSION
        ),
        "figures_expected": _safe_number(
            generation.get("figures_expected")
        ),
        "figures_generated": _safe_number(
            generation.get("figures_generated")
        ),
        "figures_empty": _safe_number(
            generation.get("figures_empty")
        ),
        "figures_skipped": _safe_number(
            generation.get("figures_skipped")
        ),
        "figures_failed": _safe_number(
            generation.get("figures_failed")
        ),
        "required_figures_failed": _safe_number(
            generation.get("required_figures_failed")
        ),
        "section_statuses": {
            str(key): str(value)
            for key, value in section_statuses.items()
        },
    }


def build_analysis_summary(
    bundle,
    catalog,
    generation_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    summary = {
        "schema_version": ANALYSIS_SUMMARY_SCHEMA_VERSION,
        "match": _match_summary(bundle),
        "score_and_goals": _score_and_goals(bundle),
        "data_quality": _data_quality(bundle),
        "team_comparison": _team_comparison(bundle),
        "formations": _formations(bundle),
        "passing": _passing(bundle),
        "progression": _progression(bundle),
        "final_third_entries": _final_third_entries(bundle),
        "crosses": _crosses(bundle),
        "buildup": _buildup(bundle),
        "defensive_shape": _defensive_shape(bundle),
        "ppda": _ppda(bundle),
        "defensive_transitions": _transition_section(
            bundle, "defensive-transitions"
        ),
        "offensive_transitions": _transition_section(
            bundle, "offensive-transitions"
        ),
        "restarts": _restarts(bundle),
        "top_players": _top_players(bundle),
        "representative_sequences": _representative_sequences(
            bundle, catalog
        ),
        "generation_status": _generation_status(
            bundle, generation_manifest
        ),
    }
    return normalize_for_json(summary)
