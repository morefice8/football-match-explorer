"""Lean machine-readable data exports for Match Analysis Packs.

REPORT-20 keeps one canonical event table in the standard pack and exports
specialized CSVs only at aggregate / sequence / ranking granularity. Complete
bundle dumps remain available only through the pack builder's debug mode.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


EVENTS_CORE_PATH = "tables/events-core.csv"

EVENTS_CORE_COLUMNS: tuple[str, ...] = (
    "id",
    "eventId",
    "periodId",
    "timestamp",
    "minute",
    "second",
    "team",
    "player",
    "type",
    "outcome",
    "x",
    "y",
    "end_x",
    "end_y",
    "receiver",
    "receiver_confidence",
    "receiver_is_reliable",
    "buildup_sequence_id",
    "defensive_transition_sequence_id",
    "offensive_transition_sequence_id",
    "restart_sequence_id",
    "is_key_pass",
    "is_assist",
    "is_progressive_attempt",
    "is_progressive",
    "is_into_box",
    "is_cross",
    "is_shot",
    "is_defensive_action",
    "is_restart",
)

CSV_EXPORT_DOCUMENTATION: dict[str, dict[str, Any]] = {
    "tables/match-comparison.csv": {
        "granularity": "one row per team",
        "description": "Compact Home-Away game profile used by the report overview.",
        "debug_only": False,
    },
    "tables/data-coverage.csv": {
        "granularity": "one row per data-quality metric",
        "description": "Coverage and data-quality diagnostics.",
        "debug_only": False,
    },
    "tables/pass-network-connections.csv": {
        "granularity": "one row per aggregated player connection",
        "description": "Aggregated pass-network connections; no raw event rows.",
        "debug_only": False,
    },
    "tables/progressive-passers.csv": {
        "granularity": "one row per player ranking entry",
        "description": "Progressive-pass player ranking.",
        "debug_only": False,
    },
    "tables/final-third-entries.csv": {
        "granularity": "one row per team",
        "description": "Final-third entry aggregate counts and channels; event rows live in events-core.csv.",
        "debug_only": False,
    },
    "tables/cross-routes.csv": {
        "granularity": "one row per aggregated cross route",
        "description": "Origin-to-destination cross-route aggregates.",
        "debug_only": False,
    },
    "tables/buildup-summary.csv": {
        "granularity": "one row per build-up sequence",
        "description": "Canonical build-up sequence summaries; no raw event rows.",
        "debug_only": False,
    },
    "tables/defensive-transitions.csv": {
        "granularity": "aggregate transition profile rows",
        "description": "Defensive-transition KPIs, zone/channel profile and outcome distributions.",
        "debug_only": False,
    },
    "tables/offensive-transitions.csv": {
        "granularity": "aggregate transition profile rows",
        "description": "Offensive-transition KPIs, zone/channel profile and outcome distributions.",
        "debug_only": False,
    },
    "tables/restarts.csv": {
        "granularity": "one row per restart sequence",
        "description": "Restart sequence records and outcomes.",
        "debug_only": False,
    },
    "tables/player-rankings.csv": {
        "granularity": "one row per player ranking entry",
        "description": "Passing, shooting and defending player rankings.",
        "debug_only": False,
    },
    EVENTS_CORE_PATH: {
        "granularity": "one row per canonical Opta event",
        "description": "Lean canonical event table with coordinates, receiver resolution, sequence ids and principal analytical flags.",
        "debug_only": False,
    },
    "tables/event-explorer.csv": {
        "granularity": "wide diagnostic event union",
        "description": "Legacy wide event explorer retained for debugging only.",
        "debug_only": True,
    },
}


def csv_export_documentation(paths: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        metadata = CSV_EXPORT_DOCUMENTATION.get(
            path,
            {
                "granularity": "documented export",
                "description": path,
                "debug_only": False,
            },
        )
        rows.append({"path": path, **metadata})
    return rows


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


def _as_frame(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        return value.to_frame().T.reset_index(drop=True)
    if isinstance(value, Mapping):
        return pd.DataFrame([dict(value)])
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        if not value:
            return pd.DataFrame()
        if all(isinstance(item, Mapping) for item in value):
            return pd.DataFrame([dict(item) for item in value])
    return pd.DataFrame()


def _frames(value: Any):
    """Yield DataFrames from nested sequence containers without flattening cells."""

    if value is None:
        return
    if isinstance(value, (pd.DataFrame, pd.Series, Mapping)):
        frame = _as_frame(value)
        if not frame.empty:
            yield frame
        return
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        if value and all(isinstance(item, Mapping) for item in value):
            frame = _as_frame(value)
            if not frame.empty:
                yield frame
            return
        for item in value:
            yield from _frames(item)


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


def _event_key(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    return text or None


def _series(
    frame: pd.DataFrame,
    *aliases: str,
    default: Any = None,
) -> pd.Series:
    column = _column(frame, *aliases)
    if column is None:
        return pd.Series(
            [default] * len(frame),
            index=frame.index,
            dtype="object",
        )
    return frame[column].copy()


def _truthy(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series([], index=series.index, dtype=bool)

    numeric = pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    text = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        .isin(
            {
                "true",
                "yes",
                "y",
                "successful",
                "completed",
                "assist",
                "key pass",
            }
        )
    )
    return (numeric | text).astype(bool)


def _source_events(bundle) -> pd.DataFrame:
    progressive = _section_data(bundle, "progressive-passes")
    if isinstance(progressive, Mapping):
        classified = _as_frame(progressive.get("classified_passes"))
        if not classified.empty:
            return classified

    # Backward-compatible fallback for synthetic/older bundles.
    frames: list[pd.DataFrame] = []

    passes = _section_data(bundle, "pass-locations")
    if isinstance(passes, Mapping):
        frame = _as_frame(passes.get("passes"))
        if not frame.empty:
            frames.append(frame)

    overview = _section_data(bundle, "overview")
    if isinstance(overview, Mapping):
        frame = _as_frame(overview.get("shots"))
        if not frame.empty:
            frames.append(frame)

    defensive = _section_data(bundle, "defensive-shape")
    if isinstance(defensive, Mapping):
        for team_payload in defensive.values():
            if not isinstance(team_payload, Mapping):
                continue
            for half in ("first_half", "second_half"):
                profile = team_payload.get(half, {})
                if not isinstance(profile, Mapping):
                    continue
                frame = _as_frame(profile.get("actions"))
                if not frame.empty:
                    frames.append(frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    id_column = _column(combined, "id")
    if id_column is not None:
        combined = combined.drop_duplicates(
            subset=[id_column],
            keep="first",
        )
    return combined


def _lookup_by_event(
    frame: pd.DataFrame,
    value_column: Any | None,
) -> dict[str, Any]:
    id_column = _column(frame, "id")
    if frame.empty or id_column is None or value_column is None:
        return {}

    lookup: dict[str, Any] = {}
    for _, row in frame[[id_column, value_column]].iterrows():
        key = _event_key(row[id_column])
        if key is None or key in lookup:
            continue
        value = row[value_column]
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        lookup[key] = value
    return lookup


def _sequence_assignments(
    source: Any,
    *,
    sequence_aliases: tuple[str, ...],
) -> dict[str, str]:
    by_event: dict[str, list[str]] = {}

    for frame in _frames(source):
        id_column = _column(frame, "id")
        sequence_column = _column(frame, *sequence_aliases)
        if id_column is None or sequence_column is None:
            continue

        for _, row in frame[[id_column, sequence_column]].iterrows():
            event_key = _event_key(row[id_column])
            if event_key is None:
                continue
            value = row[sequence_column]
            try:
                if pd.isna(value):
                    continue
            except (TypeError, ValueError):
                pass
            sequence_id = str(value).strip()
            if not sequence_id:
                continue
            values = by_event.setdefault(event_key, [])
            if sequence_id not in values:
                values.append(sequence_id)

    return {
        event_key: "|".join(sequence_ids)
        for event_key, sequence_ids in by_event.items()
    }


def _buildup_assignments(bundle) -> dict[str, str]:
    data = _section_data(bundle, "build-up")
    teams = data.get("teams", {}) if isinstance(data, Mapping) else {}
    combined: dict[str, list[str]] = {}

    if isinstance(teams, Mapping):
        for payload in teams.values():
            if not isinstance(payload, Mapping):
                continue
            mapping = _sequence_assignments(
                payload.get("sequences"),
                sequence_aliases=(
                    "trigger_sequence_id",
                    "buildup_sequence_id",
                    "sequence_id",
                ),
            )
            for key, value in mapping.items():
                values = combined.setdefault(key, [])
                for item in value.split("|"):
                    if item and item not in values:
                        values.append(item)

    return {
        key: "|".join(values)
        for key, values in combined.items()
    }


def _transition_assignments(
    bundle,
    section_id: str,
) -> dict[str, str]:
    data = _section_data(bundle, section_id)
    combined: dict[str, list[str]] = {}

    if isinstance(data, Mapping):
        for payload in data.values():
            if not isinstance(payload, Mapping):
                continue
            mapping = _sequence_assignments(
                payload.get("combined"),
                sequence_aliases=("loss_sequence_id", "sequence_id"),
            )
            for key, value in mapping.items():
                values = combined.setdefault(key, [])
                for item in value.split("|"):
                    if item and item not in values:
                        values.append(item)

    return {
        key: "|".join(values)
        for key, values in combined.items()
    }


def _restart_assignments(bundle) -> dict[str, str]:
    data = _section_data(bundle, "restarts")
    combined: dict[str, list[str]] = {}

    if isinstance(data, Mapping):
        for payload in data.values():
            if not isinstance(payload, Mapping):
                continue
            mapping = _sequence_assignments(
                payload.get("sequences"),
                sequence_aliases=(
                    "trigger_sequence_id",
                    "restart_id",
                    "sequence_id",
                ),
            )
            for key, value in mapping.items():
                values = combined.setdefault(key, [])
                for item in value.split("|"):
                    if item and item not in values:
                        values.append(item)

    return {
        key: "|".join(values)
        for key, values in combined.items()
    }


def _event_id_set(value: Any) -> set[str]:
    ids: set[str] = set()
    for frame in _frames(value):
        id_column = _column(frame, "id")
        if id_column is None:
            continue
        for raw in frame[id_column].tolist():
            key = _event_key(raw)
            if key is not None:
                ids.add(key)
    return ids


def _defensive_action_ids(bundle) -> set[str]:
    data = _section_data(bundle, "defensive-shape")
    ids: set[str] = set()

    if not isinstance(data, Mapping):
        return ids

    for payload in data.values():
        if not isinstance(payload, Mapping):
            continue
        for half in ("first_half", "second_half"):
            profile = payload.get(half, {})
            if not isinstance(profile, Mapping):
                continue
            ids |= _event_id_set(profile.get("actions"))

    return ids


def _restart_event_ids(bundle) -> set[str]:
    data = _section_data(bundle, "restarts")
    ids: set[str] = set()
    if isinstance(data, Mapping):
        for payload in data.values():
            if isinstance(payload, Mapping):
                ids |= _event_id_set(payload.get("sequences"))
    return ids


def build_events_core(bundle) -> pd.DataFrame:
    """Return one lean canonical row per Opta event.

    The canonical full-match event source is the already-derived
    ``progressive-passes.classified_passes`` frame.  Other bundle sections are
    consulted only to enrich those canonical rows with receiver resolution,
    sequence membership and analytical flags.
    """

    source = _source_events(bundle)
    if source.empty:
        return pd.DataFrame(columns=list(EVENTS_CORE_COLUMNS))

    id_column = _column(source, "id")
    if id_column is None:
        return pd.DataFrame(columns=list(EVENTS_CORE_COLUMNS))

    source = source.loc[source[id_column].notna()].copy()
    source = source.drop_duplicates(subset=[id_column], keep="first")
    event_keys = source[id_column].map(_event_key)

    pass_network = _section_data(bundle, "pass-network")
    passes = (
        _as_frame(pass_network.get("passes"))
        if isinstance(pass_network, Mapping)
        else pd.DataFrame()
    )

    receiver_lookup = _lookup_by_event(
        passes,
        _column(passes, "receiver"),
    )
    receiver_confidence_lookup = _lookup_by_event(
        passes,
        _column(passes, "receiver_confidence"),
    )
    receiver_reliable_lookup = _lookup_by_event(
        passes,
        _column(passes, "receiver_is_reliable"),
    )
    into_box_lookup = _lookup_by_event(
        passes,
        _column(passes, "is_into_box"),
    )

    buildup = _buildup_assignments(bundle)
    defensive_transition = _transition_assignments(
        bundle,
        "defensive-transitions",
    )
    offensive_transition = _transition_assignments(
        bundle,
        "offensive-transitions",
    )
    restart = _restart_assignments(bundle)

    overview = _section_data(bundle, "overview")
    shot_ids = (
        _event_id_set(overview.get("shots"))
        if isinstance(overview, Mapping)
        else set()
    )
    defensive_action_ids = _defensive_action_ids(bundle)
    restart_ids = _restart_event_ids(bundle)

    receiver_direct = _series(
        source,
        "receiver",
        "receiver_name",
        "pass_receiver",
    )
    receiver = receiver_direct.copy()
    receiver_missing = receiver.isna() | receiver.astype(str).str.strip().eq("")
    receiver.loc[receiver_missing] = event_keys.loc[
        receiver_missing
    ].map(receiver_lookup)

    receiver_confidence = _series(
        source,
        "receiver_confidence",
    )
    confidence_missing = (
        receiver_confidence.isna()
        | receiver_confidence.astype(str).str.strip().eq("")
    )
    receiver_confidence.loc[confidence_missing] = event_keys.loc[
        confidence_missing
    ].map(receiver_confidence_lookup)

    receiver_reliable = _series(
        source,
        "receiver_is_reliable",
    )
    reliable_missing = receiver_reliable.isna()
    receiver_reliable.loc[reliable_missing] = event_keys.loc[
        reliable_missing
    ].map(receiver_reliable_lookup)

    into_box = _series(source, "is_into_box")
    into_box_missing = into_box.isna()
    into_box.loc[into_box_missing] = event_keys.loc[
        into_box_missing
    ].map(into_box_lookup)

    result = pd.DataFrame(index=source.index)
    result["id"] = _series(source, "id")
    result["eventId"] = _series(source, "eventId", "event_id")
    result["periodId"] = _series(source, "periodId", "period_id")
    result["timestamp"] = _series(
        source,
        "timeStamp",
        "timestamp",
    )
    result["minute"] = _series(source, "timeMin", "minute")
    result["second"] = _series(source, "timeSec", "second")
    result["team"] = _series(
        source,
        "team_name",
        "teamName",
        "team",
    )
    result["player"] = _series(
        source,
        "playerName",
        "player_name",
        "player",
    )
    result["type"] = _series(
        source,
        "type_name",
        "event_type",
        "type",
    )
    result["outcome"] = _series(source, "outcome", "Outcome")
    result["x"] = _series(source, "x")
    result["y"] = _series(source, "y")
    result["end_x"] = _series(source, "end_x", "endX")
    result["end_y"] = _series(source, "end_y", "endY")
    result["receiver"] = receiver
    result["receiver_confidence"] = receiver_confidence
    result["receiver_is_reliable"] = receiver_reliable
    result["buildup_sequence_id"] = event_keys.map(buildup)
    result["defensive_transition_sequence_id"] = (
        event_keys.map(defensive_transition)
    )
    result["offensive_transition_sequence_id"] = (
        event_keys.map(offensive_transition)
    )
    result["restart_sequence_id"] = event_keys.map(restart)
    result["is_key_pass"] = _truthy(
        _series(source, "is_key_pass")
    )
    result["is_assist"] = _truthy(
        _series(source, "is_assist")
    )
    result["is_progressive_attempt"] = _truthy(
        _series(source, "is_progressive_attempt")
    )
    result["is_progressive"] = _truthy(
        _series(source, "is_progressive")
    )
    result["is_into_box"] = _truthy(into_box)
    result["is_cross"] = _truthy(
        _series(source, "cross", "is_cross")
    )
    result["is_shot"] = event_keys.isin(shot_ids)
    result["is_defensive_action"] = event_keys.isin(
        defensive_action_ids
    )
    result["is_restart"] = (
        event_keys.isin(restart_ids)
        | result["restart_sequence_id"].notna()
    )

    return result.loc[:, list(EVENTS_CORE_COLUMNS)].reset_index(drop=True)


def build_final_third_summary(bundle) -> pd.DataFrame:
    data = _section_data(bundle, "final-third-entries")
    teams_data = (
        data.get("teams", {})
        if isinstance(data, Mapping)
        else {}
    )
    teams = tuple(getattr(bundle, "teams", ()) or ())
    rows: list[dict[str, Any]] = []

    for team in teams:
        payload = (
            teams_data.get(team, {})
            if isinstance(teams_data, Mapping)
            else {}
        )
        stats = (
            payload.get("stats", {})
            if isinstance(payload, Mapping)
            else {}
        )
        row = {
            "report_team": team,
            "team_name": team,
        }
        if isinstance(stats, Mapping):
            row.update(dict(stats))
        rows.append(row)

    return pd.DataFrame(rows)


def _percentage(count: Any, total: Any) -> float:
    try:
        count_value = float(count)
        total_value = float(total)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(count_value) or not math.isfinite(total_value):
        return 0.0
    return count_value / total_value * 100.0 if total_value else 0.0


def build_transition_summary(
    bundle,
    section_id: str,
) -> pd.DataFrame:
    """Return compact canonical transition profiles, never event rows."""

    data = _section_data(bundle, section_id)
    teams = tuple(getattr(bundle, "teams", ()) or ())
    rows: list[dict[str, Any]] = []

    if not isinstance(data, Mapping):
        data = {}

    for team in teams:
        payload = data.get(team, {})
        payload = payload if isinstance(payload, Mapping) else {}
        stats = payload.get("stats", {}) or {}
        stats = stats if isinstance(stats, Mapping) else {}
        kpis = payload.get("kpis", {}) or {}
        kpis = kpis if isinstance(kpis, Mapping) else {}

        total = stats.get("total", kpis.get("transition_count", 0)) or 0

        rows.append(
            {
                "report_team": team,
                "team_name": team,
                "dimension": "summary",
                "category": "all",
                "count": total,
                "percentage": 100.0 if total else 0.0,
                "avg_duration_s": None,
                "avg_passes": None,
                "median_duration_s": kpis.get(
                    "median_duration_seconds"
                ),
                "final_third_pct": kpis.get("final_third_pct"),
                "penalty_area_pct": kpis.get("penalty_area_pct"),
                "shot_pct": kpis.get("shot_pct"),
            }
        )

        profile = _as_frame(stats.get("transition_profile_table"))
        if not profile.empty:
            zone_column = _column(
                profile,
                "Loss Zone",
                "Recovery Zone",
                "Start Zone",
            )
            channel_column = _column(
                profile,
                "Counterattack Side",
                "Attack Side",
                "Channel",
            )
            count_column = _column(
                profile,
                "Num_Sequences",
                "Transitions",
                "count",
            )
            duration_column = _column(
                profile,
                "Avg Duration (s)",
                "avg_duration_s",
            )
            passes_column = _column(
                profile,
                "Avg Passes",
                "avg_passes",
            )

            for _, profile_row in profile.iterrows():
                zone = (
                    profile_row.get(zone_column)
                    if zone_column is not None
                    else None
                )
                channel = (
                    profile_row.get(channel_column)
                    if channel_column is not None
                    else None
                )
                count = (
                    profile_row.get(count_column)
                    if count_column is not None
                    else None
                )
                category = " | ".join(
                    str(value)
                    for value in (zone, channel)
                    if value not in (None, "")
                )
                rows.append(
                    {
                        "report_team": team,
                        "team_name": team,
                        "dimension": "start_zone_channel",
                        "category": category or "unknown",
                        "count": count,
                        "percentage": _percentage(count, total),
                        "avg_duration_s": (
                            profile_row.get(duration_column)
                            if duration_column is not None
                            else None
                        ),
                        "avg_passes": (
                            profile_row.get(passes_column)
                            if passes_column is not None
                            else None
                        ),
                        "median_duration_s": None,
                        "final_third_pct": None,
                        "penalty_area_pct": None,
                        "shot_pct": None,
                    }
                )

        for dimension, key in (
            ("outcome", "outcomes"),
            ("terminal_outcome", "terminal_outcomes"),
            ("channel", "flanks"),
            ("trigger_type", "types"),
        ):
            values = stats.get(key, {})
            if not isinstance(values, Mapping):
                continue
            for category, count in values.items():
                rows.append(
                    {
                        "report_team": team,
                        "team_name": team,
                        "dimension": dimension,
                        "category": category,
                        "count": count,
                        "percentage": _percentage(count, total),
                        "avg_duration_s": None,
                        "avg_passes": None,
                        "median_duration_s": None,
                        "final_third_pct": None,
                        "penalty_area_pct": None,
                        "shot_pct": None,
                    }
                )

    columns = (
        "report_team",
        "team_name",
        "dimension",
        "category",
        "count",
        "percentage",
        "avg_duration_s",
        "avg_passes",
        "median_duration_s",
        "final_third_pct",
        "penalty_area_pct",
        "shot_pct",
    )
    return pd.DataFrame(rows, columns=list(columns))
