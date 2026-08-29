
"""Pure Match Report data-bundle orchestration.

This module is intentionally independent from Dash, app.py and report rendering.
It coordinates canonical metric functions and returns neutral Python/pandas data.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from src.reporting.manifest import REPORT_MANIFEST
from src.reporting.models import ReportManifest, ReportScope
from src.utils.derived_cache import cache_derived_result, dataframe_signature


class ReportSectionStatus(str, Enum):
    GENERATED = "generated"
    EMPTY = "empty"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass(frozen=True)
class MatchReportBundleConfig:
    """Configuration for a Full Match, two-team report bundle."""

    manifest: ReportManifest = REPORT_MANIFEST
    scope: ReportScope = ReportScope.FULL_MATCH
    teams: tuple[str, str] | None = None
    enabled_sections: tuple[str, ...] | None = None
    progressive_pass_exclusions: tuple[str, ...] = ()
    buildup_triggers: tuple[str, ...] | None = None
    top_players: int = 10
    section_options: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.scope is not ReportScope.FULL_MATCH:
            raise ValueError("REPORT-02 currently supports Full Match only.")
        if self.teams is not None and len(self.teams) != 2:
            raise ValueError("Report bundle requires exactly two teams.")
        if self.top_players <= 0:
            raise ValueError("top_players must be positive.")

        manifest_ids = {section.id for section in self.manifest.sections}
        if self.enabled_sections is not None:
            unknown = set(self.enabled_sections) - manifest_ids
            if unknown:
                raise ValueError(
                    "Unknown report sections: " + ", ".join(sorted(unknown))
                )

    @classmethod
    def from_value(
        cls,
        value: "MatchReportBundleConfig | Mapping[str, Any] | None",
    ) -> "MatchReportBundleConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(
                "report config must be MatchReportBundleConfig, mapping or None"
            )

        payload = dict(value)
        scope = payload.get("scope", ReportScope.FULL_MATCH)
        if not isinstance(scope, ReportScope):
            scope = ReportScope(str(scope))

        teams = payload.get("teams")
        if teams is not None:
            teams = tuple(str(team) for team in teams)

        enabled = payload.get("enabled_sections")
        if enabled is not None:
            enabled = tuple(str(item) for item in enabled)

        exclusions = tuple(
            str(item)
            for item in payload.get("progressive_pass_exclusions", ()) or ()
        )
        triggers = payload.get("buildup_triggers")
        if triggers is not None:
            triggers = tuple(str(item) for item in triggers)

        return cls(
            manifest=payload.get("manifest", REPORT_MANIFEST),
            scope=scope,
            teams=teams,
            enabled_sections=enabled,
            progressive_pass_exclusions=exclusions,
            buildup_triggers=triggers,
            top_players=int(payload.get("top_players", 10)),
            section_options=dict(payload.get("section_options", {}) or {}),
        )


@dataclass(frozen=True)
class ReportSectionBundle:
    id: str
    status: ReportSectionStatus
    data: Any = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class MatchReportDataBundle:
    """Neutral result of one report-bundle build."""

    manifest_id: str
    manifest_version: str
    source_signature: str
    scope: ReportScope
    teams: tuple[str, str]
    match_info: Mapping[str, Any]
    sections: tuple[ReportSectionBundle, ...]

    def section(self, section_id: str) -> ReportSectionBundle:
        for section in self.sections:
            if section.id == section_id:
                return section
        raise KeyError(section_id)

    def status_by_section(self) -> dict[str, str]:
        return {
            section.id: section.status.value
            for section in self.sections
        }

    def to_dict(self) -> dict[str, Any]:
        return normalize_for_json(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
        )


@dataclass(frozen=True)
class _Produced:
    data: Any
    empty: bool = False


@dataclass(frozen=True)
class _CachedFailure:
    exc: Exception


def _produced(data: Any, *, empty: bool = False) -> _Produced:
    return _Produced(data=data, empty=bool(empty))


def normalize_for_json(value: Any) -> Any:
    """Recursively convert bundle content to strict JSON-compatible values."""

    if isinstance(value, Enum):
        return value.value

    if is_dataclass(value):
        return {
            item.name: normalize_for_json(getattr(value, item.name))
            for item in fields(value)
        }

    if isinstance(value, pd.DataFrame):
        frame = value.copy()
        if (
            not isinstance(frame.index, pd.RangeIndex)
            or frame.index.name is not None
        ):
            index_name = frame.index.name or "index"
            if index_name in frame.columns:
                index_name = "_index"
            frame = frame.reset_index(names=index_name)
        frame = frame.astype(object).where(pd.notna(frame), None)
        return [
            normalize_for_json(record)
            for record in frame.to_dict(orient="records")
        ]

    if isinstance(value, pd.Series):
        series = value.copy()
        return {
            str(key): normalize_for_json(item)
            for key, item in series.to_dict().items()
        }

    if isinstance(value, np.generic):
        return normalize_for_json(value.item())

    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value

    if value is pd.NA:
        return None

    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(key): normalize_for_json(item)
            for key, item in value.items()
        }

    if isinstance(value, (tuple, list)):
        return [normalize_for_json(item) for item in value]

    if isinstance(value, (set, frozenset)):
        return sorted(
            (normalize_for_json(item) for item in value),
            key=lambda item: str(item),
        )

    if isinstance(value, Path):
        return str(value)

    return value


@cache_derived_result("report_bundle_passes")
def _cached_passes(df_processed: pd.DataFrame) -> pd.DataFrame:
    from src.data_processing import pass_processing

    result = pass_processing.get_passes_df(df_processed.copy())
    return result if result is not None else pd.DataFrame()


@cache_derived_result("report_bundle_carries")
def _cached_carries(df_processed: pd.DataFrame) -> pd.DataFrame:
    from src.data_processing import pass_processing

    result = pass_processing.infer_carries(df_processed.copy())
    return result if result is not None else pd.DataFrame()


@cache_derived_result("report_bundle_progressive")
def _cached_progressive(
    df_processed: pd.DataFrame,
    exclusions: tuple[str, ...],
) -> pd.DataFrame:
    from src.metrics import pass_metrics

    return pass_metrics.classify_progressive_passes(
        df_processed,
        exclude_qualifiers=list(exclusions),
    )


@cache_derived_result("report_bundle_player_stats")
def _cached_player_stats(
    df_processed: pd.DataFrame,
    exclusions: tuple[str, ...],
) -> pd.DataFrame:
    from src.metrics import player_metrics

    return player_metrics.calculate_player_stats(
        df_processed,
        prog_pass_exclusions=list(exclusions),
    )


@cache_derived_result("report_bundle_goal_origins")
def _cached_goal_origins(
    df_processed: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> list[dict[str, Any]]:
    from src.metrics import goal_origin_metrics

    return goal_origin_metrics.classify_goal_origins(
        df_processed,
        home_team=home_team,
        away_team=away_team,
    )


@cache_derived_result("report_bundle_formation_timeline")
def _cached_formation_timeline(
    df_processed: pd.DataFrame,
    match_info_json: str,
) -> dict[str, Any]:
    # The canonical neutral timeline model currently lives beside the Plotly
    # renderer. This function calls only the model builder: no figure is built.
    from src.visualization import formation_plotly

    return formation_plotly.build_formation_timeline_model(
        df_processed,
        json.loads(match_info_json),
    )


@cache_derived_result("report_bundle_shot_sequence_stats")
def _cached_shot_sequence_stats(
    df_processed: pd.DataFrame,
) -> pd.DataFrame:
    from src.metrics import shot_sequence_metrics

    return shot_sequence_metrics.calculate_shot_sequence_player_stats(
        df_processed
    )


class _BundleContext:
    def __init__(
        self,
        df_processed: pd.DataFrame,
        match_info: Mapping[str, Any],
        config: MatchReportBundleConfig,
        teams: tuple[str, str],
    ) -> None:
        self.df = df_processed
        self.match_info = dict(match_info)
        self.config = config
        self.teams = teams
        self._once_cache: dict[Any, Any] = {}

    @property
    def home_team(self) -> str:
        return self.teams[0]

    @property
    def away_team(self) -> str:
        return self.teams[1]

    def once(self, key: Any, factory: Callable[[], Any]) -> Any:
        if key not in self._once_cache:
            try:
                self._once_cache[key] = factory()
            except Exception as exc:
                self._once_cache[key] = _CachedFailure(exc)

        value = self._once_cache[key]
        if isinstance(value, _CachedFailure):
            raise value.exc
        return value

    def other_team(self, team_name: str) -> str:
        if team_name == self.home_team:
            return self.away_team
        if team_name == self.away_team:
            return self.home_team
        raise ValueError(f"Unknown report team: {team_name}")

    def passes(self) -> pd.DataFrame:
        return self.once("passes", lambda: _cached_passes(self.df))

    def carries(self) -> pd.DataFrame:
        return self.once("carries", lambda: _cached_carries(self.df))

    def progressive(self) -> pd.DataFrame:
        return self.once(
            "progressive",
            lambda: _cached_progressive(
                self.df,
                self.config.progressive_pass_exclusions,
            ),
        )

    def player_stats(self) -> pd.DataFrame:
        return self.once(
            "player-stats",
            lambda: _cached_player_stats(
                self.df,
                self.config.progressive_pass_exclusions,
            ),
        )

    def goal_origins(self) -> list[dict[str, Any]]:
        return self.once(
            "goal-origins",
            lambda: _cached_goal_origins(
                self.df,
                self.home_team,
                self.away_team,
            ),
        )

    def formation_timeline(self) -> dict[str, Any]:
        match_info_json = json.dumps(
            normalize_for_json(self.match_info),
            ensure_ascii=False,
            sort_keys=True,
        )
        return self.once(
            "formation-timeline",
            lambda: _cached_formation_timeline(
                self.df,
                match_info_json,
            ),
        )

    def final_third(self, team_name: str) -> tuple[pd.DataFrame, dict]:
        def build():
            from src.metrics import pass_metrics

            passes = self.passes()
            if (
                passes is not None
                and not passes.empty
                and "team_name" in passes.columns
                and "outcome" in passes.columns
            ):
                team_passes = passes[
                    passes["team_name"].eq(team_name)
                    & passes["outcome"].eq("Successful")
                ].copy()
            else:
                team_passes = pd.DataFrame()

            carries = self.carries()
            if (
                carries is not None
                and not carries.empty
                and "team_name" in carries.columns
            ):
                team_carries = carries[
                    carries["team_name"].eq(team_name)
                ].copy()
            else:
                team_carries = pd.DataFrame()

            return pass_metrics.analyze_final_third_entries(
                team_passes,
                team_carries,
            )

        return self.once(("final-third", team_name), build)

    def crosses(self, team_name: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
        def build():
            from src.metrics import cross_metrics

            analyzed = cross_metrics.analyze_crosses(
                self.df,
                team_name,
            )
            summary, routes = cross_metrics.build_cross_flow_profile(
                analyzed,
                limit=8,
            )
            return analyzed, summary, routes

        return self.once(("crosses", team_name), build)


def _resolve_teams(
    df_processed: pd.DataFrame,
    match_info: Mapping[str, Any],
    config: MatchReportBundleConfig,
) -> tuple[str, str]:
    if config.teams is not None:
        return tuple(config.teams)

    home = (
        match_info.get("hteamName")
        or match_info.get("home_team")
        or match_info.get("homeTeam")
    )
    away = (
        match_info.get("ateamName")
        or match_info.get("away_team")
        or match_info.get("awayTeam")
    )

    if home and away:
        return str(home), str(away)

    discovered: list[str] = []
    if (
        df_processed is not None
        and not df_processed.empty
        and "team_name" in df_processed.columns
    ):
        discovered = [
            str(value)
            for value in df_processed["team_name"].dropna().unique().tolist()
            if str(value).strip()
        ]

    if not home and discovered:
        home = discovered[0]
    if not away:
        away = next(
            (team for team in discovered if team != home),
            None,
        )

    return str(home or "Home"), str(away or "Away")


def _team_frame(frame: pd.DataFrame, team_name: str) -> pd.DataFrame:
    if (
        frame is None
        or frame.empty
        or "team_name" not in frame.columns
    ):
        return pd.DataFrame()
    return frame[frame["team_name"].eq(team_name)].copy()


def _successful_pass_mask(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=bool)
    outcome = frame.get(
        "outcome",
        pd.Series("", index=frame.index),
    )
    text = outcome.fillna("").astype(str).str.strip().str.lower()
    numeric = pd.to_numeric(outcome, errors="coerce")
    return text.eq("successful") | numeric.eq(1)


def _score_from_match_info(
    match_info: Mapping[str, Any],
    *,
    home: bool,
) -> Any:
    keys = (
        ("hteamScore", "home_score", "homeScore", "hScore")
        if home
        else ("ateamScore", "away_score", "awayScore", "aScore")
    )
    for key in keys:
        value = match_info.get(key)
        if value not in (None, ""):
            return value
    return None


def _first_present(
    record: Mapping[str, Any],
    *keys: str,
) -> Any:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _scorer_clock(
    record: Mapping[str, Any],
) -> tuple[Any, Any]:
    minute = _first_present(
        record,
        "minute",
        "timeMin",
    )
    second = _first_present(
        record,
        "second",
        "timeSec",
    )

    clock = _first_present(
        record,
        "timeMinSec",
        "display_time",
    )
    if clock is not None and ":" in str(clock):
        clock_minute, clock_second = str(clock).split(":", 1)
        if minute is None:
            minute = clock_minute
        if second is None:
            second = clock_second

    return minute, second


def _normalize_scorer_record(
    ctx: _BundleContext,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    team_position = _first_present(
        record,
        "team_position",
        "teamPosition",
    )
    team_name = _first_present(
        record,
        "team_name",
        "teamName",
    )

    if team_name is None:
        if team_position == "home":
            team_name = ctx.home_team
        elif team_position == "away":
            team_name = ctx.away_team

    if team_position is None:
        if team_name == ctx.home_team:
            team_position = "home"
        elif team_name == ctx.away_team:
            team_position = "away"

    minute, second = _scorer_clock(record)

    return {
        "team_name": team_name,
        "team_position": team_position,
        "scorer": (
            _first_present(
                record,
                "scorer",
                "scorerName",
                "playerName",
            )
            or "Unknown"
        ),
        "minute": minute,
        "second": second,
        "period_id": _first_present(
            record,
            "period_id",
            "periodId",
        ),
        "goal_type": _first_present(
            record,
            "goal_type",
            "goalType",
        ),
        "goal_event_id": _first_present(
            record,
            "goal_event_id",
            "optaEventId",
            "eventId",
            "id",
        ),
    }


def _goal_list(
    ctx: _BundleContext,
    goal_origins: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    supplied = ctx.match_info.get("goals")
    source = (
        supplied
        if isinstance(supplied, list) and supplied
        else goal_origins
    )

    return [
        _normalize_scorer_record(
            ctx,
            record,
        )
        for record in source
        if isinstance(record, Mapping)
    ]


def _overview(ctx: _BundleContext) -> _Produced:
    from src.metrics import data_quality, defensive_metrics
    from src.metrics import pass_metrics, shot_metrics

    goals = ctx.goal_origins()
    goal_list = _goal_list(ctx, goals)

    home_score = _score_from_match_info(ctx.match_info, home=True)
    away_score = _score_from_match_info(ctx.match_info, home=False)
    if home_score is None:
        home_score = sum(
            item.get("team_name") == ctx.home_team
            for item in goals
        )
    if away_score is None:
        away_score = sum(
            item.get("team_name") == ctx.away_team
            for item in goals
        )

    passes = ctx.passes()
    progressive = ctx.progressive()
    defensive_actions = ctx.once(
        "defensive-actions",
        lambda: defensive_metrics.get_defensive_actions(ctx.df),
    )

    hxg = ctx.match_info.get("hxG")
    axg = ctx.match_info.get("axG")
    hxgot = ctx.match_info.get("hxGOT")
    axgot = ctx.match_info.get("axGOT")
    shots_df, home_shot_stats, away_shot_stats = ctx.once(
        "shot-stats",
        lambda: shot_metrics.calculate_shot_stats(
            ctx.df,
            ctx.home_team,
            ctx.away_team,
            hxg,
            axg,
            hxgot,
            axgot,
        ),
    )

    game_profile = {}
    for team_name, shot_stats in (
        (ctx.home_team, home_shot_stats),
        (ctx.away_team, away_shot_stats),
    ):
        team_passes = _team_frame(passes, team_name)
        pass_total = int(len(team_passes))
        successful = (
            int(_successful_pass_mask(team_passes).sum())
            if pass_total
            else 0
        )

        team_progressive = _team_frame(progressive, team_name)
        progressive_summary = pass_metrics.progressive_pass_summary(
            team_progressive
        )
        entries, entry_stats = ctx.final_third(team_name)
        crosses, cross_summary, _ = ctx.crosses(team_name)
        team_defensive = _team_frame(defensive_actions, team_name)
        recoveries = (
            int(
                team_defensive["type_name"]
                .fillna("")
                .eq("Ball recovery")
                .sum()
            )
            if "type_name" in team_defensive.columns
            else 0
        )

        game_profile[team_name] = {
            "passes": pass_total,
            "successful_passes": successful,
            "pass_completion_pct": (
                successful / pass_total * 100.0
                if pass_total
                else 0.0
            ),
            "shots": int(shot_stats.get("total_shots", 0) or 0),
            "shots_on_target": int(
                shot_stats.get("shots_on_target", 0) or 0
            ),
            "progressive_passes": int(
                progressive_summary.get("successful", 0) or 0
            ),
            "final_third_entries": int(
                entry_stats.get("total_final_third", 0) or 0
            ),
            "crosses": int(len(crosses)),
            "cross_retention_pct": cross_summary.get(
                "retention_pct", 0.0
            ),
            "ball_recoveries": recoveries,
        }

    coverage = {
        "event_rows": int(len(ctx.df)),
        "receiver": data_quality.receiver_coverage(passes),
        "coordinates": data_quality.coordinate_coverage(
            ctx.df,
            columns=("x", "y"),
        ),
        "outcome": data_quality.outcome_coverage(ctx.df),
        "final_third_carries": {
            team: data_quality.carry_candidate_coverage_from_stats(
                ctx.final_third(team)[1]
            )
            for team in ctx.teams
        },
    }

    return _produced(
        {
            "metadata": dict(ctx.match_info),
            "result": {
                "home_team": ctx.home_team,
                "away_team": ctx.away_team,
                "home_score": home_score,
                "away_score": away_score,
            },
            "scorers": goal_list,
            "goal_origins": goals,
            "game_profile": game_profile,
            "data_coverage": coverage,
            "shots": shots_df,
        },
        empty=ctx.df.empty,
    )


def _formation_timeline(ctx: _BundleContext) -> _Produced:
    model = ctx.formation_timeline()
    return _produced(
        model,
        empty=not bool((model or {}).get("moments")),
    )


def _mean_positions(ctx: _BundleContext) -> _Produced:
    from src.metrics import player_metrics

    data = {}
    non_empty = False
    for team in ctx.teams:
        players, summary = player_metrics.get_mean_positions_profile(
            ctx.df,
            team,
            period="full",
        )
        data[team] = {
            "players": players,
            "summary": summary,
        }
        non_empty |= not players.empty

    return _produced(data, empty=not non_empty)


def _pass_network(ctx: _BundleContext) -> _Produced:
    from src.metrics import pass_network_metrics

    passes = ctx.passes()
    data = {}
    non_empty = False
    for team in ctx.teams:
        edges, nodes, summary = (
            pass_network_metrics.build_pass_network_profile(
                passes,
                ctx.df,
                team,
                period="full",
            )
        )
        data[team] = {
            "edges": edges,
            "nodes": nodes,
            "summary": summary,
        }
        non_empty |= not edges.empty or not nodes.empty

    return _produced(
        {
            "passes": passes,
            "teams": data,
        },
        empty=not non_empty,
    )


def _progressive_passes(ctx: _BundleContext) -> _Produced:
    from src.metrics import pass_metrics

    classified = ctx.progressive()
    teams = {}
    non_empty = False

    for team in ctx.teams:
        team_data = _team_frame(classified, team)
        attempts = (
            team_data[
                team_data["is_progressive_attempt"]
                .fillna(False)
                .astype(bool)
            ].copy()
            if "is_progressive_attempt" in team_data.columns
            else pd.DataFrame()
        )
        teams[team] = {
            "events": attempts,
            "summary": pass_metrics.progressive_pass_summary(
                team_data
            ),
            "player_ranking": (
                pass_metrics.progressive_pass_player_summary(
                    team_data,
                    limit=ctx.config.top_players,
                )
            ),
        }
        non_empty |= not attempts.empty

    return _produced(
        {
            "classified_passes": classified,
            "teams": teams,
        },
        empty=not non_empty,
    )


def _final_third_entries(ctx: _BundleContext) -> _Produced:
    data = {}
    non_empty = False
    for team in ctx.teams:
        entries, stats = ctx.final_third(team)
        data[team] = {
            "entries": entries,
            "stats": stats,
        }
        non_empty |= not entries.empty

    return _produced(
        {
            "carries": ctx.carries(),
            "teams": data,
        },
        empty=not non_empty,
    )


def _pass_locations(ctx: _BundleContext) -> _Produced:
    passes = ctx.passes()
    teams = {
        team: _team_frame(passes, team)
        for team in ctx.teams
    }
    return _produced(
        {
            "passes": passes,
            "teams": teams,
        },
        empty=passes.empty,
    )


def _cross_flow(ctx: _BundleContext) -> _Produced:
    data = {}
    non_empty = False
    for team in ctx.teams:
        crosses, summary, routes = ctx.crosses(team)
        data[team] = {
            "crosses": crosses,
            "summary": summary,
            "routes": routes,
        }
        non_empty |= not crosses.empty

    return _produced(data, empty=not non_empty)


def _build_up(ctx: _BundleContext) -> _Produced:
    from src.metrics import buildup_metrics, sequence_outcome_metrics

    data = {}
    non_empty = False

    for team in ctx.teams:
        opponent = ctx.other_team(team)

        def build(team_name=team, opponent_name=opponent):
            return buildup_metrics.find_buildup_sequences(
                ctx.df,
                team_name,
                opponent_name,
                metric_to_analyze="buildup_phase",
                triggers_buildups=(
                    list(ctx.config.buildup_triggers)
                    if ctx.config.buildup_triggers is not None
                    else None
                ),
            )

        sequences = ctx.once(("build-up", team), build)
        summary = sequence_outcome_metrics.summarize_sequences(
            sequences,
            sequence_kind="buildup",
        )
        data[team] = {
            "sequences": sequences,
            "summary": summary,
        }
        non_empty |= not sequences.empty

    comparison = sequence_outcome_metrics.build_sequence_comparison(
        data[ctx.home_team]["summary"],
        data[ctx.away_team]["summary"],
        ctx.home_team,
        ctx.away_team,
    )

    return _produced(
        {
            "teams": data,
            "comparison": comparison,
        },
        empty=not non_empty,
    )


def _defensive_shape(ctx: _BundleContext) -> _Produced:
    from src.metrics import defensive_metrics

    data = {}
    non_empty = False
    for team in ctx.teams:
        profile = defensive_metrics.build_defensive_shape_profile(
            ctx.df,
            team,
            period="full",
        )
        data[team] = profile
        non_empty |= bool(profile.get("action_count", 0))

    return _produced(data, empty=not non_empty)


def _ppda(ctx: _BundleContext) -> _Produced:
    from src.metrics import defensive_metrics

    teams = {}
    non_empty = False
    for team in ctx.teams:
        opponent = ctx.other_team(team)
        profile = defensive_metrics.calculate_ppda_profile(
            ctx.df,
            team,
            opponent,
        )
        teams[team] = profile
        overall = profile.get("overall", {}) if profile else {}
        non_empty |= bool(
            overall.get("opponent_passes", 0)
            or overall.get("defensive_actions", 0)
        )

    return _produced(
        {
            "teams": teams,
            "key_events": defensive_metrics.extract_ppda_key_events(
                ctx.df
            ),
        },
        empty=not non_empty,
    )


def _split_transition_sequences(
    frame: pd.DataFrame,
) -> list[pd.DataFrame]:
    if frame is None or frame.empty:
        return []
    if "loss_sequence_id" not in frame.columns:
        return [frame.copy()]
    return [
        group.copy().reset_index(drop=True)
        for _, group in frame.groupby(
            "loss_sequence_id",
            sort=False,
        )
    ]


def _defensive_transitions(ctx: _BundleContext) -> _Produced:
    from src.metrics import transition_metrics

    data = {}
    non_empty = False

    for team in ctx.teams:
        combined = transition_metrics.find_buildup_after_possession_loss(
            ctx.df,
            team,
            metric_to_analyze="defensive_transitions",
        )
        sequences = _split_transition_sequences(combined)
        stats = transition_metrics.calculate_def_transition_stats(
            sequences,
            team == ctx.away_team,
        ) if sequences else {}

        data[team] = {
            "combined": combined,
            "sequences": sequences,
            "stats": stats,
        }
        non_empty |= bool(sequences)

    return _produced(data, empty=not non_empty)


def _offensive_transitions(ctx: _BundleContext) -> _Produced:
    from src.metrics import transition_metrics

    data = {}
    non_empty = False

    for team in ctx.teams:
        opponent = ctx.other_team(team)
        combined = transition_metrics.find_buildup_after_possession_loss(
            ctx.df,
            opponent,
            metric_to_analyze="offensive_transitions",
        )
        sequences = _split_transition_sequences(combined)
        stats = (
            transition_metrics.calculate_off_transition_stats(
                sequences
            )
            if sequences
            else {}
        )
        data[team] = {
            "combined": combined,
            "sequences": sequences,
            "stats": stats,
        }
        non_empty |= bool(sequences)

    return _produced(data, empty=not non_empty)


def _restarts(ctx: _BundleContext) -> _Produced:
    from src.metrics import (
        restart_metrics,
        restart_panel_metrics,
        set_piece_metrics,
    )

    data = {}
    non_empty = False

    for team in ctx.teams:
        penalties = set_piece_metrics.extract_penalty_set_piece_sequences(
            ctx.df,
            team,
        )

        source = ctx.df
        if "Penalty" in source.columns and "type_name" in source.columns:
            penalty_flag = pd.to_numeric(
                source["Penalty"],
                errors="coerce",
            ).fillna(0).eq(1)
            duplicate_awards = (
                source["type_name"].fillna("").eq("Foul")
                & penalty_flag
            )
            source = source.loc[~duplicate_awards].copy()

        restarts = restart_metrics.extract_restart_sequences(
            source,
            team,
        )
        all_sequences = list(restarts or []) + list(penalties or [])
        analyzed, stats = (
            set_piece_metrics.analyze_and_summarize_set_pieces(
                all_sequences
            )
        )
        records = restart_panel_metrics.build_restart_records(
            analyzed,
            all_sequences,
        )

        data[team] = {
            "sequences": all_sequences,
            "analysis": analyzed,
            "summary": stats,
            "records": records,
        }
        non_empty |= bool(all_sequences)

    return _produced(data, empty=not non_empty)


def _player_highlights(ctx: _BundleContext) -> _Produced:
    from src.metrics import (
        defensive_contribution_metrics,
        player_pass_map_metrics,
        shot_sequence_involvement_metrics,
        threat_reception_metrics,
    )

    player_stats = ctx.player_stats()
    shot_sequence_stats = ctx.once(
        "shot-sequence-stats",
        lambda: _cached_shot_sequence_stats(ctx.df),
    )
    shot_ranking = (
        shot_sequence_involvement_metrics.prepare_shot_sequence_ranking(
            shot_sequence_stats,
            num_players=ctx.config.top_players,
        )
        if shot_sequence_stats is not None
        and not shot_sequence_stats.empty
        else pd.DataFrame()
    )
    defensive_ranking = (
        defensive_contribution_metrics.build_defensive_ranking(
            ctx.df,
            num_players=ctx.config.top_players,
        )
    )

    passes = ctx.passes()
    map_datasets = {}
    options = {}

    for team in ctx.teams:
        team_passes = _team_frame(passes, team)
        pass_players = []
        if (
            not team_passes.empty
            and "playerName" in team_passes.columns
        ):
            pass_players = [
                str(value)
                for value in team_passes["playerName"]
                .dropna()
                .unique()
                .tolist()
            ]

        passing_maps = {}
        for player in pass_players:
            player_passes = team_passes[
                team_passes["playerName"].astype(str).eq(player)
            ].copy()
            passing_maps[player] = {
                "events": player_pass_map_metrics.classify_player_passes(
                    player_passes
                ),
                "profile": player_pass_map_metrics.player_pass_profile(
                    player_passes
                ),
            }

        shooting_options = threat_reception_metrics.team_player_options(
            ctx.df,
            team,
        )
        shooting_maps = {}
        for option in shooting_options:
            player = option.get("player_name")
            if not player:
                continue
            receptions = (
                threat_reception_metrics.received_passes_for_player(
                    ctx.df,
                    player,
                    team,
                )
            )
            shooting_maps[player] = {
                "receptions": receptions,
                "summary": threat_reception_metrics.reception_summary(
                    receptions
                ),
            }

        defensive_options = (
            defensive_contribution_metrics.team_defensive_player_options(
                ctx.df,
                team,
            )
        )
        defensive_maps = {}
        if (
            "team_name" in ctx.df.columns
            and "playerName" in ctx.df.columns
        ):
            for option in defensive_options:
                player = option.get("player_name")
                if not player:
                    continue
                events = ctx.df[
                    ctx.df["team_name"].eq(team)
                    & ctx.df["playerName"].astype(str).eq(str(player))
                ].copy()
                defensive_maps[player] = {
                    "events": (
                        defensive_contribution_metrics
                        .classify_defensive_events(events)
                    ),
                    "profile": (
                        defensive_contribution_metrics
                        .player_defensive_profile(events)
                    ),
                }

        options[team] = {
            "passing": pass_players,
            "shooting": shooting_options,
            "defending": defensive_options,
        }
        map_datasets[team] = {
            "passing": passing_maps,
            "shooting": shooting_maps,
            "defending": defensive_maps,
        }

    return _produced(
        {
            "player_stats": player_stats,
            "shot_sequence_ranking": shot_ranking,
            "defensive_ranking": defensive_ranking,
            "player_options": options,
            "map_datasets": map_datasets,
        },
        empty=(
            (player_stats is None or player_stats.empty)
            and shot_ranking.empty
            and defensive_ranking.empty
        ),
    )


def _methodology(ctx: _BundleContext) -> _Produced:
    return _produced(
        {
            "scope": ctx.config.scope.value,
            "teams": ctx.teams,
            "source_signature": dataframe_signature(ctx.df),
            "manifest_id": ctx.config.manifest.id,
            "manifest_version": ctx.config.manifest.schema_version,
            "contract": (
                "REPORT-02 reuses canonical metric modules and returns "
                "neutral data only; figures and PDF rendering are out of scope."
            ),
        }
    )


_SECTION_PRODUCERS: dict[str, Callable[[_BundleContext], _Produced]] = {
    "overview": _overview,
    "formation-timeline": _formation_timeline,
    "mean-positions": _mean_positions,
    "pass-network": _pass_network,
    "progressive-passes": _progressive_passes,
    "final-third-entries": _final_third_entries,
    "pass-locations": _pass_locations,
    "cross-flow": _cross_flow,
    "build-up": _build_up,
    "defensive-shape": _defensive_shape,
    "ppda": _ppda,
    "defensive-transitions": _defensive_transitions,
    "offensive-transitions": _offensive_transitions,
    "restarts": _restarts,
    "player-highlights": _player_highlights,
    "methodology-appendix": _methodology,
}


def build_match_report_data_bundle(
    df_processed: pd.DataFrame,
    match_info: Mapping[str, Any] | None,
    report_config: MatchReportBundleConfig | Mapping[str, Any] | None = None,
) -> MatchReportDataBundle:
    """Build every manifest section independently for Full Match / both teams."""

    if not isinstance(df_processed, pd.DataFrame):
        raise TypeError("df_processed must be a pandas DataFrame")

    config = MatchReportBundleConfig.from_value(report_config)
    info = dict(match_info or {})
    teams = _resolve_teams(df_processed, info, config)

    context = _BundleContext(
        df_processed=df_processed,
        match_info=info,
        config=config,
        teams=teams,
    )

    enabled = (
        set(config.enabled_sections)
        if config.enabled_sections is not None
        else None
    )

    sections: list[ReportSectionBundle] = []

    for spec in config.manifest.sections:
        if enabled is not None and spec.id not in enabled:
            sections.append(
                ReportSectionBundle(
                    id=spec.id,
                    status=ReportSectionStatus.SKIPPED,
                )
            )
            continue

        producer = _SECTION_PRODUCERS.get(spec.id)
        if producer is None:
            sections.append(
                ReportSectionBundle(
                    id=spec.id,
                    status=ReportSectionStatus.SKIPPED,
                    error_message="No REPORT-02 producer registered.",
                )
            )
            continue

        try:
            produced = producer(context)
            if not isinstance(produced, _Produced):
                produced = _produced(produced)

            sections.append(
                ReportSectionBundle(
                    id=spec.id,
                    status=(
                        ReportSectionStatus.EMPTY
                        if produced.empty
                        else ReportSectionStatus.GENERATED
                    ),
                    data=produced.data,
                )
            )
        except Exception as exc:
            sections.append(
                ReportSectionBundle(
                    id=spec.id,
                    status=ReportSectionStatus.ERROR,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )

    return MatchReportDataBundle(
        manifest_id=config.manifest.id,
        manifest_version=config.manifest.schema_version,
        source_signature=dataframe_signature(df_processed),
        scope=config.scope,
        teams=teams,
        match_info=info,
        sections=tuple(sections),
    )
