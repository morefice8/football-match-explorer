"""Static Plotly figure catalog for Match Reports.

The catalog is intentionally independent from app.py, Dash callbacks, DOM state
and browser screenshots. It consumes REPORT-02 bundle data and delegates to
validated Plotly renderers through renderer_registry.py whenever one exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go
from src.utils.sequence_normalization import normalize_sequence

from src.reporting.manifest import REPORT_MANIFEST
from src.reporting.models import ReportManifest
from src.reporting.renderer_registry import RendererRegistry
from src.reporting.selectors import (
    select_representative_restart,
    select_representative_sequence,
    select_top_defender,
    select_top_passer,
    select_top_shooting_contributor,
)


class ReportFigureStatus(str, Enum):
    GENERATED = "generated"
    EMPTY = "empty"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass(frozen=True)
class FigureCatalogConfig:
    home_color: str | None = None
    away_color: str | None = None


@dataclass(frozen=True)
class FigurePlan:
    section_id: str
    section_order: int
    figure_id: str
    title: str
    width_px: int
    height_px: int
    team_name: str | None = None
    team_index: int | None = None
    variant: str = "summary"

    @property
    def is_away(self) -> bool:
        return self.team_index == 1


@dataclass(frozen=True)
class ReportFigureArtifact:
    id: str
    section_id: str
    title: str
    variant: str
    filename: str
    width_px: int
    height_px: int
    status: ReportFigureStatus
    figure: go.Figure
    team_name: str | None = None
    is_away: bool | None = None
    renderer_id: str | None = None
    selection_reason: str | None = None
    source_section_status: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    selection: dict | None = None


@dataclass(frozen=True)
class MatchReportFigureCatalog:
    manifest_id: str
    manifest_version: str
    figures: tuple[ReportFigureArtifact, ...]

    def by_id(
        self,
        figure_id: str,
    ) -> tuple[ReportFigureArtifact, ...]:
        return tuple(
            item
            for item in self.figures
            if item.id == figure_id
        )


@dataclass(frozen=True)
class _Rendered:
    figure: go.Figure
    status: ReportFigureStatus
    renderer_id: str | None = None
    selection_reason: str | None = None
    selection: dict | None = None


def _slug(value: str | None) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "item"


def _filename(plan: FigurePlan) -> str:
    parts = [
        f"{plan.section_order:02d}",
        _slug(plan.figure_id),
    ]
    if plan.team_name:
        parts.append(_slug(plan.team_name))
    if plan.variant and plan.variant != "summary":
        parts.append(_slug(plan.variant))
    return "--".join(parts) + ".png"


def _placeholder(plan: FigurePlan, message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        width=plan.width_px,
        height=plan.height_px,
        template="plotly_white",
        margin=dict(l=50, r=50, t=90, b=50),
        title=plan.title,
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="center",
                font=dict(size=18),
            )
        ],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _pdf_layout(figure: go.Figure, plan: FigurePlan) -> go.Figure:
    figure.update_layout(
        width=plan.width_px,
        height=plan.height_px,
    )
    return figure


def _section(bundle, section_id: str):
    try:
        return bundle.section(section_id)
    except Exception:
        return None


def _status_value(section) -> str | None:
    if section is None:
        return None
    status = getattr(section, "status", None)
    return getattr(status, "value", status)


def _teams(bundle) -> tuple[str, ...]:
    return tuple(str(team) for team in (getattr(bundle, "teams", ()) or ()))


def _match_info(bundle) -> Mapping[str, Any]:
    return getattr(bundle, "match_info", {}) or {}


def _team_color(
    bundle,
    plan: FigurePlan,
    config: FigureCatalogConfig,
) -> str:
    info = _match_info(bundle)
    if plan.team_index == 0:
        return str(
            config.home_color
            or info.get("hcol")
            or info.get("home_color")
            or info.get("homeColor")
            or "tomato"
        )
    return str(
        config.away_color
        or info.get("acol")
        or info.get("away_color")
        or info.get("awayColor")
        or "skyblue"
    )


def _as_frame(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        return value.to_frame().T
    if isinstance(value, Mapping):
        return pd.DataFrame([dict(value)])
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes),
    ):
        if not value:
            return pd.DataFrame()
        if all(isinstance(item, Mapping) for item in value):
            return pd.DataFrame(list(value))
    return pd.DataFrame()


def _id_column(frame: pd.DataFrame) -> str | None:
    for column in (
        "sequence_id",
        "trigger_sequence_id",
        "buildup_sequence_id",
        "loss_sequence_id",
        "restart_id",
        "id",
    ):
        if column in frame.columns:
            return column
    return None


def _player_column(frame: pd.DataFrame) -> str | None:
    for column in (
        "playerName",
        "player_name",
        "Player",
        "name",
    ):
        if column in frame.columns:
            return column
    return None


def _filter_player_candidates(
    frame: pd.DataFrame,
    valid_players: Sequence[str],
) -> pd.DataFrame:
    if frame.empty:
        return frame
    valid = {str(player) for player in valid_players}
    column = _player_column(frame)
    if column is not None:
        return frame[
            frame[column].astype(str).isin(valid)
        ].copy()
    if frame.index.name in {
        "playerName",
        "player_name",
        "Player",
    }:
        return frame[
            frame.index.astype(str).isin(valid)
        ].copy()
    return frame.copy()


def _sequence_candidates(payload: Mapping[str, Any]) -> pd.DataFrame:
    for key in (
        "summary",
        "sequence_summary",
        "sequences_summary",
    ):
        frame = _as_frame(payload.get(key))
        if not frame.empty:
            return frame

    sequences = payload.get("sequences")
    if isinstance(sequences, Sequence) and not isinstance(
        sequences,
        (str, bytes),
    ):
        rows: list[dict[str, Any]] = []
        for sequence in sequences:
            frame = _as_frame(sequence)
            if frame.empty:
                continue

            id_column = _id_column(frame)
            if id_column is None:
                continue

            identifier = frame[id_column].dropna()
            if identifier.empty:
                continue

            row: dict[str, Any] = {
                id_column: identifier.iloc[0],
                "event_count": int(len(frame)),
            }

            for team_column in ("team_name", "teamName", "Team", "team"):
                if team_column in frame.columns:
                    values = frame[team_column].dropna()
                    if not values.empty:
                        row[team_column] = values.iloc[0]
                    break

            milestone = None
            for column in (
                "sequence_outcome_type",
                "terminal_outcome",
                "sequence_outcome",
                "final_outcome",
            ):
                if column in frame.columns:
                    values = frame[column].dropna()
                    if not values.empty:
                        milestone = values.iloc[0]
                        break
            if milestone is not None:
                row["milestone"] = milestone

            for progression_column in (
                "max_controlled_x",
                "territorial_progression",
                "territorial_gain",
            ):
                if progression_column in frame.columns:
                    values = pd.to_numeric(
                        frame[progression_column],
                        errors="coerce",
                    ).dropna()
                    if not values.empty:
                        row["max_controlled_x"] = float(values.max())
                        break

            if "total_seconds" in frame.columns:
                seconds = pd.to_numeric(
                    frame["total_seconds"],
                    errors="coerce",
                ).dropna()
                if not seconds.empty:
                    row["duration_seconds"] = float(
                        seconds.max() - seconds.min()
                    )
            elif {"timeMin", "timeSec"}.issubset(frame.columns):
                minutes = pd.to_numeric(frame["timeMin"], errors="coerce")
                seconds = pd.to_numeric(frame["timeSec"], errors="coerce")
                clock = (minutes * 60 + seconds).dropna()
                if not clock.empty:
                    row["duration_seconds"] = float(
                        clock.max() - clock.min()
                    )

            rows.append(row)

        if rows:
            return pd.DataFrame(rows)

    combined = _as_frame(payload.get("combined"))
    id_column = _id_column(combined)
    if not combined.empty and id_column is not None:
        return combined.drop_duplicates(
            subset=[id_column],
            keep="first",
        )
    return pd.DataFrame()


def _find_sequence(
    sequences: Any,
    selected_id: str,
) -> pd.DataFrame | None:
    # Real REPORT-02 payloads preserve multiple source shapes. REPORT-04 is
    # the presentation adapter and accepts them without changing calculations.
    def _selected(frame: pd.DataFrame) -> pd.DataFrame | None:
        if frame.empty:
            return None
        id_column = _id_column(frame)
        if id_column is None:
            return None
        found = frame[
            frame[id_column]
            .astype(str)
            .eq(str(selected_id))
        ].copy()
        return None if found.empty else found

    direct = _selected(_as_frame(sequences))
    if direct is not None:
        return direct

    if isinstance(sequences, Sequence) and not isinstance(
        sequences,
        (str, bytes),
    ):
        for sequence in sequences:
            frame = _as_frame(sequence)
            if frame.empty:
                continue
            id_column = _id_column(frame)
            if id_column is None:
                continue
            if str(frame[id_column].iloc[0]) == str(selected_id):
                return frame.copy()

    return None


def _formation_states(
    model: Mapping[str, Any],
    team_name: str,
    team_index: int,
) -> tuple[Any, Any, Any]:
    side = "home" if team_index == 0 else "away"
    player_map = (
        model.get(f"{side}_player_data_map")
        or model.get("player_data_map")
        or {}
    )

    side_payload = model.get(side)
    if isinstance(side_payload, Mapping):
        states = (
            side_payload.get("states")
            or side_payload.get("timeline")
            or side_payload.get("moments")
            or []
        )
        if isinstance(states, Sequence) and states:
            return states[0], states[-1], player_map

    teams_payload = model.get("teams")
    if isinstance(teams_payload, Mapping):
        team_payload = (
            teams_payload.get(team_name)
            or teams_payload.get(side)
        )
        if isinstance(team_payload, Mapping):
            states = (
                team_payload.get("states")
                or team_payload.get("timeline")
                or team_payload.get("moments")
                or []
            )
            if isinstance(states, Sequence) and states:
                return (
                    states[0],
                    states[-1],
                    team_payload.get("player_data_map") or player_map,
                )

    extracted = []
    for moment in model.get("moments") or []:
        if not isinstance(moment, Mapping):
            continue
        state = (
            moment.get(f"{side}_state")
            or moment.get(side)
            or moment.get(team_name)
        )
        if state is not None:
            extracted.append(state)

    if extracted:
        return extracted[0], extracted[-1], player_map

    return None, None, player_map


def _ppda_summary(
    bundle,
    plan: FigurePlan,
) -> _Rendered:
    section = _section(bundle, "ppda")
    data = getattr(section, "data", {}) or {}
    profiles = data.get("teams", {})

    rows = []
    for team in _teams(bundle):
        overall = (profiles.get(team, {}) or {}).get("overall", {}) or {}
        rows.append([
            team,
            overall.get("full", overall.get("ppda", "—")),
            overall.get("1H", overall.get("first_half", "—")),
            overall.get("2H", overall.get("second_half", "—")),
        ])

    if not rows:
        return _Rendered(
            _placeholder(
                plan,
                "No PPDA Full/1H/2H summary available.",
            ),
            ReportFigureStatus.EMPTY,
        )

    headers = ["Team", "Full", "1H", "2H"]
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=headers),
                cells=dict(
                    values=[
                        [row[index] for row in rows]
                        for index in range(len(headers))
                    ]
                ),
            )
        ]
    )
    fig.update_layout(
        title="PPDA — Full Match / 1H / 2H",
        template="plotly_white",
        margin=dict(l=30, r=30, t=80, b=30),
    )
    return _Rendered(
        _pdf_layout(fig, plan),
        ReportFigureStatus.GENERATED,
        "report-ppda-summary",
    )


def _cross_flow_summary(
    bundle,
    plan: FigurePlan,
) -> _Rendered:
    section = _section(bundle, "cross-flow")
    payload = (getattr(section, "data", {}) or {}).get(
        plan.team_name,
        {},
    )
    routes = _as_frame(payload.get("routes"))

    if routes.empty:
        return _Rendered(
            _placeholder(
                plan,
                f"No cross-flow routes for {plan.team_name}.",
            ),
            ReportFigureStatus.EMPTY,
        )

    origin = next(
        (
            column
            for column in ("Origin Zone", "origin_zone", "origin")
            if column in routes.columns
        ),
        None,
    )
    destination = next(
        (
            column
            for column in (
                "Destination Zone",
                "destination_zone",
                "destination",
            )
            if column in routes.columns
        ),
        None,
    )
    count = next(
        (
            column
            for column in ("Crosses", "crosses", "count")
            if column in routes.columns
        ),
        None,
    )

    if not all((origin, destination, count)):
        return _Rendered(
            _placeholder(
                plan,
                "Cross routes do not expose canonical origin, "
                "destination and count fields.",
            ),
            ReportFigureStatus.EMPTY,
        )

    labels = list(
        dict.fromkeys(
            routes[origin].astype(str).tolist()
            + routes[destination].astype(str).tolist()
        )
    )
    lookup = {
        label: index
        for index, label in enumerate(labels)
    }

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=labels),
                link=dict(
                    source=[
                        lookup[value]
                        for value in routes[origin].astype(str)
                    ],
                    target=[
                        lookup[value]
                        for value in routes[destination].astype(str)
                    ],
                    value=routes[count].tolist(),
                ),
            )
        ]
    )
    fig.update_layout(
        title=f"Cross Flow — {plan.team_name}",
        template="plotly_white",
        margin=dict(l=30, r=30, t=80, b=30),
    )
    return _Rendered(
        _pdf_layout(fig, plan),
        ReportFigureStatus.GENERATED,
        "report-cross-flow-sankey",
    )


def _render_team_renderer(
    bundle,
    plan: FigurePlan,
    registry: RendererRegistry,
    config: FigureCatalogConfig,
) -> _Rendered:
    color = _team_color(bundle, plan, config)

    if plan.figure_id == "mean-positions-figure":
        section = _section(bundle, "mean-positions")
        payload = (getattr(section, "data", {}) or {}).get(
            plan.team_name,
            {},
        )
        players = _as_frame(payload.get("players"))
        if players.empty:
            return _Rendered(
                _placeholder(
                    plan,
                    f"No mean-position sample for {plan.team_name}.",
                ),
                ReportFigureStatus.EMPTY,
            )
        figure = registry.resolve("mean-positions")(
            players,
            payload.get("summary", {}),
            is_away=plan.is_away,
        )
        return _Rendered(
            _pdf_layout(figure, plan),
            ReportFigureStatus.GENERATED,
            "mean-positions",
        )

    if plan.figure_id == "pass-network-figure":
        section = _section(bundle, "pass-network")
        payload = (
            (getattr(section, "data", {}) or {})
            .get("teams", {})
            .get(plan.team_name, {})
        )
        edges = _as_frame(payload.get("edges"))
        nodes = _as_frame(payload.get("nodes"))
        if edges.empty and nodes.empty:
            return _Rendered(
                _placeholder(
                    plan,
                    f"No pass-network sample for {plan.team_name}.",
                ),
                ReportFigureStatus.EMPTY,
            )
        figure = registry.resolve("pass-network")(
            edges,
            nodes,
            plan.team_name,
            is_away=plan.is_away,
        )
        return _Rendered(
            _pdf_layout(figure, plan),
            ReportFigureStatus.GENERATED,
            "pass-network",
        )

    if plan.figure_id == "progressive-passes-figure":
        section = _section(bundle, "progressive-passes")
        payload = (
            (getattr(section, "data", {}) or {})
            .get("teams", {})
            .get(plan.team_name, {})
        )
        events = _as_frame(payload.get("events"))
        if events.empty:
            return _Rendered(
                _placeholder(
                    plan,
                    f"No progressive passes for {plan.team_name}.",
                ),
                ReportFigureStatus.EMPTY,
            )
        figure = registry.resolve("progressive-passes")(
            events,
            plan.team_name,
            color,
            is_away=plan.is_away,
        )
        return _Rendered(
            _pdf_layout(figure, plan),
            ReportFigureStatus.GENERATED,
            "progressive-passes",
        )

    if plan.figure_id == "final-third-entries-figure":
        section = _section(bundle, "final-third-entries")
        payload = (
            (getattr(section, "data", {}) or {})
            .get("teams", {})
            .get(plan.team_name, {})
        )
        entries = _as_frame(payload.get("entries"))
        if entries.empty:
            return _Rendered(
                _placeholder(
                    plan,
                    f"No final-third entries for {plan.team_name}.",
                ),
                ReportFigureStatus.EMPTY,
            )
        figure = registry.resolve("final-third-entries")(
            entries,
            payload.get("stats", {}),
            plan.team_name,
            color,
            is_away=plan.is_away,
        )
        return _Rendered(
            _pdf_layout(figure, plan),
            ReportFigureStatus.GENERATED,
            "final-third-entries",
        )

    if plan.figure_id == "pass-locations-figure":
        section = _section(bundle, "pass-locations")
        passes = _as_frame(
            (getattr(section, "data", {}) or {})
            .get("teams", {})
            .get(plan.team_name)
        )
        if passes.empty:
            return _Rendered(
                _placeholder(
                    plan,
                    f"No pass locations for {plan.team_name}.",
                ),
                ReportFigureStatus.EMPTY,
            )
        figure = registry.resolve("pass-locations")(
            passes,
            plan.team_name,
            is_away=plan.is_away,
        )
        return _Rendered(
            _pdf_layout(figure, plan),
            ReportFigureStatus.GENERATED,
            "pass-locations",
        )

    if plan.figure_id in {
        "cross-origin-map",
        "cross-destination-map",
    }:
        section = _section(bundle, "cross-flow")
        crosses = _as_frame(
            (getattr(section, "data", {}) or {})
            .get(plan.team_name, {})
            .get("crosses")
        )
        if crosses.empty:
            return _Rendered(
                _placeholder(
                    plan,
                    f"No crosses for {plan.team_name}.",
                ),
                ReportFigureStatus.EMPTY,
            )
        location_type = (
            "origin"
            if plan.figure_id == "cross-origin-map"
            else "destination"
        )
        figure = registry.resolve("cross-heatmap")(
            crosses,
            location_type=location_type,
            is_away=plan.is_away,
        )
        return _Rendered(
            _pdf_layout(figure, plan),
            ReportFigureStatus.GENERATED,
            "cross-heatmap",
        )

    if plan.figure_id == "defensive-shape-figure":
        section = _section(bundle, "defensive-shape")
        profile = (getattr(section, "data", {}) or {}).get(
            plan.team_name,
            {},
        )
        if not profile or not profile.get("action_count", 0):
            return _Rendered(
                _placeholder(
                    plan,
                    f"No defensive-shape sample for {plan.team_name}.",
                ),
                ReportFigureStatus.EMPTY,
            )
        figure = registry.resolve("defensive-shape")(
            profile,
            color,
            mode="density",
        )
        return _Rendered(
            _pdf_layout(figure, plan),
            ReportFigureStatus.GENERATED,
            "defensive-shape",
        )

    raise KeyError(plan.figure_id)


def _render_formation(
    bundle,
    plan: FigurePlan,
    registry: RendererRegistry,
) -> _Rendered:
    section = _section(bundle, "formation-timeline")
    model = getattr(section, "data", {}) or {}
    starting, final, player_map = _formation_states(
        model,
        plan.team_name or "",
        int(plan.team_index or 0),
    )
    state = starting if plan.variant == "starting" else final

    if state is None:
        return _Rendered(
            _placeholder(
                plan,
                f"No {plan.variant} formation state for {plan.team_name}.",
            ),
            ReportFigureStatus.EMPTY,
        )

    figure = registry.resolve("formation-state")(
        state,
        player_map,
        is_away=plan.is_away,
    )
    return _Rendered(
        _pdf_layout(figure, plan),
        ReportFigureStatus.GENERATED,
        "formation-state",
    )


def _render_build_up(
    bundle,
    plan: FigurePlan,
    registry: RendererRegistry,
    config: FigureCatalogConfig,
) -> _Rendered:
    section = _section(bundle, "build-up")
    payload = (
        (getattr(section, "data", {}) or {})
        .get("teams", {})
        .get(plan.team_name, {})
    )
    candidates = _sequence_candidates(payload)
    selection = (
        select_representative_sequence(
            candidates,
            plan.team_name,
            category="build-up",
        )
        if not candidates.empty
        else None
    )

    if selection is None:
        return _Rendered(
            _placeholder(
                plan,
                f"No representative build-up for {plan.team_name}.",
            ),
            ReportFigureStatus.EMPTY,
        )

    sequence = _find_sequence(
        payload.get("sequences"),
        selection.selected_id,
    )
    if sequence is None:
        return _Rendered(
            _placeholder(
                plan,
                "Selected build-up event rows are not available.",
            ),
            ReportFigureStatus.EMPTY,
            selection_reason=selection.selection_reason,
        )

    figure = registry.resolve("build-up-sequence")(
        sequence,
        _team_color(bundle, plan, config),
        plan.is_away,
    )
    return _Rendered(
        _pdf_layout(figure, plan),
        ReportFigureStatus.GENERATED,
        "build-up-sequence",
        selection.selection_reason,
        selection.to_dict(),
    )


def _render_transition(
    bundle,
    plan: FigurePlan,
    registry: RendererRegistry,
    config: FigureCatalogConfig,
    category: str,
) -> _Rendered:
    section = _section(bundle, plan.section_id)
    payload = (getattr(section, "data", {}) or {}).get(
        plan.team_name,
        {},
    )
    candidates = _sequence_candidates(payload)
    if category == "defensive-transition" and not candidates.empty:
        # Defensive-transition payloads are already keyed/scoped by the team
        # that lost the ball. Their event rows describe the opponent's actions
        # after that loss, so row-level team_name is the opponent. Adapt only
        # the candidate view so selection stays attributed to the defending team.
        candidates = candidates.copy()
        for team_column in ("team_name", "teamName", "Team", "team"):
            if team_column in candidates.columns:
                candidates[team_column] = plan.team_name

    selection = (
        select_representative_sequence(
            candidates,
            plan.team_name,
            category=category,
        )
        if not candidates.empty
        else None
    )
    if selection is None:
        return _Rendered(
            _placeholder(
                plan,
                f"No representative {category} for {plan.team_name}.",
            ),
            ReportFigureStatus.EMPTY,
        )

    sequence = _find_sequence(
        payload.get("sequences"),
        selection.selected_id,
    )
    if sequence is None:
        combined = _as_frame(payload.get("combined"))
        id_column = _id_column(combined)
        if id_column is not None:
            selected = combined[
                combined[id_column]
                .astype(str)
                .eq(str(selection.selected_id))
            ].copy()
            if not selected.empty:
                sequence = selected

    if sequence is None:
        return _Rendered(
            _placeholder(
                plan,
                "Selected transition event rows are not available.",
            ),
            ReportFigureStatus.EMPTY,
            selection_reason=selection.selection_reason,
        )

    sequence_type = category.replace("-", "_")
    normalize_kwargs: dict[str, Any] = {}
    if sequence_type == "offensive_transition":
        normalize_kwargs["team_name"] = plan.team_name
        normalize_kwargs["opponent_name"] = next(
            (
                team
                for team in _teams(bundle)
                if team != plan.team_name
            ),
            None,
        )

    normalized_sequence = normalize_sequence(
        sequence,
        sequence_type=sequence_type,
        **normalize_kwargs,
    )
    figure = registry.resolve("sequence-explorer")(
        normalized_sequence,
        team_color=_team_color(bundle, plan, config),
        height=plan.height_px,
    )
    return _Rendered(
        _pdf_layout(figure, plan),
        ReportFigureStatus.GENERATED,
        "sequence-explorer",
        selection.selection_reason,
        selection.to_dict(),
    )


def _render_restart(
    bundle,
    plan: FigurePlan,
    registry: RendererRegistry,
    config: FigureCatalogConfig,
) -> _Rendered:
    section = _section(bundle, "restarts")
    payload = (getattr(section, "data", {}) or {}).get(
        plan.team_name,
        {},
    )
    records_frame = _as_frame(payload.get("records"))
    if records_frame.empty:
        return _Rendered(
            _placeholder(
                plan,
                f"No restart sample for {plan.team_name}.",
            ),
            ReportFigureStatus.EMPTY,
        )

    selection = select_representative_restart(
        records_frame,
        plan.team_name,
    )
    if (
        plan.figure_id == "restart-top-sequence"
        and selection is None
    ):
        return _Rendered(
            _placeholder(
                plan,
                f"No representative restart for {plan.team_name}.",
            ),
            ReportFigureStatus.EMPTY,
        )

    # restart_map is list-of-records based; DataFrame truth testing is ambiguous.
    records = records_frame.to_dict(orient="records")
    figure = registry.resolve("restart-map")(
        records,
        team_color=_team_color(bundle, plan, config),
        selected_sequence_id=(
            selection.selected_id
            if selection is not None
            else None
        ),
    )
    return _Rendered(
        _pdf_layout(figure, plan),
        ReportFigureStatus.GENERATED,
        "restart-map",
        selection.selection_reason if selection else None,
        selection.to_dict() if selection else None,
    )


def _render_ppda(
    bundle,
    plan: FigurePlan,
    registry: RendererRegistry,
    config: FigureCatalogConfig,
) -> _Rendered:
    if plan.variant == "summary":
        return _ppda_summary(bundle, plan)

    section = _section(bundle, "ppda")
    data = getattr(section, "data", {}) or {}
    teams = _teams(bundle)
    if len(teams) != 2:
        return _Rendered(
            _placeholder(
                plan,
                "PPDA timeline requires both teams.",
            ),
            ReportFigureStatus.EMPTY,
        )

    profiles = data.get("teams", {})
    home_profile = profiles.get(teams[0], {})
    away_profile = profiles.get(teams[1], {})
    if not home_profile and not away_profile:
        return _Rendered(
            _placeholder(plan, "No PPDA profiles available."),
            ReportFigureStatus.EMPTY,
        )

    home_plan = FigurePlan(
        plan.section_id,
        plan.section_order,
        plan.figure_id,
        plan.title,
        plan.width_px,
        plan.height_px,
        teams[0],
        0,
        "timeline",
    )
    away_plan = FigurePlan(
        plan.section_id,
        plan.section_order,
        plan.figure_id,
        plan.title,
        plan.width_px,
        plan.height_px,
        teams[1],
        1,
        "timeline",
    )

    figure = registry.resolve("ppda-timeline")(
        home_profile,
        away_profile,
        data.get("key_events"),
        teams[0],
        teams[1],
        _team_color(bundle, home_plan, config),
        _team_color(bundle, away_plan, config),
    )
    return _Rendered(
        _pdf_layout(figure, plan),
        ReportFigureStatus.GENERATED,
        "ppda-timeline",
    )


def _render_player_highlight(
    bundle,
    plan: FigurePlan,
    registry: RendererRegistry,
    config: FigureCatalogConfig,
) -> _Rendered:
    section = _section(bundle, "player-highlights")
    data = getattr(section, "data", {}) or {}
    team_maps = (
        data.get("map_datasets", {})
        .get(plan.team_name, {})
    )

    if plan.figure_id == "player-highlight-passing":
        maps = team_maps.get("passing", {}) or {}
        candidates = _filter_player_candidates(
            _as_frame(data.get("player_stats")),
            list(maps),
        )
        selection = (
            select_top_passer(candidates, plan.team_name)
            if not candidates.empty
            else None
        )
        if selection is None or selection.selected_name not in maps:
            return _Rendered(
                _placeholder(
                    plan,
                    f"No top-passer map for {plan.team_name}.",
                ),
                ReportFigureStatus.EMPTY,
            )
        payload = maps[selection.selected_name]
        figure = registry.resolve("player-pass-map")(
            _as_frame(payload.get("events")),
            selection.selected_name,
            _team_color(bundle, plan, config),
            player_jersey="?",
            is_away_team=plan.is_away,
        )
        return _Rendered(
            _pdf_layout(figure, plan),
            ReportFigureStatus.GENERATED,
            "player-pass-map",
            selection.selection_reason,
            selection.to_dict(),
        )

    if plan.figure_id == "player-highlight-shooting":
        maps = team_maps.get("shooting", {}) or {}
        candidates = _filter_player_candidates(
            _as_frame(data.get("shot_sequence_ranking")),
            list(maps),
        )
        selection = (
            select_top_shooting_contributor(
                candidates,
                plan.team_name,
            )
            if not candidates.empty
            else None
        )
        if selection is None or selection.selected_name not in maps:
            return _Rendered(
                _placeholder(
                    plan,
                    f"No shooting-contributor map for {plan.team_name}.",
                ),
                ReportFigureStatus.EMPTY,
            )
        payload = maps[selection.selected_name]
        figure = registry.resolve("player-reception-map")(
            _as_frame(payload.get("receptions")),
            selected_player=selection.selected_name,
            jersey="?",
            team_color=_team_color(bundle, plan, config),
            summary=payload.get("summary", {}),
            is_away=plan.is_away,
        )
        return _Rendered(
            _pdf_layout(figure, plan),
            ReportFigureStatus.GENERATED,
            "player-reception-map",
            selection.selection_reason,
            selection.to_dict(),
        )

    maps = team_maps.get("defending", {}) or {}
    candidates = _filter_player_candidates(
        _as_frame(data.get("defensive_ranking")),
        list(maps),
    )
    selection = (
        select_top_defender(candidates, plan.team_name)
        if not candidates.empty
        else None
    )
    if selection is None or selection.selected_name not in maps:
        return _Rendered(
            _placeholder(
                plan,
                f"No top-defender map for {plan.team_name}.",
            ),
            ReportFigureStatus.EMPTY,
        )
    payload = maps[selection.selected_name]
    figure = registry.resolve("player-defensive-map")(
        _as_frame(payload.get("events")),
        selected_player=selection.selected_name,
        jersey="?",
        team_color=_team_color(bundle, plan, config),
        is_away=plan.is_away,
    )
    return _Rendered(
        _pdf_layout(figure, plan),
        ReportFigureStatus.GENERATED,
        "player-defensive-map",
        selection.selection_reason,
        selection.to_dict(),
    )


def _render_plan(
    bundle,
    plan: FigurePlan,
    registry: RendererRegistry,
    config: FigureCatalogConfig,
) -> _Rendered:
    if plan.figure_id == "formation-timeline-figure":
        return _render_formation(bundle, plan, registry)

    if plan.figure_id in {
        "mean-positions-figure",
        "pass-network-figure",
        "progressive-passes-figure",
        "final-third-entries-figure",
        "pass-locations-figure",
        "cross-origin-map",
        "cross-destination-map",
        "defensive-shape-figure",
    }:
        return _render_team_renderer(
            bundle,
            plan,
            registry,
            config,
        )

    if plan.figure_id == "cross-flow-figure":
        return _cross_flow_summary(bundle, plan)

    if plan.figure_id == "build-up-top-sequence":
        return _render_build_up(
            bundle,
            plan,
            registry,
            config,
        )

    if plan.figure_id == "ppda-figure":
        return _render_ppda(
            bundle,
            plan,
            registry,
            config,
        )

    if plan.figure_id == "defensive-transitions-top-sequence":
        return _render_transition(
            bundle,
            plan,
            registry,
            config,
            "defensive-transition",
        )

    if plan.figure_id == "offensive-transitions-top-sequence":
        return _render_transition(
            bundle,
            plan,
            registry,
            config,
            "offensive-transition",
        )

    if plan.figure_id in {
        "restart-map",
        "restart-top-sequence",
    }:
        return _render_restart(
            bundle,
            plan,
            registry,
            config,
        )

    if plan.figure_id.startswith("player-highlight-"):
        return _render_player_highlight(
            bundle,
            plan,
            registry,
            config,
        )

    return _Rendered(
        _placeholder(
            plan,
            "No static renderer registered for this manifest figure.",
        ),
        ReportFigureStatus.SKIPPED,
    )


def build_figure_plans(
    bundle,
    manifest: ReportManifest = REPORT_MANIFEST,
) -> tuple[FigurePlan, ...]:
    teams = _teams(bundle)
    plans: list[FigurePlan] = []

    for section in manifest.sections:
        for figure in section.figures:
            if figure.id == "formation-timeline-figure":
                for index, team_name in enumerate(teams):
                    for variant in ("starting", "final"):
                        plans.append(
                            FigurePlan(
                                section.id,
                                section.order,
                                figure.id,
                                figure.title,
                                figure.export.width_px,
                                figure.export.height_px,
                                team_name,
                                index,
                                variant,
                            )
                        )
                continue

            if figure.id == "ppda-figure":
                for variant in ("timeline", "summary"):
                    plans.append(
                        FigurePlan(
                            section.id,
                            section.order,
                            figure.id,
                            figure.title,
                            figure.export.width_px,
                            figure.export.height_px,
                            None,
                            None,
                            variant,
                        )
                    )
                continue

            for index, team_name in enumerate(teams):
                plans.append(
                    FigurePlan(
                        section.id,
                        section.order,
                        figure.id,
                        figure.title,
                        figure.export.width_px,
                        figure.export.height_px,
                        team_name,
                        index,
                        "summary",
                    )
                )

    return tuple(plans)


def build_report_figure_catalog(
    bundle,
    manifest: ReportManifest = REPORT_MANIFEST,
    *,
    config: FigureCatalogConfig | None = None,
    registry: RendererRegistry | None = None,
) -> MatchReportFigureCatalog:
    config = config or FigureCatalogConfig()
    registry = registry or RendererRegistry()
    artifacts: list[ReportFigureArtifact] = []

    for plan in build_figure_plans(bundle, manifest):
        section = _section(bundle, plan.section_id)
        source_status = _status_value(section)
        error_type = None
        error_message = None

        if section is None:
            rendered = _Rendered(
                _placeholder(
                    plan,
                    "Corresponding report-data section is missing.",
                ),
                ReportFigureStatus.SKIPPED,
            )
        elif source_status == "error":
            error_type = getattr(section, "error_type", None)
            error_message = getattr(section, "error_message", None)
            rendered = _Rendered(
                _placeholder(
                    plan,
                    "Corresponding report-data section failed: "
                    f"{error_message or 'unknown error'}",
                ),
                ReportFigureStatus.ERROR,
            )
        elif source_status == "skipped":
            rendered = _Rendered(
                _placeholder(
                    plan,
                    "Corresponding report-data section was skipped.",
                ),
                ReportFigureStatus.SKIPPED,
            )
        else:
            try:
                rendered = _render_plan(
                    bundle,
                    plan,
                    registry,
                    config,
                )
            except Exception as exc:
                error_type = type(exc).__name__
                error_message = str(exc)
                rendered = _Rendered(
                    _placeholder(
                        plan,
                        "Static figure generation failed: "
                        f"{error_type}: {error_message}",
                    ),
                    ReportFigureStatus.ERROR,
                )

        artifacts.append(
            ReportFigureArtifact(
                id=plan.figure_id,
                section_id=plan.section_id,
                title=plan.title,
                variant=plan.variant,
                filename=_filename(plan),
                width_px=plan.width_px,
                height_px=plan.height_px,
                status=rendered.status,
                figure=rendered.figure,
                team_name=plan.team_name,
                is_away=(
                    plan.is_away
                    if plan.team_index is not None
                    else None
                ),
                renderer_id=rendered.renderer_id,
                selection_reason=rendered.selection_reason,
                selection=rendered.selection,
                source_section_status=source_status,
                error_type=error_type,
                error_message=error_message,
            )
        )

    return MatchReportFigureCatalog(
        manifest_id=manifest.id,
        manifest_version=manifest.schema_version,
        figures=tuple(artifacts),
    )
