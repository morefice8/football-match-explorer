"""Deterministic selectors for Match Report highlights.

REPORT-03 is deliberately limited to selection.  It consumes already-derived
candidate data and never recalculates canonical football metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class ReportSelection:
    """Neutral, serializable selection result."""

    team_name: str
    category: str
    selected_id: str
    selected_name: str | None
    selection_reason: str
    criteria: tuple[tuple[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["criteria"] = [
            {"criterion": name, "value": value}
            for name, value in self.criteria
        ]
        return payload


_PASSER_CRITERIA = (
    (
        "Offensive Pass Contributions",
        (
            "Offensive Pass Contributions",
            "offensive_pass_contributions",
        ),
        True,
    ),
    (
        "Progressive completed",
        (
            "Progressive Passes",
            "Progressive completed",
            "Progressive Completed",
            "Progressive Passes Completed",
            "progressive_passes",
            "progressive_completed",
            "successful_progressive_passes",
        ),
        True,
    ),
    (
        "Passes into box",
        (
            "Passes into Box",
            "passes_into_box",
            "box_passes",
        ),
        True,
    ),
    (
        "Key passes",
        (
            "Key Passes",
            "key_passes",
        ),
        True,
    ),
    (
        "Completed passes",
        (
            "Completed Passes",
            "Accurate Passes",
            "Successful Passes",
            "completed_passes",
            "successful_passes",
        ),
        True,
    ),
)

_SHOOTING_CRITERIA = (
    (
        "Shot Sequence Involvements",
        (
            "Shot Sequence Involvements",
            "shot_sequence_involvements",
        ),
        True,
    ),
    (
        "Shots",
        (
            "Shot Sequence Shots",
            "Shots",
            "shots",
        ),
        True,
    ),
    (
        "Shot assists",
        (
            "Shot Sequence Shot Assists",
            "Shot Sequence Assists",
            "Shot Assists",
            "shot_assists",
        ),
        True,
    ),
    (
        "Pre-assists",
        (
            "Shot Sequence Pre-Assists",
            "Pre-Assists",
            "Pre assists",
            "pre_assists",
        ),
        True,
    ),
)

_DEFENDER_CRITERIA = (
    (
        "Unique defensive contributions",
        (
            "unique",
            "Unique Defensive Contributions",
            "Defensive Contributions",
            "unique_defensive_contributions",
            "defensive_contributions",
        ),
        True,
    ),
    (
        "Interceptions",
        (
            "Interceptions",
            "interceptions",
        ),
        True,
    ),
    (
        "Tackles won",
        (
            "Tackles Won",
            "Successful Tackles",
            "tackles_won",
            "successful_tackles",
        ),
        True,
    ),
    (
        "Recoveries",
        (
            "Recoveries",
            "Ball Recoveries",
            "recoveries",
            "ball_recoveries",
        ),
        True,
    ),
    (
        "Clearances",
        (
            "Clearances",
            "clearances",
        ),
        True,
    ),
    (
        "Blocks",
        (
            "Blocks",
            "blocks",
        ),
        True,
    ),
    (
        "Fouls",
        (
            "Fouls",
            "Fouls Committed",
            "fouls",
            "fouls_committed",
        ),
        False,
    ),
)

_PLAYER_NAME_ALIASES = (
    "playerName",
    "player_name",
    "Player",
    "player",
    "name",
)

_TEAM_ALIASES = (
    "team_name",
    "teamName",
    "Team",
    "team",
)

_SEQUENCE_ID_ALIASES = (
    "sequence_id",
    "loss_sequence_id",
    "restart_id",
    "id",
)

_TERRITORIAL_PROGRESSION_ALIASES = (
    "max_controlled_x",
    "territorial_progression",
    "territorial_gain",
    "progression",
    "progression_m",
    "x_progression",
    "delta_x",
)

_ACTION_COUNT_ALIASES = (
    "event_count",
    "action_count",
    "actions",
    "num_actions",
    "number_of_actions",
    "sequence_length",
)

_DURATION_ALIASES = (
    "duration_seconds",
    "duration",
    "sequence_duration",
)

_DESTINATION_ALIASES = (
    "destination",
    "Destination",
    "restart_destination",
)

_OUTCOME_ALIASES = (
    "milestone",
    "outcome",
    "final_outcome",
    "terminal_outcome",
    "sequence_outcome_type",
    "sequence_outcome",
    "result",
)

_RESTART_EXECUTION_ALIASES = (
    "restart_execution_outcome",
    "outcome",
    "Outcome",
    "delivery_outcome",
    "restart_outcome",
)

_RESTART_DEVELOPMENT_ALIASES = (
    "restart_development_outcome",
    "development_outcome",
    "Development Outcome",
)

_RESTART_TERMINAL_ALIASES = (
    "terminal_outcome",
    "final_outcome",
    "result",
)


def _to_frame(candidates: Any) -> pd.DataFrame:
    if candidates is None:
        return pd.DataFrame()
    if isinstance(candidates, pd.DataFrame):
        return candidates.copy()
    if isinstance(candidates, Mapping):
        return pd.DataFrame([dict(candidates)])
    if isinstance(candidates, Iterable) and not isinstance(
        candidates,
        (str, bytes),
    ):
        return pd.DataFrame(list(candidates))
    raise TypeError(
        "candidates must be a DataFrame, mapping, iterable of mappings or None"
    )


def _find_column(
    frame: pd.DataFrame,
    aliases: Sequence[str],
) -> str | None:
    by_lower = {
        str(column).strip().casefold(): column
        for column in frame.columns
    }
    for alias in aliases:
        direct = by_lower.get(alias.strip().casefold())
        if direct is not None:
            return direct
    return None


def _value(
    row: pd.Series,
    frame: pd.DataFrame,
    aliases: Sequence[str],
) -> Any:
    column = _find_column(frame, aliases)
    if column is None:
        return None
    value = row.get(column)
    return None if _is_missing(value) else value


def _numeric(
    value: Any,
) -> float | None:
    if _is_missing(value):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(converted):
        return None
    return converted


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, type(pd.NA))) else False


def _scope_to_team(
    frame: pd.DataFrame,
    team_name: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    team_col = _find_column(frame, _TEAM_ALIASES)
    if team_col is None:
        # Canonical player ranking tables can be pre-scoped and indexed by
        # player name without carrying a team column.  In that case REPORT-03
        # treats the supplied candidates as already scoped by the caller.
        return frame.copy()

    mask = (
        frame[team_col]
        .fillna("")
        .astype(str)
        .eq(str(team_name))
    )
    return frame.loc[mask].copy()


def _player_name(
    row: pd.Series,
    frame: pd.DataFrame,
) -> str | None:
    value = _value(row, frame, _PLAYER_NAME_ALIASES)
    if value is None and row.name is not None:
        value = row.name
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _stable_text(value: Any) -> str:
    if _is_missing(value):
        return ""
    return str(value).strip()


def _numeric_rank_component(
    value: Any,
    descending: bool,
) -> tuple[int, float]:
    number = _numeric(value)
    if number is None:
        return (1, 0.0)

    if descending:
        return (0, -number)
    return (0, number)


def _player_sort_key(
    row: pd.Series,
    frame: pd.DataFrame,
    criteria: Sequence[tuple[str, Sequence[str], bool]],
) -> tuple[Any, ...]:
    parts: list[Any] = []
    for _, aliases, descending in criteria:
        parts.extend(
            _numeric_rank_component(
                _value(row, frame, aliases),
                descending,
            )
        )

    name = _player_name(row, frame) or ""
    parts.append(name.casefold())
    parts.append(name)
    return tuple(parts)


def _reason_from_player(
    row: pd.Series,
    frame: pd.DataFrame,
    category: str,
    criteria: Sequence[tuple[str, Sequence[str], bool]],
) -> str:
    ordered = []
    for label, aliases, _ in criteria:
        value = _value(row, frame, aliases)
        ordered.append(
            f"{label}={value!r}" if value is not None else f"{label}=missing"
        )
    ordered.append(f"name={_player_name(row, frame)!r}")
    return (
        f"Selected {category} by deterministic lexicographic order: "
        + " → ".join(ordered)
    )


def _criteria_values(
    row: pd.Series,
    frame: pd.DataFrame,
    criteria: Sequence[tuple[str, Sequence[str], bool]],
) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (label, _value(row, frame, aliases))
        for label, aliases, _ in criteria
    )


def _select_player(
    candidates: Any,
    team_name: str,
    category: str,
    criteria: Sequence[tuple[str, Sequence[str], bool]],
) -> ReportSelection | None:
    frame = _scope_to_team(_to_frame(candidates), team_name)
    if frame.empty:
        return None

    rows = [
        (index, row)
        for index, row in frame.iterrows()
        if _player_name(row, frame)
    ]
    if not rows:
        return None

    _, selected = min(
        rows,
        key=lambda item: _player_sort_key(
            item[1],
            frame,
            criteria,
        ),
    )
    name = _player_name(selected, frame)
    if name is None:
        return None

    return ReportSelection(
        team_name=str(team_name),
        category=category,
        selected_id=name,
        selected_name=name,
        selection_reason=_reason_from_player(
            selected,
            frame,
            category,
            criteria,
        ),
        criteria=_criteria_values(
            selected,
            frame,
            criteria,
        ),
    )


def select_top_passer(
    candidates: Any,
    team_name: str,
) -> ReportSelection | None:
    """Select one top passer for a team using the REPORT-03 tie-break order."""

    return _select_player(
        candidates,
        team_name,
        "top-passer",
        _PASSER_CRITERIA,
    )


def select_top_shooting_contributor(
    candidates: Any,
    team_name: str,
) -> ReportSelection | None:
    """Select one top shooting contributor for a team."""

    return _select_player(
        candidates,
        team_name,
        "top-shooting-contributor",
        _SHOOTING_CRITERIA,
    )


def select_top_defender(
    candidates: Any,
    team_name: str,
) -> ReportSelection | None:
    """Select one top defender for a team."""

    return _select_player(
        candidates,
        team_name,
        "top-defender",
        _DEFENDER_CRITERIA,
    )


def _outcome_text(
    row: pd.Series,
    frame: pd.DataFrame,
    aliases: Sequence[str],
) -> str:
    value = _value(row, frame, aliases)
    return _stable_text(value).casefold()


def _truthy_flag(
    row: pd.Series,
    frame: pd.DataFrame,
    aliases: Sequence[str],
) -> bool:
    value = _value(row, frame, aliases)
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().casefold() in {
            "1",
            "true",
            "yes",
            "y",
            "goal",
            "shot",
            "successful",
            "retained",
        }
    return bool(value)


def _sequence_milestone(
    row: pd.Series,
    frame: pd.DataFrame,
) -> tuple[int, str]:
    outcome = _outcome_text(row, frame, _OUTCOME_ALIASES)

    if _truthy_flag(
        row,
        frame,
        (
            "milestone_goal",
            "produced_goal",
            "is_goal",
            "goal",
            "goal_scored",
        ),
    ) or "goal" in outcome:
        return 5, "goal"

    if _truthy_flag(
        row,
        frame,
        (
            "milestone_shot",
            "produced_shot",
            "has_shot",
            "shot",
            "shot_generated",
        ),
    ) or "shot" in outcome:
        return 4, "shot"

    if _truthy_flag(
        row,
        frame,
        (
            "milestone_box",
            "entered_penalty_area",
            "reached_box",
            "box_entry",
            "entered_box",
        ),
    ) or any(
        token in outcome
        for token in ("box", "penalty area")
    ):
        return 3, "box"

    if _truthy_flag(
        row,
        frame,
        (
            "milestone_final_third",
            "reached_final_third",
            "final_third_entry",
            "entered_final_third",
        ),
    ) or "final third" in outcome or "final_third" in outcome:
        return 2, "final third"

    if _truthy_flag(
        row,
        frame,
        (
            "retained",
            "consolidated",
            "possession_retained",
        ),
    ) or any(
        token in outcome
        for token in (
            "retained",
            "consolidated",
            "possession kept",
            "possession retained",
        )
    ):
        return 1, "retained/consolidated"

    return 0, "no milestone"


def _stable_id(
    row: pd.Series,
    frame: pd.DataFrame,
) -> str | None:
    value = _value(row, frame, _SEQUENCE_ID_ALIASES)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sequence_sort_key(
    row: pd.Series,
    frame: pd.DataFrame,
) -> tuple[Any, ...]:
    milestone_rank, _ = _sequence_milestone(row, frame)
    progression = _value(
        row,
        frame,
        _TERRITORIAL_PROGRESSION_ALIASES,
    )
    action_count = _value(
        row,
        frame,
        _ACTION_COUNT_ALIASES,
    )
    duration = _value(
        row,
        frame,
        _DURATION_ALIASES,
    )
    stable_id = _stable_id(row, frame) or ""

    parts: list[Any] = [-milestone_rank]
    parts.extend(_numeric_rank_component(progression, True))
    parts.extend(_numeric_rank_component(action_count, True))
    parts.extend(_numeric_rank_component(duration, True))
    parts.extend((stable_id.casefold(), stable_id))
    return tuple(parts)


def select_representative_sequence(
    candidates: Any,
    team_name: str,
    *,
    category: str,
) -> ReportSelection | None:
    """Select one build-up or transition sequence for a team.

    Valid categories are intentionally explicit so report callers cannot
    accidentally reuse this selector for a different ranking contract.
    """

    allowed = {
        "build-up",
        "defensive-transition",
        "offensive-transition",
    }
    if category not in allowed:
        raise ValueError(
            "category must be one of: " + ", ".join(sorted(allowed))
        )

    frame = _scope_to_team(_to_frame(candidates), team_name)
    if frame.empty:
        return None

    rows = [
        row
        for _, row in frame.iterrows()
        if _stable_id(row, frame)
    ]
    if not rows:
        return None

    selected = min(
        rows,
        key=lambda row: _sequence_sort_key(row, frame),
    )
    selected_id = _stable_id(selected, frame)
    if selected_id is None:
        return None

    milestone_rank, milestone = _sequence_milestone(
        selected,
        frame,
    )
    progression = _value(
        selected,
        frame,
        _TERRITORIAL_PROGRESSION_ALIASES,
    )
    action_count = _value(
        selected,
        frame,
        _ACTION_COUNT_ALIASES,
    )
    duration = _value(
        selected,
        frame,
        _DURATION_ALIASES,
    )

    criteria = (
        ("milestone", milestone),
        ("territorial_progression", progression),
        ("action_count", action_count),
        ("duration", duration),
        ("stable_id", selected_id),
    )
    reason = (
        f"Selected {category} by deterministic lexicographic order: "
        f"milestone={milestone!r} (priority {milestone_rank})"
        f" → territorial_progression={progression!r}"
        f" → action_count={action_count!r}"
        f" → duration={duration!r}"
        f" → stable_id={selected_id!r}"
    )

    return ReportSelection(
        team_name=str(team_name),
        category=category,
        selected_id=selected_id,
        selected_name=None,
        selection_reason=reason,
        criteria=criteria,
    )


def _restart_text_values(
    row: pd.Series,
    frame: pd.DataFrame,
    aliases: Sequence[str],
) -> tuple[str, ...]:
    values: list[str] = []
    seen = set()

    for alias in aliases:
        column = _find_column(frame, (alias,))
        if column is None:
            continue
        text = _stable_text(row.get(column)).casefold()
        if text and text not in seen:
            values.append(text)
            seen.add(text)

    return tuple(values)


def _restart_priority(
    row: pd.Series,
    frame: pd.DataFrame,
) -> tuple[int, str]:
    execution = " ".join(
        _restart_text_values(
            row,
            frame,
            _RESTART_EXECUTION_ALIASES,
        )
    )
    development = " ".join(
        _restart_text_values(
            row,
            frame,
            _RESTART_DEVELOPMENT_ALIASES,
        )
    )
    terminal = " ".join(
        _restart_text_values(
            row,
            frame,
            _RESTART_TERMINAL_ALIASES,
        )
    )
    all_text = " ".join(
        value
        for value in (development, execution, terminal)
        if value
    )

    if _truthy_flag(
        row,
        frame,
        ("is_goal", "goal", "goal_scored"),
    ) or "goal" in all_text:
        return 4, "goal"

    if _truthy_flag(
        row,
        frame,
        ("has_shot", "shot", "shot_generated"),
    ) or "shot" in development or "shot" in terminal:
        return 3, "shot"

    if _truthy_flag(
        row,
        frame,
        (
            "successful_delivery",
            "delivery_successful",
            "successful_cross",
        ),
    ) or any(
        token in execution
        for token in (
            "successful delivery",
            "delivery successful",
        )
    ):
        return 2, "successful delivery"

    if _truthy_flag(
        row,
        frame,
        (
            "retained_development",
            "retained",
            "possession_retained",
        ),
    ) or any(
        token in development or token in terminal
        for token in (
            "retained development",
            "retained",
            "consolidated",
        )
    ):
        return 1, "retained development"

    return 0, "no milestone"


def _restart_destination_quality(
    row: pd.Series,
    frame: pd.DataFrame,
) -> tuple[int, str | None]:
    """Return an ordinal quality band from canonical destination taxonomy.

    The rank is only a deterministic category order, not a weighted score and
    not a new football metric.  It consumes the existing restart destination
    label produced by the canonical set-piece/restart adapters.
    """

    raw = _value(row, frame, _DESTINATION_ALIASES)
    if raw is None:
        return 0, None

    text = _stable_text(raw)
    normalized = text.casefold()

    if any(
        token in normalized
        for token in (
            "6-yard",
            "6 yard",
            "six-yard",
            "six yard",
            "near post",
            "far post",
        )
    ):
        return 5, text

    if any(
        token in normalized
        for token in (
            "center box",
            "centre box",
            "penalty area",
        )
    ):
        return 4, text

    if "final third" in normalized:
        return 3, text

    if "middle third" in normalized:
        return 2, text

    if "own third" in normalized:
        return 1, text

    return 0, text


def _restart_sort_key(
    row: pd.Series,
    frame: pd.DataFrame,
) -> tuple[Any, ...]:
    priority, _ = _restart_priority(row, frame)
    destination_rank, _ = _restart_destination_quality(row, frame)
    stable_id = _stable_id(row, frame) or ""

    return (
        -priority,
        -destination_rank,
        stable_id.casefold(),
        stable_id,
    )


def select_representative_restart(
    candidates: Any,
    team_name: str,
) -> ReportSelection | None:
    """Select one representative restart for a team."""

    frame = _scope_to_team(_to_frame(candidates), team_name)
    if frame.empty:
        return None

    rows = [
        row
        for _, row in frame.iterrows()
        if _stable_id(row, frame)
    ]
    if not rows:
        return None

    selected = min(
        rows,
        key=lambda row: _restart_sort_key(row, frame),
    )
    selected_id = _stable_id(selected, frame)
    if selected_id is None:
        return None

    priority, milestone = _restart_priority(
        selected,
        frame,
    )
    destination_rank, destination = _restart_destination_quality(
        selected,
        frame,
    )

    criteria = (
        ("restart_priority", milestone),
        ("destination", destination),
        ("destination_quality_rank", destination_rank),
        ("stable_id", selected_id),
    )
    reason = (
        "Selected restart by deterministic lexicographic order: "
        f"priority={milestone!r} ({priority})"
        f" → destination={destination!r}"
        f" (quality band {destination_rank})"
        f" → stable_id={selected_id!r}"
    )

    return ReportSelection(
        team_name=str(team_name),
        category="restart",
        selected_id=selected_id,
        selected_name=None,
        selection_reason=reason,
        criteria=criteria,
    )

