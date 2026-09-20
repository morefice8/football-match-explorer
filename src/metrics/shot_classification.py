"""Canonical Opta shot classification shared by Match Analysis and reporting.

The Opta event taxonomy uses ``Attempt Saved`` both for genuine goalkeeper
saves and for blocked shots. Qualifier 82 (``Blocked``) disambiguates the
latter. This module centralises that interpretation so UI and pack exports do
not drift apart.

The classifier operates on already-preprocessed Opta rows. It deliberately
keeps nullable boolean audit fields: ``pd.NA`` means not applicable / unknown,
which is distinct from an explicit ``False`` on a recognised shot event.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import pandas as pd


SHOT_EVENT_TYPES = frozenset({"Goal", "Miss", "Attempt Saved", "Post"})

SHOT_OUTCOME_GOAL = "goal"
SHOT_OUTCOME_SAVED = "saved"
SHOT_OUTCOME_BLOCKED = "blocked"
SHOT_OUTCOME_OFF_TARGET = "off_target"
SHOT_OUTCOME_POST = "post"
SHOT_OUTCOME_OWN_GOAL = "own_goal"
SHOT_OUTCOME_UNKNOWN = "unknown"

SHOT_OUTCOMES = (
    SHOT_OUTCOME_GOAL,
    SHOT_OUTCOME_SAVED,
    SHOT_OUTCOME_BLOCKED,
    SHOT_OUTCOME_OFF_TARGET,
    SHOT_OUTCOME_POST,
    SHOT_OUTCOME_OWN_GOAL,
    SHOT_OUTCOME_UNKNOWN,
)

# Qualifier names are the canonical names from optaQualifierCodes.json after
# preprocess.process_opta_events(). Raw qualifier aliases are retained as a
# defensive fallback for partially-normalised synthetic inputs.
_BLOCKED_QUALIFIER_ALIASES = (
    "Blocked",
    "qualifier_82",
)
_BLOCK_CONTEXT_FLAG_ALIASES = (
    "Temp_Blocked",
    "Own shot blocked",
    "Def block",
    "By Wall",
    "Block by hand",
    "qualifier_250",
    "qualifier_228",
    "qualifier_94",
    "qualifier_239",
    "qualifier_192",
)
_BLOCK_CONTEXT_VALUE_ALIASES = (
    "Blocked x co-ordinate",
    "Blocked y co-ordinate",
    "qualifier_146",
    "qualifier_147",
)
_OWN_GOAL_ALIASES = (
    "Own goal",
    "qualifier_28",
)
_OFF_TARGET_SAVE_ALIASES = (
    "Keeper Saved",
    "From shot off target",
    "qualifier_137",
    "qualifier_190",
)
_WOODWORK_ALIASES = (
    "Hit Woodwork",
    "qualifier_138",
)


@dataclass(frozen=True)
class ShotClassification:
    outcome: str | None
    counts_as_shot: bool
    on_target: bool | None
    blocked: bool | None
    own_goal: bool | None
    issue: str | None = None


def _normalise_scalar(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _truthy_value(value: Any) -> bool:
    value = _normalise_scalar(value)
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().casefold()
    if not text:
        return False
    if text in {"0", "0.0", "false", "no", "n", "none", "nan", "<na>"}:
        return False
    return True


def _has_truthy(row: Mapping[str, Any], aliases: Iterable[str]) -> bool:
    return any(alias in row and _truthy_value(row.get(alias)) for alias in aliases)


def _has_value(row: Mapping[str, Any], aliases: Iterable[str]) -> bool:
    for alias in aliases:
        if alias not in row:
            continue
        value = _normalise_scalar(row.get(alias))
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return True
    return False


def classify_shot_event(row: Mapping[str, Any]) -> ShotClassification:
    """Classify one preprocessed Opta event.

    Semantics used here mirror the mapping files shipped with the repository:
    - Goal -> on target, unless qualifier 28 marks an own goal;
    - Miss -> off target;
    - Post -> woodwork and not on target;
    - Attempt Saved -> saved/on target unless qualifier 82 (or equivalent
      blocking evidence) marks a blocked shot;
    - qualifier 137 / 190 marks a saved/collected attempt that was actually
      travelling off target;
    - contradictory blocked + off-target-save evidence is kept as ``unknown``
      rather than silently choosing one interpretation.
    """

    event_type = str(row.get("type_name") or row.get("type") or "").strip()
    if event_type not in SHOT_EVENT_TYPES:
        return ShotClassification(
            outcome=None,
            counts_as_shot=False,
            on_target=None,
            blocked=None,
            own_goal=None,
        )

    own_goal = _has_truthy(row, _OWN_GOAL_ALIASES)
    blocked = _has_truthy(row, _BLOCKED_QUALIFIER_ALIASES)
    off_target_save = _has_truthy(row, _OFF_TARGET_SAVE_ALIASES)
    woodwork = _has_truthy(row, _WOODWORK_ALIASES)

    # Opta event 15 (Attempt Saved) uses qualifier 82 as the canonical
    # shooter-side discriminator for blocked shots. Related block context
    # qualifiers and blocked x/y coordinates are supporting metadata only and
    # must not independently turn an Attempt Saved into a blocked shot.
    auxiliary_block_context = (
        _has_truthy(row, _BLOCK_CONTEXT_FLAG_ALIASES)
        or _has_value(row, _BLOCK_CONTEXT_VALUE_ALIASES)
    )

    if own_goal:
        return ShotClassification(
            outcome=SHOT_OUTCOME_OWN_GOAL,
            counts_as_shot=False,
            on_target=None,
            blocked=bool(blocked),
            own_goal=True,
            issue=None,
        )

    if event_type == "Goal":
        return ShotClassification(
            outcome=SHOT_OUTCOME_GOAL,
            counts_as_shot=True,
            on_target=True,
            blocked=bool(blocked),
            own_goal=False,
            issue=None,
        )

    if event_type == "Post" or woodwork:
        return ShotClassification(
            outcome=SHOT_OUTCOME_POST,
            counts_as_shot=True,
            on_target=False,
            blocked=bool(blocked),
            own_goal=False,
            issue=(
                "unexpected_blocked_post"
                if blocked
                else (
                    "woodwork_qualifier_on_non_post_event"
                    if event_type != "Post"
                    else None
                )
            ),
        )

    if event_type == "Miss":
        if blocked:
            return ShotClassification(
                outcome=SHOT_OUTCOME_UNKNOWN,
                counts_as_shot=True,
                on_target=None,
                blocked=True,
                own_goal=False,
                issue="conflicting_miss_blocked",
            )
        return ShotClassification(
            outcome=SHOT_OUTCOME_OFF_TARGET,
            counts_as_shot=True,
            on_target=False,
            blocked=False,
            own_goal=False,
            issue=None,
        )

    # Attempt Saved is the only Opta shot event requiring qualifier-based
    # disambiguation. Qualifier 82 explicitly means the shot was blocked.
    if blocked and off_target_save:
        return ShotClassification(
            outcome=SHOT_OUTCOME_UNKNOWN,
            counts_as_shot=True,
            on_target=None,
            blocked=True,
            own_goal=False,
            issue="conflicting_blocked_off_target_save",
        )

    if blocked:
        return ShotClassification(
            outcome=SHOT_OUTCOME_BLOCKED,
            counts_as_shot=True,
            on_target=False,
            blocked=True,
            own_goal=False,
            issue=None,
        )

    if off_target_save:
        return ShotClassification(
            outcome=SHOT_OUTCOME_OFF_TARGET,
            counts_as_shot=True,
            on_target=False,
            blocked=False,
            own_goal=False,
            issue="keeper_touched_off_target_attempt",
        )

    return ShotClassification(
        outcome=SHOT_OUTCOME_SAVED,
        counts_as_shot=True,
        on_target=True,
        blocked=False,
        own_goal=False,
        issue=(
            "auxiliary_block_context_without_blocked_qualifier"
            if auxiliary_block_context
            else None
        ),
    )


def classify_shots(frame: pd.DataFrame) -> pd.DataFrame:
    """Return canonical shot rows enriched with shared classification fields."""

    if frame is None or frame.empty:
        empty = pd.DataFrame(columns=list(getattr(frame, "columns", ())))
        for column in (
            "shot_outcome",
            "shot_counts_as_shot",
            "shot_on_target",
            "shot_blocked",
            "shot_own_goal",
            "shot_blocked_qualifier",
            "shot_keeper_saved_off_target",
            "shot_hit_woodwork",
            "shot_own_goal_qualifier",
            "shot_classification_issue",
        ):
            empty[column] = pd.Series(dtype="object")
        return empty

    event_column = "type_name" if "type_name" in frame.columns else "type"
    if event_column not in frame.columns:
        return pd.DataFrame(columns=list(frame.columns) + [
            "shot_outcome",
            "shot_counts_as_shot",
            "shot_on_target",
            "shot_blocked",
            "shot_own_goal",
            "shot_blocked_qualifier",
            "shot_keeper_saved_off_target",
            "shot_hit_woodwork",
            "shot_own_goal_qualifier",
            "shot_classification_issue",
        ])

    shots = frame.loc[frame[event_column].isin(SHOT_EVENT_TYPES)].copy()
    if shots.empty:
        for column in (
            "shot_outcome",
            "shot_counts_as_shot",
            "shot_on_target",
            "shot_blocked",
            "shot_own_goal",
            "shot_blocked_qualifier",
            "shot_keeper_saved_off_target",
            "shot_hit_woodwork",
            "shot_own_goal_qualifier",
            "shot_classification_issue",
        ):
            shots[column] = pd.Series(index=shots.index, dtype="object")
        return shots

    classified = [classify_shot_event(row) for row in shots.to_dict("records")]
    shots["shot_outcome"] = [item.outcome for item in classified]
    shots["shot_counts_as_shot"] = pd.Series(
        [item.counts_as_shot for item in classified],
        index=shots.index,
        dtype="boolean",
    )
    shots["shot_on_target"] = pd.Series(
        [item.on_target for item in classified],
        index=shots.index,
        dtype="boolean",
    )
    shots["shot_blocked"] = pd.Series(
        [item.blocked for item in classified],
        index=shots.index,
        dtype="boolean",
    )
    shots["shot_own_goal"] = pd.Series(
        [item.own_goal for item in classified],
        index=shots.index,
        dtype="boolean",
    )

    records = shots.to_dict("records")
    shots["shot_blocked_qualifier"] = pd.Series(
        [
            _has_truthy(row, _BLOCKED_QUALIFIER_ALIASES)
            for row in records
        ],
        index=shots.index,
        dtype="boolean",
    )
    shots["shot_keeper_saved_off_target"] = pd.Series(
        [
            _has_truthy(row, _OFF_TARGET_SAVE_ALIASES)
            for row in records
        ],
        index=shots.index,
        dtype="boolean",
    )
    shots["shot_hit_woodwork"] = pd.Series(
        [
            _has_truthy(row, _WOODWORK_ALIASES)
            for row in records
        ],
        index=shots.index,
        dtype="boolean",
    )
    shots["shot_own_goal_qualifier"] = pd.Series(
        [
            _has_truthy(row, _OWN_GOAL_ALIASES)
            for row in records
        ],
        index=shots.index,
        dtype="boolean",
    )
    shots["shot_classification_issue"] = [item.issue for item in classified]
    return shots


def shot_stats_for_team(shots: pd.DataFrame, team_name: str) -> dict[str, int]:
    """Aggregate canonical shot classes for one team."""

    if shots is None or shots.empty or "team_name" not in shots.columns:
        return {
            "goals": 0,
            "total_shots": 0,
            "shots_on_target": 0,
            "saved_shots": 0,
            "blocked_shots": 0,
            "off_target_shots": 0,
            "woodwork_shots": 0,
            "unknown_shots": 0,
            "own_goals": 0,
        }

    team = shots.loc[shots["team_name"].eq(team_name)].copy()
    outcomes = team.get("shot_outcome", pd.Series(index=team.index, dtype="object"))
    counts_as_shot = team.get(
        "shot_counts_as_shot",
        pd.Series(False, index=team.index, dtype="boolean"),
    ).fillna(False)
    on_target = team.get(
        "shot_on_target",
        pd.Series(pd.NA, index=team.index, dtype="boolean"),
    )

    return {
        "goals": int(outcomes.eq(SHOT_OUTCOME_GOAL).sum()),
        "total_shots": int(counts_as_shot.sum()),
        "shots_on_target": int(on_target.eq(True).sum()),
        "saved_shots": int(outcomes.eq(SHOT_OUTCOME_SAVED).sum()),
        "blocked_shots": int(outcomes.eq(SHOT_OUTCOME_BLOCKED).sum()),
        "off_target_shots": int(outcomes.eq(SHOT_OUTCOME_OFF_TARGET).sum()),
        "woodwork_shots": int(outcomes.eq(SHOT_OUTCOME_POST).sum()),
        "unknown_shots": int(outcomes.eq(SHOT_OUTCOME_UNKNOWN).sum()),
        "own_goals": int(outcomes.eq(SHOT_OUTCOME_OWN_GOAL).sum()),
    }
