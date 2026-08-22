"""Canonical sequence-outcome contract shared by Match Analysis phases.

The analytical domain uses three independent concepts:
- terminal_outcome: canonical terminal state of the sequence;
- termination_reason: concrete reason why tracing stopped;
- viewpoint: whether the sequence is described from the attacking or defending side.

Legacy ``sequence_outcome_type`` strings are retained only as display/backward-
compatibility labels. Domain logic should use the canonical fields instead.
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


TERMINAL_OUTCOMES = frozenset({
    "goal",
    "shot",
    "turnover",
    "retained",
    "foul",
    "offside",
    "out",
    "consolidated",
    "unknown",
})

VIEWPOINTS = frozenset({
    "attacking",
    "defending",
})

CANONICAL_SEQUENCE_COLUMNS = (
    "terminal_outcome",
    "termination_reason",
    "viewpoint",
)


def _normalize_text(value) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _validate_viewpoint(viewpoint: str) -> str:
    normalized = _normalize_text(viewpoint)
    if normalized not in VIEWPOINTS:
        raise ValueError(
            "viewpoint must be either 'attacking' or 'defending'"
        )
    return normalized


def format_legacy_outcome_label(
    terminal_outcome: str,
    viewpoint: str,
    termination_reason: str = "unknown",
) -> str:
    """Return the compatibility/UI label for a canonical outcome contract."""

    terminal_outcome = _normalize_text(terminal_outcome)
    viewpoint = _validate_viewpoint(viewpoint)
    termination_reason = _normalize_text(termination_reason) or "unknown"

    if terminal_outcome not in TERMINAL_OUTCOMES:
        raise ValueError(
            f"Unsupported terminal_outcome: {terminal_outcome!r}"
        )

    if terminal_outcome == "goal":
        if termination_reason == "penalty_kick_goal":
            return "Penalty Goal"
        if termination_reason == "own_goal":
            return (
                "Own Goal Conceded"
                if viewpoint == "defending"
                else "Forced Own Goal"
            )
        return "Goals conceded" if viewpoint == "defending" else "Goals"

    if terminal_outcome == "shot":
        if termination_reason == "penalty_kick_saved":
            return "Penalty Saved"
        if termination_reason == "penalty_kick_missed":
            return "Penalty Missed"
        return "Shots conceded" if viewpoint == "defending" else "Shots"

    if terminal_outcome == "turnover":
        return (
            "Regained Possessions"
            if viewpoint == "defending"
            else "Lost Possessions"
        )

    if terminal_outcome == "retained":
        if termination_reason == "corner_awarded":
            return "Corner"
        return "Possession Retained"

    if terminal_outcome == "foul":
        if termination_reason == "penalty_awarded":
            return (
                "Penalty conceded"
                if viewpoint == "defending"
                else "Penalty won"
            )
        return "Foul"

    if terminal_outcome == "offside":
        return "Offside"

    if terminal_outcome == "out":
        return "Out"

    if terminal_outcome == "consolidated":
        return (
            "Opponent Possession Consolidated"
            if viewpoint == "defending"
            else "Possession Consolidated"
        )

    return "Unknown"


def make_sequence_outcome(
    terminal_outcome: str,
    termination_reason: str,
    viewpoint: str,
    *,
    legacy_label: str | None = None,
) -> dict:
    """Build a validated canonical outcome payload for sequence rows."""

    terminal_outcome = _normalize_text(terminal_outcome)
    if terminal_outcome not in TERMINAL_OUTCOMES:
        raise ValueError(
            f"Unsupported terminal_outcome: {terminal_outcome!r}"
        )

    viewpoint = _validate_viewpoint(viewpoint)
    termination_reason = (
        _normalize_text(termination_reason)
        or "unknown"
    )

    if legacy_label is None or not _normalize_text(legacy_label):
        legacy_label = format_legacy_outcome_label(
            terminal_outcome,
            viewpoint,
            termination_reason,
        )
    else:
        legacy_label = str(legacy_label).strip()

    return {
        "terminal_outcome": terminal_outcome,
        "termination_reason": termination_reason,
        "viewpoint": viewpoint,
        "sequence_outcome_type": legacy_label,
    }


def canonicalize_legacy_outcome(
    legacy_outcome,
    viewpoint: str,
    *,
    termination_reason: str | None = None,
) -> dict:
    """Map historical outcome strings onto the canonical sequence contract."""

    viewpoint = _validate_viewpoint(viewpoint)
    text = _normalize_text(legacy_outcome)

    terminal_outcome = "unknown"
    inferred_reason = "unknown"

    if text in {"penalty goal"}:
        terminal_outcome = "goal"
        inferred_reason = "penalty_kick_goal"
    elif text in {
        "own goal",
        "own goal conceded",
        "forced own goal",
    }:
        terminal_outcome = "goal"
        inferred_reason = "own_goal"
    elif "goal" in text:
        terminal_outcome = "goal"
        inferred_reason = "goal"
    elif text == "penalty saved":
        terminal_outcome = "shot"
        inferred_reason = "penalty_kick_saved"
    elif text == "penalty missed":
        terminal_outcome = "shot"
        inferred_reason = "penalty_kick_missed"
    elif "shot" in text:
        terminal_outcome = "shot"
        inferred_reason = "shot"
    elif text in {"penalty won", "penalty conceded"}:
        terminal_outcome = "foul"
        inferred_reason = "penalty_awarded"
    elif text == "foul":
        terminal_outcome = "foul"
        inferred_reason = "foul"
    elif text == "offside":
        terminal_outcome = "offside"
        inferred_reason = "offside"
    elif text == "out":
        terminal_outcome = "out"
        inferred_reason = "ball_out"
    elif text in {
        "possession consolidated",
        "opponent possession consolidated",
    }:
        terminal_outcome = "consolidated"
        inferred_reason = "time_window_elapsed"
    elif text in {
        "possession retained",
        "retained",
        "corner",
    }:
        terminal_outcome = "retained"
        inferred_reason = (
            "corner_awarded"
            if text == "corner"
            else "possession_retained"
        )
    elif text in {
        "lost possessions",
        "regained possessions",
        "possession lost",
        "possession regained",
        "turnover",
    }:
        terminal_outcome = "turnover"
        inferred_reason = "possession_turnover"
    elif text in {"", "unknown", "sequence end"}:
        terminal_outcome = "unknown"
        inferred_reason = "unknown"
    elif "big chance" in text:
        # Historical code sometimes used a territorial chance label as if it
        # were terminal. Do not silently reinterpret it as a shot.
        terminal_outcome = "unknown"
        inferred_reason = "legacy_big_chance"

    return make_sequence_outcome(
        terminal_outcome,
        termination_reason or inferred_reason,
        viewpoint,
        legacy_label=(
            None
            if text in {"", "unknown", "sequence end"}
            else legacy_outcome
        ),
    )


def infer_termination_reason(
    sequence_df: pd.DataFrame,
    legacy_outcome=None,
) -> str:
    """Infer a concrete stop reason from the terminal event when possible."""

    legacy_contract = None
    legacy_text = _normalize_text(legacy_outcome)

    if legacy_text in {
        "possession consolidated",
        "opponent possession consolidated",
    }:
        return "time_window_elapsed"

    if legacy_text == "penalty goal":
        return "penalty_kick_goal"
    if legacy_text == "penalty saved":
        return "penalty_kick_saved"
    if legacy_text == "penalty missed":
        return "penalty_kick_missed"
    if legacy_text in {"penalty won", "penalty conceded"}:
        return "penalty_awarded"
    if legacy_text in {
        "own goal",
        "own goal conceded",
        "forced own goal",
    }:
        return "own_goal"
    if legacy_text == "corner":
        return "corner_awarded"

    if sequence_df is None or sequence_df.empty:
        if legacy_text:
            # viewpoint is irrelevant for the inferred reason; attacking is a
            # neutral fallback used only to access the mapping.
            legacy_contract = canonicalize_legacy_outcome(
                legacy_outcome,
                "attacking",
            )
            return legacy_contract["termination_reason"]
        return "unknown"

    last_event = sequence_df.iloc[-1]
    event_type = _normalize_text(last_event.get("type_name"))
    event_outcome = _normalize_text(last_event.get("outcome"))
    is_penalty = last_event.get("Penalty") in [1, "1", True]
    is_own_goal = last_event.get("Own goal") in [1, "1", True]

    if event_type == "goal":
        if is_penalty:
            return "penalty_kick_goal"
        if is_own_goal:
            return "own_goal"
        return "goal"

    if event_type in {"miss", "attempt saved", "post", "shot"}:
        if is_penalty:
            if event_type == "attempt saved":
                return "penalty_kick_saved"
            return "penalty_kick_missed"
        return "shot"

    if event_type == "foul":
        return "penalty_awarded" if is_penalty else "foul"

    if event_type == "offside pass":
        return "offside"

    if event_type == "out":
        return "ball_out"

    if event_type == "corner awarded":
        return "corner_awarded"

    if event_type == "dispossessed":
        return "dispossessed"

    if event_type == "pass" and event_outcome == "unsuccessful":
        return "unsuccessful_pass"

    if event_type == "take on" and event_outcome == "unsuccessful":
        return "failed_take_on"

    if event_type == "ball touch" and event_outcome == "unsuccessful":
        return "failed_control"

    if legacy_text:
        legacy_contract = canonicalize_legacy_outcome(
            legacy_outcome,
            "attacking",
        )
        return legacy_contract["termination_reason"]

    return "unknown"


def apply_sequence_outcome_contract(
    sequence_df: pd.DataFrame,
    viewpoint: str,
    *,
    legacy_outcome=None,
    termination_reason: str | None = None,
) -> pd.DataFrame:
    """Return a sequence whose terminal contract is constant on every row."""

    viewpoint = _validate_viewpoint(viewpoint)

    if sequence_df is None:
        return pd.DataFrame(
            columns=(
                "terminal_outcome",
                "termination_reason",
                "viewpoint",
                "sequence_outcome_type",
            )
        )

    df = sequence_df.copy()
    if df.empty:
        for col in (
            "terminal_outcome",
            "termination_reason",
            "viewpoint",
            "sequence_outcome_type",
        ):
            if col not in df.columns:
                df[col] = pd.Series(dtype="object")
        return df

    if legacy_outcome is None and "sequence_outcome_type" in df.columns:
        legacy_values = (
            df["sequence_outcome_type"]
            .dropna()
            .astype(str)
        )
        if not legacy_values.empty:
            legacy_outcome = legacy_values.iloc[-1]

    if "terminal_outcome" in df.columns:
        canonical_values = (
            df["terminal_outcome"]
            .dropna()
            .astype(str)
            .map(_normalize_text)
        )
        canonical_values = canonical_values[
            canonical_values.isin(TERMINAL_OUTCOMES)
        ]
    else:
        canonical_values = pd.Series(dtype="object")

    inferred_reason = (
        termination_reason
        or infer_termination_reason(
            df,
            legacy_outcome,
        )
    )

    if not canonical_values.empty:
        terminal_outcome = canonical_values.iloc[-1]
        contract = make_sequence_outcome(
            terminal_outcome,
            inferred_reason,
            viewpoint,
            legacy_label=legacy_outcome,
        )
    else:
        contract = canonicalize_legacy_outcome(
            legacy_outcome,
            viewpoint,
            termination_reason=inferred_reason,
        )

    for key, value in contract.items():
        df[key] = value

    return df


def validate_sequence_outcome_contract(
    sequence_df: pd.DataFrame,
) -> list[str]:
    """Return contract violations for one flattened sequence."""

    if sequence_df is None or sequence_df.empty:
        return []

    errors = []

    for column in CANONICAL_SEQUENCE_COLUMNS:
        if column not in sequence_df.columns:
            errors.append(f"missing column: {column}")
            continue

        values = (
            sequence_df[column]
            .dropna()
            .astype(str)
            .map(_normalize_text)
            .unique()
        )

        if len(values) != 1:
            errors.append(
                f"{column} must have exactly one value per sequence"
            )
            continue

        if column == "terminal_outcome" and values[0] not in TERMINAL_OUTCOMES:
            errors.append(
                f"invalid terminal_outcome: {values[0]}"
            )

        if column == "viewpoint" and values[0] not in VIEWPOINTS:
            errors.append(
                f"invalid viewpoint: {values[0]}"
            )

    return errors
