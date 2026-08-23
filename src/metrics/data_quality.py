"""
Shared REL-10 data-quality / inference-coverage metrics.

This module measures observability and inference coverage. It does not silently
repair data and it does not decide football semantics.

A warning is produced only when the caller supplies a configured threshold.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


UNKNOWN_TEXT_VALUES = frozenset({
    "",
    "unknown",
    "none",
    "nan",
    "null",
    "n/a",
    "na",
})


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator) * 100.0


def _is_known_value(value) -> bool:
    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass

    return str(value).strip().lower() not in UNKNOWN_TEXT_VALUES


def _successful_mask(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=bool)

    if "outcome" not in df.columns:
        return pd.Series(True, index=df.index, dtype=bool)

    outcome = df["outcome"]
    text = outcome.fillna("").astype(str).str.strip().str.lower()
    numeric = pd.to_numeric(outcome, errors="coerce")
    return text.eq("successful") | numeric.eq(1)


def receiver_coverage(passes_df: Optional[pd.DataFrame]) -> dict:
    """Coverage of reliable receiver inference among successful passes."""
    if passes_df is None or passes_df.empty:
        return {
            "eligible": 0,
            "resolved": 0,
            "unresolved": 0,
            "high": 0,
            "medium": 0,
            "coverage_pct": 0.0,
        }

    eligible = passes_df.loc[_successful_mask(passes_df)].copy()

    if "receiver_is_reliable" in eligible.columns:
        reliable = (
            eligible["receiver_is_reliable"]
            .fillna(False)
            .astype(bool)
        )
    else:
        reliable = pd.Series(False, index=eligible.index, dtype=bool)

    confidence = eligible.get(
        "receiver_confidence",
        pd.Series(index=eligible.index, dtype="object"),
    )

    resolved = int(reliable.sum())
    total = int(len(eligible))

    return {
        "eligible": total,
        "resolved": resolved,
        "unresolved": max(total - resolved, 0),
        "high": int(confidence.eq("high").sum()),
        "medium": int(confidence.eq("medium").sum()),
        "coverage_pct": _pct(resolved, total),
    }


def coordinate_coverage(
    df: Optional[pd.DataFrame],
    columns: Sequence[str] = ("x", "y"),
) -> dict:
    """
    Count rows with all required coordinates present and finite.

    Missing required columns make every row invalid rather than being ignored.
    """
    if df is None or df.empty:
        return {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "coverage_pct": 0.0,
            "columns": tuple(columns),
            "missing_columns": tuple(),
        }

    total = int(len(df))
    missing_columns = [
        column for column in columns
        if column not in df.columns
    ]

    if missing_columns:
        valid = 0
    else:
        numeric = pd.DataFrame(
            {
                column: pd.to_numeric(df[column], errors="coerce")
                for column in columns
            },
            index=df.index,
        )
        finite = np.isfinite(
            numeric.to_numpy(dtype=float)
        ).all(axis=1)
        valid = int(finite.sum())

    return {
        "total": total,
        "valid": valid,
        "invalid": max(total - valid, 0),
        "coverage_pct": _pct(valid, total),
        "columns": tuple(columns),
        "missing_columns": tuple(missing_columns),
    }


def outcome_coverage(
    df: Optional[pd.DataFrame],
    column: str = "outcome",
) -> dict:
    """Report known/unknown values for an event- or sequence-outcome column."""
    if df is None or df.empty:
        return {
            "total": 0,
            "known": 0,
            "unknown": 0,
            "known_pct": 0.0,
            "unknown_pct": 0.0,
            "column": column,
        }

    total = int(len(df))

    if column not in df.columns:
        known = 0
    else:
        known = int(df[column].map(_is_known_value).sum())

    unknown = max(total - known, 0)

    return {
        "total": total,
        "known": known,
        "unknown": unknown,
        "known_pct": _pct(known, total),
        "unknown_pct": _pct(unknown, total),
        "column": column,
    }


def carry_candidate_coverage(
    *,
    candidates: int,
    included: int,
    excluded: Optional[int] = None,
) -> dict:
    """
    Summarise inferred carry candidates.

    A low inclusion percentage is not automatically poor data: conservative
    inference is expected to reject ambiguous candidates.
    """
    candidates = max(int(candidates or 0), 0)
    included = max(int(included or 0), 0)

    if excluded is None:
        excluded = max(candidates - included, 0)
    else:
        excluded = max(int(excluded or 0), 0)

    return {
        "candidates": candidates,
        "included": included,
        "excluded": excluded,
        "inclusion_pct": _pct(included, candidates),
        "exclusion_pct": _pct(excluded, candidates),
    }


def carry_candidate_coverage_from_stats(stats: Optional[Mapping]) -> dict:
    stats = stats or {}

    candidates = int(stats.get("carry_entry_candidates", 0) or 0)
    included = int(stats.get("carry_entries", 0) or 0)

    raw_excluded = stats.get("carry_entries_excluded_total", None)
    excluded = (
        int(raw_excluded or 0)
        if raw_excluded is not None
        else None
    )

    return carry_candidate_coverage(
        candidates=candidates,
        included=included,
        excluded=excluded,
    )


def sequence_retention(
    *,
    candidates: int,
    built: int,
) -> dict:
    """
    Candidate trigger/loss events versus valid reconstructed sequences.

    "discarded" means a candidate did not produce a valid non-empty sequence
    under the current analysis contract. It is unrelated to UI filters.
    """
    candidates = max(int(candidates or 0), 0)
    built = max(int(built or 0), 0)

    if candidates:
        built = min(built, candidates)

    discarded = max(candidates - built, 0)

    return {
        "candidates": candidates,
        "built": built,
        "discarded": discarded,
        "retention_pct": _pct(built, candidates),
        "discarded_pct": _pct(discarded, candidates),
    }


def sequence_outcome_coverage(
    sequence_df: Optional[pd.DataFrame],
    *,
    sequence_id_column: str,
    outcome_column: str = "sequence_outcome_type",
) -> dict:
    """Count unknown outcomes once per sequence, not once per event row."""
    if (
        sequence_df is None
        or sequence_df.empty
        or sequence_id_column not in sequence_df.columns
    ):
        return {
            "total": 0,
            "known": 0,
            "unknown": 0,
            "known_pct": 0.0,
            "unknown_pct": 0.0,
            "column": outcome_column,
        }

    summary = sequence_df.drop_duplicates(
        subset=[sequence_id_column],
        keep="last",
    )

    return outcome_coverage(
        summary,
        column=outcome_column,
    )


def threshold_status(
    metric_key: str,
    value: float,
    thresholds: Optional[Mapping],
) -> str:
    """
    Return neutral, ok or warning.

    No threshold -> neutral.
    Supported schemas:
        {"min": 70.0}
        {"max": 5.0}
        {"min": 70.0, "max": 99.0}
    """
    thresholds = thresholds or {}
    rule = thresholds.get(metric_key)

    if rule is None:
        return "neutral"

    if isinstance(rule, (int, float)):
        rule = {"min": float(rule)}

    if not isinstance(rule, Mapping):
        return "neutral"

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "neutral"

    if not math.isfinite(numeric_value):
        return "neutral"

    min_value = rule.get("min")
    max_value = rule.get("max")

    if min_value is not None and numeric_value < float(min_value):
        return "warning"

    if max_value is not None and numeric_value > float(max_value):
        return "warning"

    return "ok"


def coverage_item(
    *,
    key: str,
    label: str,
    value: str,
    detail: str,
    threshold_value: Optional[float] = None,
    thresholds: Optional[Mapping] = None,
) -> dict:
    """Stable UI-ready representation of one coverage indicator."""
    status = (
        threshold_status(
            key,
            threshold_value,
            thresholds,
        )
        if threshold_value is not None
        else "neutral"
    )

    return {
        "key": key,
        "label": label,
        "value": value,
        "detail": detail,
        "status": status,
    }


def has_warning(items: Iterable[Mapping]) -> bool:
    return any(
        item.get("status") == "warning"
        for item in items
    )

def attach_sequence_coverage(
    sequence_df: Optional[pd.DataFrame],
    *,
    candidates: int,
    sequence_id_column: str,
) -> pd.DataFrame:
    """
    Attach detector-level candidate/built/discarded metadata to a DataFrame.

    Metadata lives in DataFrame.attrs so the event schema stays unchanged.
    Candidates are measured before UI filtering. Therefore "discarded" means
    detector rejection, never a sequence hidden by an explorer filter.
    """
    if sequence_df is None:
        sequence_df = pd.DataFrame()

    candidates = max(int(candidates or 0), 0)

    if (
        not sequence_df.empty
        and sequence_id_column in sequence_df.columns
    ):
        built = int(
            sequence_df[sequence_id_column].nunique(dropna=True)
        )
    else:
        built = 0

    coverage = sequence_retention(
        candidates=candidates,
        built=built,
    )

    sequence_df.attrs["data_coverage"] = {
        "sequence_candidates": coverage["candidates"],
        "sequence_built": coverage["built"],
        "sequence_discarded": coverage["discarded"],
        "sequence_retention_pct": coverage["retention_pct"],
    }

    return sequence_df
