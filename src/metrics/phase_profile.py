"""Phase profile: how a team attacks after winning the ball and how it
reacts after losing it, split by the third of the pitch where it happened.

This module adds no new football detection. It re-reads the transition
sequences the pipeline already produces (``find_buildup_after_possession_loss``)
and condenses each one into a single auditable row, then summarises those rows
by pitch third. The same summary functions work on one match or on several
matches concatenated, so multi-match scouting can reuse them unchanged.

Definitions (all explicit and tunable via the constants below):

* Recovery sequence (team wins the ball): zone = third where it won the ball.
  - ``first_pass_forward``: the first pass gains at least
    ``FORWARD_PASS_MIN_DELTA_X`` pitch units towards the opponent goal.
  - ``reached_final_third``: the team controls the ball beyond ``FINAL_THIRD_X``
    using the app's own "controlled location" rule (every event start counts,
    an end location only when the event succeeded); only defined when the
    recovery zone is not already the attacking third.
  - ``seconds_to_final_third``: time from the recovery to that first moment.
  - ``gain_m``: furthest point reached minus the recovery point, in metres.
* Loss sequence (team loses the ball): zone = third where it lost the ball.
  - ``regained``: the opponent possession ended with a turnover inside the
    pipeline's transition window.
  - ``seconds_to_regain``: from the loss to the end of the opponent possession.
  - ``opponent_reached_final_third``: the opponent got the ball beyond
    ``FINAL_THIRD_X`` (in its own attacking direction) before the sequence ended.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.metrics.sequence_outcome_metrics import calculate_sequence_milestones

FINAL_THIRD_X = 66.67
DEFENSIVE_THIRD_X = 33.33
FORWARD_PASS_MIN_DELTA_X = 10.0
PITCH_LENGTH_M = 105.0
QUICK_REGAIN_SECONDS = 5.0
REGAIN_WINDOW_SECONDS = 10.0
LOW_SAMPLE_THRESHOLD = 10

ZONES = ("Defensive Third", "Middle Third", "Attacking Third")

RECOVERY_SEQUENCE_COLUMNS = [
    "sequence_id", "start_zone", "start_x", "start_y", "duration_s",
    "completed_passes", "first_pass_forward", "reached_final_third",
    "seconds_to_final_third", "gain_m", "terminal_outcome", "produced_shot",
]
LOSS_SEQUENCE_COLUMNS = [
    "sequence_id", "loss_zone", "loss_x", "loss_y", "initial_loss_type",
    "regained", "seconds_to_regain", "opponent_passes",
    "opponent_reached_final_third", "terminal_outcome", "opponent_shot",
]
RECOVERY_SUMMARY_COLUMNS = [
    "zone", "n", "low_sample", "first_pass_forward_pct", "reached_final_third_pct",
    "median_seconds_to_final_third", "shot_pct", "avg_completed_passes",
    "median_gain_m", "median_duration_s",
]
LOSS_SUMMARY_COLUMNS = [
    "zone", "n", "low_sample", "regained_pct", "regained_within_5s_pct",
    "regained_within_10s_pct", "median_seconds_to_regain",
    "opponent_final_third_pct", "opponent_shot_pct",
]


def zone_from_x(x) -> str | None:
    """Pitch third from a team-relative x (own goal at 0, attacking at 100)."""
    if x is None or pd.isna(x):
        return None
    if x < DEFENSIVE_THIRD_X:
        return ZONES[0]
    if x < FINAL_THIRD_X:
        return ZONES[1]
    return ZONES[2]


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _first_valid(frame: pd.DataFrame, column: str):
    if column not in frame.columns:
        return None
    values = frame[column].dropna()
    return values.iloc[0] if len(values) else None


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    if "total_seconds" in frame.columns:
        return frame.assign(_t=_num(frame, "total_seconds")).sort_values(
            "_t", kind="stable"
        )
    return frame.assign(_t=np.arange(len(frame), dtype=float))


def _type(frame: pd.DataFrame) -> pd.Series:
    return frame["type_name"] if "type_name" in frame.columns else pd.Series("", index=frame.index)


def _controlled_points(frame: pd.DataFrame) -> pd.DataFrame:
    """(time, x) of every controlled location, same rule as the app milestones."""
    x = _num(frame, "x")
    end_x = _num(frame, "end_x")
    ok = frame["outcome"].eq("Successful") if "outcome" in frame.columns else pd.Series(False, index=frame.index)
    starts = pd.DataFrame({"t": frame["_t"], "cx": x}).dropna()
    ends = pd.DataFrame({"t": frame["_t"][ok], "cx": end_x[ok]}).dropna()
    return pd.concat([starts, ends]).sort_values("t", kind="stable")


def _reach_time(frame: pd.DataFrame, start_t: float):
    """Time from ``start_t`` to the first controlled location beyond the line."""
    pts = _controlled_points(frame)
    beyond = pts[pts["cx"] >= FINAL_THIRD_X]
    if beyond.empty:
        return None
    return float(beyond["t"].iloc[0] - start_t)


def recovery_sequence_row(sequence: pd.DataFrame) -> dict | None:
    """One row describing what a team did after winning the ball."""
    if sequence is None or sequence.empty:
        return None
    seq = _ordered(sequence)
    x = _num(seq, "x")
    first = seq[x.notna()].head(1)
    if first.empty:
        return None
    start_x = float(first["x"].iloc[0])
    start_y = float(pd.to_numeric(first["y"], errors="coerce").iloc[0]) if "y" in first else np.nan
    start_t = float(first["_t"].iloc[0])
    zone = _first_valid(seq, "loss_zone") or zone_from_x(start_x)

    is_pass = _type(seq).eq("Pass")
    passes = seq[is_pass]
    completed = int((passes["outcome"] == "Successful").sum()) if "outcome" in passes else 0
    first_pass_forward = None
    if len(passes):
        p = passes.iloc[0]
        delta = pd.to_numeric(p.get("end_x"), errors="coerce") - pd.to_numeric(p.get("x"), errors="coerce")
        if pd.notna(delta):
            first_pass_forward = bool(delta >= FORWARD_PASS_MIN_DELTA_X)

    milestones = calculate_sequence_milestones(sequence)
    reached = None
    seconds_to = None
    if zone != ZONES[2]:
        reached = bool(milestones["reached_final_third"])
        seconds_to = _reach_time(seq, start_t) if reached else None

    furthest = _controlled_points(seq)["cx"]
    gain_units = max(0.0, float(furthest.max()) - start_x) if len(furthest) else 0.0
    terminal = _first_valid(seq, "terminal_outcome")
    duration = max(0.0, float(seq["_t"].max() - start_t))
    return {
        "sequence_id": _first_valid(seq, "loss_sequence_id"),
        "start_zone": zone,
        "start_x": round(start_x, 1),
        "start_y": None if pd.isna(start_y) else round(start_y, 1),
        "duration_s": round(duration, 1),
        "completed_passes": completed,
        "first_pass_forward": first_pass_forward,
        "reached_final_third": reached,
        "seconds_to_final_third": None if seconds_to is None else round(seconds_to, 1),
        "gain_m": round(gain_units * PITCH_LENGTH_M / 100.0, 1),
        "terminal_outcome": terminal,
        "produced_shot": bool(milestones["produced_shot"]),
    }


def loss_sequence_row(sequence: pd.DataFrame) -> dict | None:
    """One row describing what happened after a team lost the ball."""
    if sequence is None or sequence.empty:
        return None
    seq = _ordered(sequence)
    loss_x = _first_valid(seq, "loss_x")
    loss_y = _first_valid(seq, "loss_y")
    zone = _first_valid(seq, "loss_zone") or zone_from_x(loss_x)
    terminal = _first_valid(seq, "terminal_outcome")
    regained = bool(str(terminal).lower() == "turnover") if terminal is not None else False

    minute = _first_valid(seq, "timeMin_at_loss")
    second = _first_valid(seq, "timeSec_at_loss")
    seconds_to = None
    if regained and minute is not None and second is not None:
        loss_t = float(minute) * 60.0 + float(second)
        seconds_to = max(0.0, float(seq["_t"].max()) - loss_t)

    milestones = calculate_sequence_milestones(sequence)
    opponent_passes = _first_valid(seq, "opponent_pass_count")
    return {
        "sequence_id": _first_valid(seq, "loss_sequence_id"),
        "loss_zone": zone,
        "loss_x": None if loss_x is None else round(float(loss_x), 1),
        "loss_y": None if loss_y is None else round(float(loss_y), 1),
        "initial_loss_type": _first_valid(seq, "type_of_initial_loss"),
        "regained": regained,
        "seconds_to_regain": None if seconds_to is None else round(seconds_to, 1),
        "opponent_passes": None if opponent_passes is None else int(opponent_passes),
        "opponent_reached_final_third": bool(milestones["reached_final_third"]),
        "terminal_outcome": terminal,
        "opponent_shot": bool(milestones["produced_shot"]),
    }


def _rows(sequences: Iterable[pd.DataFrame], builder, columns) -> pd.DataFrame:
    rows = [r for r in (builder(s) for s in (sequences or [])) if r is not None]
    return pd.DataFrame(rows, columns=columns)


def recovery_sequences(sequences: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Per-sequence recovery table (offensive transitions of one team)."""
    return _rows(sequences, recovery_sequence_row, RECOVERY_SEQUENCE_COLUMNS)


def loss_sequences(sequences: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Per-sequence loss table (defensive transitions of one team)."""
    return _rows(sequences, loss_sequence_row, LOSS_SEQUENCE_COLUMNS)


def _pct(series: pd.Series) -> float | None:
    values = series.dropna()
    if values.empty:
        return None
    return round(float(values.astype(bool).mean() * 100.0), 1)


def _median(series: pd.Series, digits: int = 1) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return None if values.empty else round(float(values.median()), digits)


def summarize_recoveries(table: pd.DataFrame) -> pd.DataFrame:
    """Summarise per-sequence recoveries by pitch third (plus an ``All`` row)."""
    rows = []
    groups = [(z, table[table["start_zone"] == z]) for z in ZONES] + [("All", table)]
    for zone, g in groups:
        n = int(len(g))
        rows.append({
            "zone": zone,
            "n": n,
            "low_sample": n < LOW_SAMPLE_THRESHOLD,
            "first_pass_forward_pct": _pct(g["first_pass_forward"]) if n else None,
            "reached_final_third_pct": _pct(g["reached_final_third"]) if n else None,
            "median_seconds_to_final_third": _median(g["seconds_to_final_third"]) if n else None,
            "shot_pct": _pct(g["produced_shot"]) if n else None,
            "avg_completed_passes": round(float(g["completed_passes"].mean()), 2) if n else None,
            "median_gain_m": _median(g["gain_m"]) if n else None,
            "median_duration_s": _median(g["duration_s"]) if n else None,
        })
    return pd.DataFrame(rows, columns=RECOVERY_SUMMARY_COLUMNS)


def summarize_losses(table: pd.DataFrame) -> pd.DataFrame:
    """Summarise per-sequence losses by pitch third (plus an ``All`` row)."""
    rows = []
    groups = [(z, table[table["loss_zone"] == z]) for z in ZONES] + [("All", table)]
    for zone, g in groups:
        n = int(len(g))
        secs = pd.to_numeric(g["seconds_to_regain"], errors="coerce")
        rows.append({
            "zone": zone,
            "n": n,
            "low_sample": n < LOW_SAMPLE_THRESHOLD,
            "regained_pct": _pct(g["regained"]) if n else None,
            "regained_within_5s_pct": round(float((secs <= QUICK_REGAIN_SECONDS).sum() / n * 100.0), 1) if n else None,
            "regained_within_10s_pct": round(float((secs <= REGAIN_WINDOW_SECONDS).sum() / n * 100.0), 1) if n else None,
            "median_seconds_to_regain": _median(secs) if n else None,
            "opponent_final_third_pct": _pct(g["opponent_reached_final_third"]) if n else None,
            "opponent_shot_pct": _pct(g["opponent_shot"]) if n else None,
        })
    return pd.DataFrame(rows, columns=LOSS_SUMMARY_COLUMNS)
