"""Match phases: stretches of play between the events that change the game.

A match is cut at

* goals (the score state changes),
* dismissals (a team plays with ten),
* substitution clusters (the shape and the people change),

and optionally at the interval. Cuts that would create tiny stretches are
merged, with a priority rule: a goal or a dismissal always wins over a
substitution that sits right before it. Every phase carries its score state,
its length in minutes of *play* (see ``match_clock``) and who came on, so any
phase-level analysis can say "after the 2-1, with the same eleven" instead of
quoting clock minutes.

Defaults (all tunable):

* ``min_event_minutes``: minimum length of a phase that starts at a goal or a
  dismissal (3 min).
* ``min_change_minutes``: minimum length of a phase that starts at a
  substitution cluster, also measured to the end of the match (10 min). A
  substitution cluster is every change by one team within
  ``CLUSTER_SECONDS`` of the first one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.metrics import match_clock as mc

CLUSTER_SECONDS = 150.0
MIN_EVENT_MINUTES = 3.0
MIN_CHANGE_MINUTES = 10.0
SHORT_PHASE_MINUTES = 5.0

TRIGGER_KICKOFF = "kickoff"
TRIGGER_GOAL = "goal"
TRIGGER_DISMISSAL = "dismissal"
TRIGGER_SUBSTITUTION = "substitution"
TRIGGER_HALF_TIME = "half_time"
_PRIORITY = {TRIGGER_GOAL: 3, TRIGGER_DISMISSAL: 3, TRIGGER_HALF_TIME: 2, TRIGGER_SUBSTITUTION: 1}

STATE_LEADING = "leading"
STATE_LEVEL = "level"
STATE_TRAILING = "trailing"

PHASE_COLUMNS = (
    "phase_id", "label", "start_play_s", "end_play_s", "start_clock", "end_clock", "play_minutes",
    "trigger", "home_team", "away_team", "home_score", "away_score", "home_score_end",
    "away_score_end", "goals_inside", "home_state", "away_state", "home_subs_on",
    "away_subs_on", "short",
)


@dataclass(frozen=True)
class _Boundary:
    play_seconds: float
    trigger: str


def _state(own: int, other: int) -> str:
    if own > other:
        return STATE_LEADING
    if own < other:
        return STATE_TRAILING
    return STATE_LEVEL


def goal_events(
    frame: pd.DataFrame,
    home_team: str,
    away_team: str,
    clock: mc.PlayClock | None = None,
    shots_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Goals in play-clock order, credited to the team that scored (own goals to the opponent)."""
    clock = clock or mc.PlayClock.from_events(frame)
    shots = shots_df
    if shots is None:
        try:
            from src.metrics.shot_classification import classify_shots

            shots = classify_shots(frame)
        except Exception:  # pragma: no cover - degraded input falls back to raw Goal events
            shots = None
    rows = pd.DataFrame()
    if shots is not None and not shots.empty and "shot_outcome" in shots.columns:
        rows = shots[shots["shot_outcome"].isin(["goal", "own_goal"])].copy()
        credited = rows["team_name"].where(
            rows["shot_outcome"].ne("own_goal"),
            rows["team_name"].map({home_team: away_team, away_team: home_team}),
        )
    elif "type_name" in frame.columns:
        rows = frame[frame["type_name"].eq("Goal")].copy()
        credited = rows["team_name"]
    if rows.empty:
        return pd.DataFrame(columns=["play_seconds", "clock", "team", "player"])
    out = pd.DataFrame({
        "play_seconds": clock.series(rows),
        "clock": [mc.format_clock(v) for v in mc.clock_seconds(rows)],
        "team": credited.to_numpy(),
        "player": rows["playerName"].to_numpy() if "playerName" in rows.columns else None,
    })
    out = out[out["team"].isin([home_team, away_team]) & out["play_seconds"].notna()]
    return out.sort_values("play_seconds", kind="stable").reset_index(drop=True)


def _substitution_clusters(frame: pd.DataFrame, clock: mc.PlayClock, cluster_seconds: float) -> list[dict]:
    """One record per team and cluster of substitutions."""
    if "type_name" not in frame.columns:
        return []
    changes = frame[frame["type_name"].isin(["Player on", "Player off"])].copy()
    if changes.empty:
        return []
    changes["_t"] = clock.series(changes)
    changes = changes[changes["_t"].notna()].sort_values("_t", kind="stable")
    clusters: list[dict] = []
    for team, group in changes.groupby("team_name", sort=False):
        current: dict | None = None
        for _, row in group.iterrows():
            if current is None or row["_t"] - current["start"] > cluster_seconds:
                current = {"team": team, "start": float(row["_t"]), "on": [], "off": []}
                clusters.append(current)
            name = row["playerName"] if "playerName" in group.columns and pd.notna(row.get("playerName")) else None
            (current["on"] if row["type_name"] == "Player on" else current["off"]).append(name)
    return clusters


def _dismissals(frame: pd.DataFrame, clock: mc.PlayClock) -> list[float]:
    from src.metrics.card_events import extract_card_events

    cards = extract_card_events(frame)
    if cards.empty or "card_resulted_in_dismissal" not in cards.columns:
        return []
    cards = cards[cards["card_resulted_in_dismissal"].fillna(False).astype(bool)]
    if "card_rescinded" in cards.columns:
        cards = cards[~cards["card_rescinded"].fillna(False).astype(bool)]
    times = clock.series(cards).dropna()
    return [float(t) for t in times]


def _merge_boundaries(
    candidates: list[_Boundary],
    end: float,
    min_event_s: float,
    min_change_s: float,
) -> list[_Boundary]:
    """Drop cuts that create tiny phases; goals/dismissals win over substitutions."""
    kept: list[_Boundary] = []
    last_t = 0.0
    for cut in sorted(candidates, key=lambda b: (b.play_seconds, -_PRIORITY.get(b.trigger, 0))):
        if cut.play_seconds <= 0.0 or cut.play_seconds >= end:
            continue
        need = min_change_s if cut.trigger == TRIGGER_SUBSTITUTION else min_event_s
        if cut.play_seconds - last_t >= need:
            kept.append(cut)
            last_t = cut.play_seconds
            continue
        # too close to the previous cut: a stronger cut replaces a weaker one
        if kept and _PRIORITY.get(cut.trigger, 0) > _PRIORITY.get(kept[-1].trigger, 0):
            previous_t = kept[-2].play_seconds if len(kept) > 1 else 0.0
            if cut.play_seconds - previous_t >= min_event_s:
                kept[-1] = cut
                last_t = cut.play_seconds
    # a substitution cut right before the end would leave a sliver
    while kept and kept[-1].trigger == TRIGGER_SUBSTITUTION and end - kept[-1].play_seconds < min_change_s:
        kept.pop()
    return kept


def build_match_phases(
    frame: pd.DataFrame,
    home_team: str,
    away_team: str,
    *,
    goals: pd.DataFrame | None = None,
    min_event_minutes: float = MIN_EVENT_MINUTES,
    min_change_minutes: float = MIN_CHANGE_MINUTES,
    cluster_seconds: float = CLUSTER_SECONDS,
    split_at_half_time: bool = False,
    clock: mc.PlayClock | None = None,
) -> pd.DataFrame:
    """Return one row per phase (see ``PHASE_COLUMNS``) for the whole match."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(PHASE_COLUMNS))
    clock = clock or mc.PlayClock.from_events(frame)
    end = clock.total_seconds
    if end <= 0:
        return pd.DataFrame(columns=list(PHASE_COLUMNS))
    goals = goals if goals is not None else goal_events(frame, home_team, away_team, clock)

    candidates: list[_Boundary] = [_Boundary(float(t), TRIGGER_GOAL) for t in goals["play_seconds"]] if len(goals) else []
    candidates += [_Boundary(t, TRIGGER_DISMISSAL) for t in _dismissals(frame, clock)]
    clusters = _substitution_clusters(frame, clock, cluster_seconds)
    candidates += [_Boundary(c["start"], TRIGGER_SUBSTITUTION) for c in clusters]
    if split_at_half_time:
        candidates += [_Boundary(offset, TRIGGER_HALF_TIME) for _, _, _, offset in clock.periods[1:]]
    cuts = _merge_boundaries(candidates, end, min_event_minutes * 60.0, min_change_minutes * 60.0)

    starts = [(0.0, TRIGGER_KICKOFF)] + [(c.play_seconds, c.trigger) for c in cuts]
    ends = [c.play_seconds for c in cuts] + [end]
    rows = []
    for index, ((start, trigger), stop) in enumerate(zip(starts, ends), start=1):
        last = index == len(starts)
        # a goal on the boundary belongs to the phase that starts there
        before_stop = (lambda t: t <= stop + 1e-6) if last else (lambda t: t < stop - 1e-6)
        in_start = goals[goals["play_seconds"] <= start + 1e-6] if len(goals) else goals
        inside = goals[(goals["play_seconds"] > start + 1e-6) & before_stop(goals["play_seconds"])] if len(goals) else goals
        in_end = goals[before_stop(goals["play_seconds"])] if len(goals) else goals
        count = lambda g, team: int((g["team"] == team).sum()) if len(g) else 0
        hs, as_ = count(in_start, home_team), count(in_start, away_team)
        he, ae = count(in_end, home_team), count(in_end, away_team)
        on = {home_team: [], away_team: []}
        for c in clusters:
            if start - 1e-6 <= c["start"] < stop and c["team"] in on:
                on[c["team"]] += [n for n in c["on"] if n]
        start_label = mc.format_clock(clock.locate(start)[1]) if clock.locate(start) else ""
        end_label = clock.label(stop)
        start_min = int(np.ceil(clock.locate(start)[1] / 60.0)) if clock.locate(start) and start > 0 else 0
        end_min = int(np.floor(clock.locate(stop)[1] / 60.0)) if clock.locate(stop) else 0
        rows.append({
            "phase_id": index,
            "label": f"{start_min}'-{end_min}'",
            "start_play_s": round(start, 1),
            "end_play_s": round(stop, 1),
            "start_clock": start_label,
            "end_clock": end_label,
            "play_minutes": round((stop - start) / 60.0, 1),
            "trigger": trigger,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": hs,
            "away_score": as_,
            "home_score_end": he,
            "away_score_end": ae,
            "goals_inside": int(len(inside)),
            "home_state": _state(hs, as_),
            "away_state": _state(as_, hs),
            "home_subs_on": "; ".join(on[home_team]),
            "away_subs_on": "; ".join(on[away_team]),
            "short": bool((stop - start) / 60.0 < SHORT_PHASE_MINUTES),
        })
    return pd.DataFrame(rows, columns=list(PHASE_COLUMNS))
