"""Insight candidates: the findings of a match that are worth a sentence in an article.

The rules read the window-level tables (``window_profile``, ``pressure_profile``,
``shot_droughts``) and flag only patterns that pass an explicit size *and*
evidence bar, so a candidate is something a writer can quote with its numbers
and its caveat instead of a number that merely looks big. Nothing here invents
football meaning: each candidate states a measured fact, its strength and what
the measure cannot say.

Rules (thresholds are the constants below):

* ``pressure_hotspot_player``: one player's touches are followed by an opponent
  intervention at least twice as often as his team's average (and at least
  ``HOTSPOT_MIN_GAP_PP`` points more), tested against the rest of the team.
* ``pressure_free_players``: players with plenty of touches that are almost
  never followed by an opponent intervention.
* ``pressure_lane_gap``: the most and the least contested lane (of the team's own
  frame) differ by a wide margin, tested with Fisher's exact test.
* ``block_height_shift`` / ``team_length_shift``: a team's defensive block, or
  its length, changes by many metres between consecutive phases.
* ``shot_drought`` / ``box_touch_drought``: a long stretch without a shot, or
  without a touch in the opponent's box, with what the team did meanwhile.
"""

from __future__ import annotations

import json

import pandas as pd

from src.metrics.proportion_stats import fisher_exact_two_sided

PRESSURE_MIN_TOUCHES = 25
HOTSPOT_RATIO = 2.0
HOTSPOT_MIN_GAP_PP = 12.0
FREE_MAX_PCT = 5.0
LANE_MIN_TOUCHES = 40
LANE_GAP_PP = 15.0
STRONG_P = 0.01
MODERATE_P = 0.05
BLOCK_SHIFT_M = 10.0
LENGTH_SHIFT_M = 10.0
SHIFT_MIN_PHASE_MINUTES = 5.0
BLOCK_MIN_INTERVENTIONS = 10
DROUGHT_MIN_MINUTES = 20.0
MAX_CANDIDATES = 15

COLUMNS = ("id", "report_team", "team_name", "kind", "scope", "strength", "statement", "numbers", "caveat")
_STRENGTH_ORDER = {"strong": 0, "moderate": 1, "weak": 2}

_CAVEAT_PRESSURE = (
    "Single match. An 'intervention' is any tackle, challenge, interception, blocked pass, recovery, "
    "clearance or committed foul by the opponent within 6 s and 10 m of a touch, so it also counts "
    "lost balls that were recovered next to the touch; deep touches are naturally less often contested."
)
_CAVEAT_PHASE = "Phases are short and event-based: small samples, and positions are where players act, not where they stand."
_CAVEAT_DROUGHT = "Counts events recorded by the data provider; a touch in the box needs a player to act there, so quiet presence is not counted."


def _row(team: str, kind: str, scope: str, strength: str, statement: str, numbers: dict, caveat: str, slug: str) -> dict:
    return {
        "id": f"{kind}:{team}:{scope}:{slug}", "report_team": team, "team_name": team, "kind": kind, "scope": scope,
        "strength": strength, "statement": statement, "numbers": json.dumps(numbers, ensure_ascii=False, sort_keys=True), "caveat": caveat,
    }


def _strength_from_p(p: float | None) -> str | None:
    if p is None or pd.isna(p):
        return None
    if p < STRONG_P:
        return "strong"
    if p < MODERATE_P:
        return "moderate"
    return None


def _pressure_rules(pressure: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    if pressure is None or pressure.empty:
        return out
    match = pressure[pressure["phase_id"] == 0]
    for team, part in match.groupby("team_name", sort=False):
        overall = part[part["group_type"] == "all"]
        if overall.empty:
            continue
        team_pct = float(overall["pressed_pct"].iloc[0])
        team_n, team_p = int(overall["touches"].iloc[0]), int(overall["pressed"].iloc[0])
        players = part[part["group_type"] == "player"]
        # hotspot players
        for r in players.itertuples():
            strength = _strength_from_p(r.vs_rest_p)
            if (r.touches >= PRESSURE_MIN_TOUCHES and strength and r.pressed_pct >= HOTSPOT_RATIO * team_pct
                    and r.pressed_pct >= team_pct + HOTSPOT_MIN_GAP_PP):
                rest_n, rest_p = team_n - r.touches, team_p - r.pressed
                rest_pct = round(100.0 * rest_p / rest_n, 1) if rest_n else None
                out.append(_row(
                    team, "pressure_hotspot_player", "match", strength,
                    f"{r.group} ({team}): {r.pressed} of {r.touches} touches ({r.pressed_pct:.0f}%) were followed by an opponent intervention, "
                    f"against {rest_pct}% for the rest of the team (Fisher p={r.vs_rest_p:.4f}).",
                    {"player": r.group, "touches": int(r.touches), "pressed": int(r.pressed), "pressed_pct": float(r.pressed_pct),
                     "ci_low_pct": float(r.ci_low_pct), "ci_high_pct": float(r.ci_high_pct), "rest_pressed_pct": rest_pct, "p": float(r.vs_rest_p),
                     "team_pressed_pct": team_pct},
                    _CAVEAT_PRESSURE, str(r.group)))
        # players left almost free
        free = players[(players["touches"] >= PRESSURE_MIN_TOUCHES) & (players["pressed_pct"] <= FREE_MAX_PCT) & (players["vs_rest_p"].fillna(1.0) < STRONG_P)]
        if len(free):
            names = list(free["group"])
            touches = int(free["touches"].sum())
            hits = int(free["pressed"].sum())
            out.append(_row(
                team, "pressure_free_players", "match", "strong",
                f"{', '.join(names)} ({team}): {hits} of {touches} touches ({100.0 * hits / touches:.0f}%) were followed by an opponent intervention "
                f"against {team_pct:.0f}% for the team: the opponent almost never got hands on the play there.",
                {"players": names, "touches": touches, "pressed": hits, "team_pressed_pct": team_pct}, _CAVEAT_PRESSURE, "+".join(names)))
        # lane gap
        lanes = part[(part["group_type"] == "lane") & (part["touches"] >= LANE_MIN_TOUCHES)]
        if len(lanes) >= 2:
            hi, lo = lanes.loc[lanes["pressed_pct"].idxmax()], lanes.loc[lanes["pressed_pct"].idxmin()]
            gap = float(hi["pressed_pct"] - lo["pressed_pct"])
            p = fisher_exact_two_sided(int(hi["pressed"]), int(hi["touches"] - hi["pressed"]), int(lo["pressed"]), int(lo["touches"] - lo["pressed"]))
            strength = _strength_from_p(p)
            if gap >= LANE_GAP_PP and strength:
                out.append(_row(
                    team, "pressure_lane_gap", "match", strength,
                    f"{team}: touches in the {hi['group']} lane were followed by an opponent intervention {hi['pressed_pct']:.0f}% of the time "
                    f"({int(hi['pressed'])}/{int(hi['touches'])}), against {lo['pressed_pct']:.0f}% in the {lo['group']} lane "
                    f"({int(lo['pressed'])}/{int(lo['touches'])}); Fisher p={p:.4f}.",
                    {"most_contested_lane": hi["group"], "most_contested_pct": float(hi["pressed_pct"]), "least_contested_lane": lo["group"],
                     "least_contested_pct": float(lo["pressed_pct"]), "gap_pp": round(gap, 1), "p": round(p, 4)},
                    _CAVEAT_PRESSURE + " Lanes are in the team's own attacking frame (right = low y).", f"{hi['group']}-vs-{lo['group']}"))
    return out


def _shift_rules(window: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    if window is None or window.empty:
        return out
    phases = window[window["phase_id"] > 0]
    for team, part in phases.groupby("team_name", sort=False):
        part = part.sort_values("phase_id")
        for column, kind, threshold, label, need in (
            ("block_height_m", "block_height_shift", BLOCK_SHIFT_M, "defensive block", "interventions"),
            ("team_length_m", "team_length_shift", LENGTH_SHIFT_M, "team length", None),
        ):
            usable = part[(part["play_minutes"] >= SHIFT_MIN_PHASE_MINUTES) & part[column].notna()]
            if need:
                usable = usable[usable[need] >= BLOCK_MIN_INTERVENTIONS]
            rows = list(usable.itertuples())
            for a, b in zip(rows[:-1], rows[1:]):
                delta = float(getattr(b, column) - getattr(a, column))
                if abs(delta) >= threshold:
                    strength = "strong" if abs(delta) >= 1.5 * threshold else "moderate"
                    direction = "rose" if delta > 0 else "dropped"
                    out.append(_row(
                        team, kind, f"{a.phase_label}->{b.phase_label}", strength,
                        f"{team}: the {label} {direction} by {abs(delta):.0f} m between {a.phase_label} ({getattr(a, column):.0f} m) and {b.phase_label} ({getattr(b, column):.0f} m).",
                        {"from_phase": a.phase_label, "to_phase": b.phase_label, "from_m": float(getattr(a, column)), "to_m": float(getattr(b, column)),
                         "delta_m": round(delta, 1), "from_minutes": float(a.play_minutes), "to_minutes": float(b.play_minutes)},
                        _CAVEAT_PHASE, f"{a.phase_id}-{b.phase_id}"))
    return out


def _drought_rules(droughts: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    if droughts is None or droughts.empty:
        return out
    for r in droughts.itertuples():
        if r.play_minutes < DROUGHT_MIN_MINUTES:
            continue
        what = "shot" if r.kind == "shots" else "touch in the opponent's box"
        kind = "shot_drought" if r.kind == "shots" else "box_touch_drought"
        out.append(_row(
            r.team_name, kind, f"{r.start_clock}-{r.end_clock}", "strong" if r.play_minutes >= 30 else "moderate",
            f"{r.team_name}: {r.play_minutes:.0f} minutes of play ({r.start_clock} to {r.end_clock}) without a {what}; "
            f"meanwhile {int(r.final_third_touches)} touches in the final third, {int(r.box_touches)} in the box, {int(r.crosses)} crosses; the opponent shot {int(r.opponent_shots)} times.",
            {"minutes": float(r.play_minutes), "start_clock": r.start_clock, "end_clock": r.end_clock, "final_third_touches": int(r.final_third_touches),
             "box_touches": int(r.box_touches), "crosses": int(r.crosses), "opponent_shots": int(r.opponent_shots), "ends_with_event": bool(r.ends_with_event)},
            _CAVEAT_DROUGHT, r.kind))
    return out


def build_insight_candidates(
    window_profile: pd.DataFrame,
    pressure_profile: pd.DataFrame,
    droughts: pd.DataFrame,
    *,
    max_candidates: int = MAX_CANDIDATES,
) -> pd.DataFrame:
    """Ranked candidates (strong first), one row per finding with its numbers and caveat."""
    rows = _pressure_rules(pressure_profile) + _shift_rules(window_profile) + _drought_rules(droughts)
    if not rows:
        return pd.DataFrame(columns=list(COLUMNS))
    out = pd.DataFrame(rows, columns=list(COLUMNS))
    out["_rank"] = out["strength"].map(_STRENGTH_ORDER)
    out = out.sort_values(["_rank", "kind", "team_name"], kind="stable").drop(columns="_rank")
    return out.head(max_candidates).reset_index(drop=True)
