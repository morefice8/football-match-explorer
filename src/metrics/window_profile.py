"""Window profile: what a team did, and what the opponent did to it, in a stretch of play.

Everything here works on one *window* of the play clock (``start`` / ``end`` in
play seconds, see ``match_clock``) and on both teams of one match, so the same
functions serve a phase, a half or the whole match.

Definitions (explicit, tunable through the constants):

* **Touch**: a ball action of the team (pass, take-on, ball touch, shot,
  dispossession). **Intervention**: a defensive event of the *other* team
  (tackle, challenge, interception, blocked pass, ball recovery, clearance, or a
  foul it committed).
* **Pressed touch**: a touch followed, within ``PRESSURE_WINDOW_S`` seconds and
  ``PRESSURE_RADIUS_M`` metres, by an opponent intervention *in the same period*.
  It is a contest-near-the-ball measure: it also includes balls lost and
  recovered next to the touch, so read it as "where the opponent got hands on
  the play", not as intent to press. Deep touches are naturally less often
  pressed than touches near the opponent's area: compare like with like (the
  tables carry thirds and lanes for that).
* **Block height**: median distance from the own goal (metres) of the team's own
  interventions, goalkeepers excluded. It uses an explicit set of event types
  (see ``INTERVENTION_TYPES``); the app's ``get_defensive_actions`` set differs
  (it misses ``Blocked Pass`` because of a case mismatch and counts fouls
  suffered), so the two figures are not interchangeable.
* **Box touch**: a touch inside the penalty area the team is attacking.
* **Final-third entry**: a completed pass that starts before and ends beyond
  two thirds of the pitch.
* **Position**: median location of a player's touches and interventions in the
  window, in the team's own frame and in a shared frame (see ``pitch_zones``).
  It is where the player acts, not where he stands.
* **Drought**: the gaps between consecutive shots (or box touches) of a team in
  play minutes, with what the team did meanwhile.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.metrics import match_clock as mc
from src.metrics import pitch_zones as pz
from src.metrics.proportion_stats import group_vs_rest, wilson_interval

TOUCH_TYPES = frozenset({"Pass", "Take On", "Ball touch", "Shot", "Miss", "Attempt Saved", "Post", "Goal", "Dispossessed"})
INTERVENTION_TYPES = frozenset({"Tackle", "Challenge", "Interception", "Blocked Pass", "Ball recovery", "Clearance"})
SHOT_TYPES = frozenset({"Goal", "Miss", "Attempt Saved", "Post"})

PRESSURE_RADIUS_M = 10.0
PRESSURE_WINDOW_S = 6.0
LOW_SAMPLE_TOUCHES = 20
LOW_SAMPLE_ACTIONS = 8
MIN_POSITION_MINUTES = 12.0
MIN_LENGTH_PLAYERS = 7

VIEW_COLUMNS = (
    "team", "type", "period", "play_seconds", "x", "y", "end_x", "end_y", "player", "player_key", "jersey",
    "role", "is_keeper", "is_touch", "is_intervention", "is_shot", "is_cross", "is_loss", "ok", "fail",
)


# --------------------------------------------------------------------------- view

def _truthy(series: pd.Series) -> pd.Series:
    text = series.astype(object).where(series.notna(), "").astype(str).str.strip().str.casefold()
    return ~text.isin({"", "0", "0.0", "false", "no", "n", "none", "nan", "<na>"})


def _col(frame: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    return frame[name] if name in frame.columns else pd.Series(default, index=frame.index)


def prepare_events(frame: pd.DataFrame, clock: mc.PlayClock | None = None) -> pd.DataFrame:
    """Lean, typed view of the events used by every function below (one row per event)."""
    if frame is None or frame.empty or "team_name" not in frame.columns or "type_name" not in frame.columns:
        return pd.DataFrame(columns=list(VIEW_COLUMNS))
    clock = clock or mc.PlayClock.from_events(frame)
    kind = frame["type_name"].astype(object).where(frame["type_name"].notna(), "").astype(str)
    outcome = _col(frame, "outcome", "").astype(object).where(_col(frame, "outcome", "").notna(), "").astype(str).str.casefold()
    ok = outcome.isin({"successful", "1", "true"})
    fail = outcome.isin({"unsuccessful", "0", "false"})
    committed_foul = kind.eq("Foul") & fail
    own_goal = _truthy(_col(frame, "Own goal", "")) if "Own goal" in frame.columns else pd.Series(False, index=frame.index)
    player = _col(frame, "playerName", "").astype(object).where(_col(frame, "playerName", "").notna(), "")
    pid = _col(frame, "playerId", "")
    key = pid.astype(object).where(pid.notna() & pid.astype(str).str.strip().ne(""), player)
    keeper = _keeper_mask(frame)
    cross = _truthy(_col(frame, "cross", "")) | _truthy(_col(frame, "is_cross", ""))
    view = pd.DataFrame({
        "team": frame["team_name"],
        "type": kind,
        "period": pd.to_numeric(_col(frame, "periodId"), errors="coerce"),
        "play_seconds": clock.series(frame),
        "x": pd.to_numeric(_col(frame, "x"), errors="coerce"),
        "y": pd.to_numeric(_col(frame, "y"), errors="coerce"),
        "end_x": pd.to_numeric(_col(frame, "end_x"), errors="coerce"),
        "end_y": pd.to_numeric(_col(frame, "end_y"), errors="coerce"),
        "player": player,
        "player_key": key,
        "jersey": pd.to_numeric(_col(frame, "Mapped Jersey Number"), errors="coerce"),
        "role": _col(frame, "positional_role", "").astype(object).where(_col(frame, "positional_role").notna(), ""),
        "is_keeper": keeper,
        "is_touch": kind.isin(TOUCH_TYPES),
        "is_intervention": kind.isin(INTERVENTION_TYPES) | committed_foul,
        "is_shot": kind.isin(SHOT_TYPES) & ~own_goal,
        "is_cross": cross & kind.eq("Pass"),
        "is_loss": (kind.eq("Pass") & fail) | kind.eq("Dispossessed"),
        "ok": ok,
        "fail": fail,
    }, index=frame.index)
    return view[view["play_seconds"].notna()].copy()


def _keeper_mask(frame: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for column in ("positional_role", "position", "positionName", "Mapped Position", "mapped_position", "role"):
        if column in frame.columns:
            text = frame[column].astype(object).where(frame[column].notna(), "").astype(str).str.upper()
            mask |= text.eq("GK") | text.str.contains("GOALKEEP", na=False)
    return mask


def window(view: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    return view[(view["play_seconds"] >= start) & (view["play_seconds"] < end)]


def _minutes(start: float, end: float) -> float:
    return max(0.0, (end - start) / 60.0)


def _own_frame(view: pd.DataFrame) -> pd.DataFrame:
    return view[view["x"].notna() & view["y"].notna()]


# --------------------------------------------------------------------------- territory

def territory_profile(view: pd.DataFrame, team: str, opponent: str, start: float, end: float) -> dict:
    """Possession, access to the final third / box, shots and block height for one team and window."""
    w = window(view, start, end)
    mine, theirs = w[w["team"] == team], w[w["team"] == opponent]
    minutes = _minutes(start, end)
    touches = _own_frame(mine[mine["is_touch"]])
    opp_touches = int(theirs["is_touch"].sum())
    final = touches[touches["x"] > pz.THIRD_EDGES_X[1]]
    box = touches[pz.in_penalty_area(touches["x"], touches["y"])] if len(touches) else touches
    entries = mine[(mine["type"] == "Pass") & mine["ok"] & (mine["x"] < pz.THIRD_EDGES_X[1]) & (mine["end_x"] >= pz.THIRD_EDGES_X[1])]
    lost_final = int(final["is_loss"].sum())
    own_iv = _own_frame(mine[mine["is_intervention"] & ~mine["is_keeper"]])
    per10 = lambda n: round(n / minutes * 10.0, 2) if minutes else None
    n_touch = int(len(touches))
    return {
        "play_minutes": round(minutes, 1),
        "touches": n_touch,
        "touches_per_min": round(n_touch / minutes, 2) if minutes else None,
        "touch_share_pct": round(100.0 * n_touch / (n_touch + opp_touches), 1) if (n_touch + opp_touches) else None,
        "own_half_touch_pct": round(100.0 * float((touches["x"] <= 50.0).mean()), 1) if n_touch else None,
        "final_third_touches": int(len(final)),
        "final_third_per_10min": per10(len(final)),
        "final_third_entries": int(len(entries)),
        "box_touches": int(len(box)),
        "box_touches_per_10min": per10(len(box)),
        "crosses": int(mine["is_cross"].sum()),
        "crosses_completed": int((mine["is_cross"] & mine["ok"]).sum()),
        "shots": int(mine["is_shot"].sum()),
        "final_third_loss_pct": round(100.0 * lost_final / len(final), 1) if len(final) else None,
        "interventions": int(len(own_iv)),
        "block_height_m": round(float(own_iv["x"].median() * pz.X_METRES), 1) if len(own_iv) else None,
    }


# --------------------------------------------------------------------------- pressure

PRESSURE_COLUMNS = (
    "team", "player", "jersey", "is_keeper", "play_seconds", "period", "x", "y", "lane", "third", "pressed",
    "presser", "intervention", "distance_m", "event_x", "event_y",
)


def pressure_table(
    view: pd.DataFrame, team: str, opponent: str, start: float, end: float,
    *, radius_m: float = PRESSURE_RADIUS_M, within_s: float = PRESSURE_WINDOW_S,
) -> pd.DataFrame:
    """One row per touch of ``team`` with whether an opponent intervention followed close by."""
    mine = view[(view["team"] == team) & view["is_touch"]]
    touches = _own_frame(window(mine, start, end))
    if touches.empty:
        return pd.DataFrame(columns=list(PRESSURE_COLUMNS))
    theirs = view[(view["team"] == opponent) & view["is_intervention"] & view["x"].notna() & view["y"].notna()]
    theirs = theirs[(theirs["play_seconds"] >= start) & (theirs["play_seconds"] <= end + within_s)]
    # the opponent's events are in its own frame: rotate them into this team's frame
    ev_x, ev_y = 100.0 - theirs["x"].to_numpy(), 100.0 - theirs["y"].to_numpy()
    ev_t, ev_p = theirs["play_seconds"].to_numpy(), theirs["period"].to_numpy()
    t_t, t_p = touches["play_seconds"].to_numpy(), touches["period"].to_numpy()
    t_x, t_y = touches["x"].to_numpy(), touches["y"].to_numpy()
    pressed = np.zeros(len(touches), dtype=bool)
    nearest = np.full(len(touches), -1)
    dist = np.full(len(touches), np.nan)
    if len(theirs):
        dt = ev_t[None, :] - t_t[:, None]
        d = np.hypot((ev_x[None, :] - t_x[:, None]) * pz.X_METRES, (ev_y[None, :] - t_y[:, None]) * pz.Y_METRES)
        ok = (dt >= 0) & (dt <= within_s) & (ev_p[None, :] == t_p[:, None]) & (d <= radius_m)
        pressed = ok.any(axis=1)
        masked = np.where(ok, d, np.inf)
        nearest = np.where(pressed, masked.argmin(axis=1), -1)
        dist = np.where(pressed, masked.min(axis=1), np.nan)
    ev_names = theirs["player"].to_numpy() if len(theirs) else np.array([])
    ev_types = theirs["type"].to_numpy() if len(theirs) else np.array([])
    table = pd.DataFrame({
        "team": team,
        "player": touches["player"].to_numpy(),
        "jersey": touches["jersey"].to_numpy(),
        "is_keeper": touches["is_keeper"].to_numpy(),
        "play_seconds": t_t,
        "period": t_p,
        "x": t_x,
        "y": t_y,
        "lane": pz.lane_series(touches["y"]).to_numpy(),
        "third": pz.third_series(touches["x"]).to_numpy(),
        "pressed": pressed,
        "presser": [ev_names[i] if i >= 0 else None for i in nearest],
        "intervention": [ev_types[i] if i >= 0 else None for i in nearest],
        "distance_m": np.round(dist, 1),
        "event_x": [ev_x[i] if i >= 0 else np.nan for i in nearest],
        "event_y": [ev_y[i] if i >= 0 else np.nan for i in nearest],
    })
    return table[list(PRESSURE_COLUMNS)].reset_index(drop=True)


def summarize_pressure(table: pd.DataFrame, by: str, *, include_keepers: bool = False) -> pd.DataFrame:
    """Pressed share by ``by`` in {"all", "player", "lane", "third"} with CI and a vs-rest test."""
    columns = ["group", "touches", "pressed", "pressed_pct", "ci_low_pct", "ci_high_pct", "low_sample", "vs_rest_p"]
    if table is None or table.empty:
        return pd.DataFrame(columns=columns)
    t = table if include_keepers else table[~table["is_keeper"].astype(bool)]
    if t.empty:
        return pd.DataFrame(columns=columns)
    total_n, total_p = int(len(t)), int(t["pressed"].sum())
    if by == "all":
        groups = [("all", t)]
    else:
        groups = [(name, g) for name, g in t.groupby(by, dropna=True, sort=False)]
    rows = []
    for name, g in groups:
        n, p = int(len(g)), int(g["pressed"].sum())
        lo, hi = wilson_interval(p, n)
        rest_n, rest_p = total_n - n, total_p - p
        rows.append({
            "group": name, "touches": n, "pressed": p,
            "pressed_pct": round(100.0 * p / n, 1),
            "ci_low_pct": round(100.0 * lo, 1), "ci_high_pct": round(100.0 * hi, 1),
            "low_sample": n < LOW_SAMPLE_TOUCHES,
            "vs_rest_p": None if by == "all" or rest_n == 0 else round(group_vs_rest(p, n, rest_p, rest_n), 4),
        })
    out = pd.DataFrame(rows, columns=columns)
    if by == "lane":
        order = {lane: i for i, lane in enumerate(pz.LANES)}
        out = out.sort_values("group", key=lambda s: s.map(order), kind="stable")
    elif by == "third":
        order = {third: i for i, third in enumerate(pz.THIRDS)}
        out = out.sort_values("group", key=lambda s: s.map(order), kind="stable")
    elif by == "player":
        out = out.sort_values("touches", ascending=False, kind="stable")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- positions

def on_pitch_intervals(view: pd.DataFrame, frame: pd.DataFrame, clock: mc.PlayClock) -> dict:
    """``{(team, player_key): (on_play_s, off_play_s)}`` from starters, substitutions and dismissals."""
    end = clock.total_seconds
    intervals: dict = {}
    if frame is None or frame.empty or "type_name" not in frame.columns:
        return intervals
    base = frame.copy()
    base["_t"] = clock.series(base)
    pid = _col(base, "playerId", "")
    name = _col(base, "playerName", "").astype(object).where(_col(base, "playerName").notna(), "")
    base["_key"] = pid.astype(object).where(pid.notna() & pid.astype(str).str.strip().ne(""), name)
    starter = _truthy(_col(base, "Is Starter", "")) if "Is Starter" in base.columns else pd.Series(False, index=base.index)
    dismissals = _dismissal_times(base, clock)
    for (team, key), rows in base[base["_key"].astype(str).str.strip().ne("")].groupby(["team_name", "_key"]):
        kind = rows["type_name"].astype(str)
        on_events = rows.loc[kind.eq("Player on"), "_t"].dropna()
        off_events = rows.loc[kind.eq("Player off"), "_t"].dropna()
        if starter.reindex(rows.index).any():
            on = 0.0
        elif len(on_events):
            on = float(on_events.min())
        else:
            continue
        offs = [float(t) for t in off_events if t >= on] + [t for t in dismissals.get((team, key), []) if t >= on]
        intervals[(team, key)] = (on, min(offs) if offs else end)
    return intervals


def _dismissal_times(frame: pd.DataFrame, clock: mc.PlayClock) -> dict:
    from src.metrics.card_events import extract_card_events

    cards = extract_card_events(frame)
    out: dict = {}
    if cards.empty or "card_resulted_in_dismissal" not in cards.columns:
        return out
    cards = cards[cards["card_resulted_in_dismissal"].fillna(False).astype(bool)]
    for _, row in cards.iterrows():
        minute, second = pd.to_numeric(row.get("timeMin"), errors="coerce"), pd.to_numeric(row.get("timeSec"), errors="coerce")
        t = clock.play_seconds(row.get("periodId"), np.nan if pd.isna(minute) else float(minute) * 60.0 + (0.0 if pd.isna(second) else float(second)))
        pid = row.get("playerId")
        key = pid if pd.notna(pid) and str(pid).strip() else row.get("playerName")
        if pd.notna(t):
            out.setdefault((row.get("team_name"), key), []).append(float(t))
    return out


POSITION_COLUMNS = (
    "team", "player", "jersey", "role", "x", "y", "x_shared", "y_shared", "actions", "minutes_present", "low_sample",
)


def window_positions(
    view: pd.DataFrame, team: str, start: float, end: float, intervals: dict, reference_team: str,
    *, min_minutes: float = MIN_POSITION_MINUTES, min_actions: int = LOW_SAMPLE_ACTIONS, minimum_rows: int = 3,
) -> pd.DataFrame:
    """Median action location per outfield player present long enough in the window."""
    w = window(view[(view["team"] == team) & (view["is_touch"] | view["is_intervention"])], start, end)
    w = _own_frame(w[~w["is_keeper"]])
    rows = []
    for key, g in w.groupby("player_key", sort=False):
        on, off = intervals.get((team, key), (None, None))
        if on is None:
            continue
        minutes = max(0.0, min(off, end) - max(on, start)) / 60.0
        if minutes < min(min_minutes, _minutes(start, end)) or len(g) < minimum_rows:
            continue
        x, y = float(g["x"].median()), float(g["y"].median())
        rotate = team != reference_team
        jersey = g["jersey"].dropna()
        role = g["role"].astype(str)
        rows.append({
            "team": team, "player": g["player"].iloc[0], "jersey": int(jersey.iloc[0]) if len(jersey) else None,
            "role": role[role.ne("")].iloc[0] if role.ne("").any() else "",
            "x": round(x, 1), "y": round(y, 1),
            "x_shared": round(100.0 - x if rotate else x, 1), "y_shared": round(100.0 - y if rotate else y, 1),
            "actions": int(len(g)), "minutes_present": round(minutes, 1), "low_sample": bool(len(g) < min_actions),
        })
    return pd.DataFrame(rows, columns=list(POSITION_COLUMNS))


def team_length_m(
    positions: pd.DataFrame, *, min_actions: int = LOW_SAMPLE_ACTIONS, min_players: int = MIN_LENGTH_PLAYERS,
) -> float | None:
    """Distance (m) between the deepest and the highest outfield player (median action x).

    Needs ``min_players`` players with at least ``min_actions`` actions each: a
    "length" computed from two or three players of a short phase is noise.
    """
    if positions is None or positions.empty:
        return None
    p = positions[positions["actions"] >= min_actions]
    if len(p) < max(2, min_players):
        return None
    return round(float((p["x"].max() - p["x"].min()) * pz.X_METRES), 1)


# --------------------------------------------------------------------------- droughts

DROUGHT_COLUMNS = (
    "team", "kind", "start_clock", "end_clock", "play_minutes", "box_touches", "final_third_touches",
    "crosses", "opponent_shots", "ends_with_event",
)


def droughts(
    view: pd.DataFrame, clock: mc.PlayClock, team: str, opponent: str, *, kind: str = "shots",
    min_minutes: float = 15.0,
) -> pd.DataFrame:
    """Long stretches without a shot (``kind="shots"``) or without a box touch (``"box_touches"``)."""
    mine = view[view["team"] == team]
    if kind == "shots":
        marks = mine[mine["is_shot"]]["play_seconds"]
    else:
        t = _own_frame(mine[mine["is_touch"]])
        marks = t[pz.in_penalty_area(t["x"], t["y"])]["play_seconds"] if len(t) else pd.Series(dtype=float)
    times = [0.0] + sorted(float(v) for v in marks) + [clock.total_seconds]
    rows = []
    for index, (a, b) in enumerate(zip(times[:-1], times[1:])):
        minutes = (b - a) / 60.0
        if minutes < min_minutes:
            continue
        # the event that opens the drought belongs to the previous stretch (not for the kick-off)
        span = view[((view["play_seconds"] > a) if index > 0 else (view["play_seconds"] >= a)) & (view["play_seconds"] < b)]
        ours = span[span["team"] == team]
        touches = _own_frame(ours[ours["is_touch"]])
        box = touches[pz.in_penalty_area(touches["x"], touches["y"])] if len(touches) else touches
        rows.append({
            "team": team, "kind": kind, "start_clock": clock.label(a), "end_clock": clock.label(b),
            "play_minutes": round(minutes, 1), "box_touches": int(len(box)),
            "final_third_touches": int((touches["x"] > pz.THIRD_EDGES_X[1]).sum()),
            "crosses": int(ours["is_cross"].sum()),
            "opponent_shots": int(span[(span["team"] == opponent) & span["is_shot"]].shape[0]),
            "ends_with_event": index < len(times) - 2,
        })
    return pd.DataFrame(rows, columns=list(DROUGHT_COLUMNS))
