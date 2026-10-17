"""Turn a Match Analysis Pack (schema 1.12+) into a markdown insight sheet.

The sheet is working material for an article, not a finished text: it lays out
how the match split into phases, the ranked insight candidates (each with its
numbers, its caveat and the questions to settle before writing it), the pressure
and window tables and the droughts, with players shown as name and shirt number.
It adds no new analysis; every figure comes from the pack tables.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from src.metrics import pitch_zones as pz

REQUIRED_TABLES = (
    "match-phases",
    "window-profile",
    "pressure-profile",
    "shot-droughts",
    "insight-candidates",
)
OPTIONAL_TABLES = ("pressure-touches", "window-positions")

MATCH_LABEL = "match"
MAX_PLAYER_ROWS = 14

# What to settle before an insight becomes a sentence in an article. These are
# questions on purpose: the pack says what happened, the author decides why.
CHECK_BEFORE_WRITING = {
    "pressure_hotspot_player": (
        "Which lane and third did he play in (pressure grows towards the opponent's area)? "
        "Who closed him down, and can you name one moment a reader could spot live?"
    ),
    "pressure_free_players": (
        "Was the opponent deliberately keeping away from them, or were their touches simply deep and "
        "unhurried? Check their third of the pitch before calling it a shield."
    ),
    "pressure_lane_gap": (
        "Is the lane pressed more because the team attacked there most, or because the opponent was "
        "strongest there? Compare the same lane in the other team and across phases."
    ),
    "block_height_shift": (
        "What changed at the boundary of the two phases: score, substitutions, the opponent's touches "
        "per minute? A deeper block against more opposing touches is a retreat; against fewer, check the stoppages."
    ),
    "team_length_shift": (
        "Did the team stretch because the block dropped or because the forwards stayed high? "
        "Read it with the block height of the same phases and with who came on."
    ),
    "shot_drought": (
        "Was it the opponent's block, the team's own access (final-third touches, box touches, crosses) "
        "or the match state? The drought row carries all three for the same stretch."
    ),
    "box_touch_drought": (
        "Compare final-third touches with box touches: the gap is where the opponent's last line held. "
        "Was the team crossing instead of getting in?"
    ),
}

DEFINITIONS = (
    "**Pressed touch**: a touch followed by an opponent intervention (tackle, challenge, interception, "
    "blocked pass, recovery, clearance, committed foul) within 6 s and 10 m in the same period. It includes "
    "balls lost and recovered next to the touch: read it as where the opponent got hands on the play, not as "
    "intent to press. Touches deep in the own third are rarely pressed, so compare like with like.",
    "**Block height**: median distance from the own goal (m) of the team's own interventions, goalkeepers "
    "excluded. **Team length**: deepest to highest player position, only when at least 7 players qualify.",
    "**Position**: where a player acts (median of touches and interventions), not where he stands. "
    "Lanes are seen from the team's own goal: right is the team's right.",
    "**Phases** are cut at goals, dismissals and substitution clusters; durations are minutes of play, "
    "so first-half stoppage time is not counted twice. Starred rows (*) have too few actions to lean on.",
    "Everything is one match: it says what happened, not what either team usually does.",
)


class PackTooOldError(ValueError):
    """The pack was built before the window analysis tables existed."""


def load_pack(path: str | Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Read the window-analysis tables and the analysis summary out of a pack zip."""
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        missing = [name for name in REQUIRED_TABLES if f"tables/{name}.csv" not in names]
        if missing:
            raise PackTooOldError(
                "this pack has no window-analysis tables (pack schema 1.12 or later is needed): "
                + ", ".join(missing)
            )
        tables = {
            name: pd.read_csv(io.BytesIO(archive.read(f"tables/{name}.csv")))
            for name in (*REQUIRED_TABLES, *OPTIONAL_TABLES)
            if f"tables/{name}.csv" in names
        }
        summary = (
            json.loads(archive.read("analysis-summary.json").decode("utf-8"))
            if "analysis-summary.json" in names
            else {}
        )
    return tables, summary


# --------------------------------------------------------------------------- helpers


def _blank(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or (isinstance(value, str) and not value.strip())


def _flag(value) -> bool:
    """A boolean read back from a CSV, where it may be a bool or the text True/False."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1"}
    return False if _blank(value) else bool(value)


def _num(value, digits: int = 1, suffix: str = "") -> str:
    if _blank(value):
        return "–"
    return f"{float(value):.{digits}f}{suffix}"


def _p_value(value) -> str:
    if _blank(value):
        return "–"
    value = float(value)
    if value < 0.001:
        return "<0.001"
    return f"{value:.4f}" if value < 0.01 else f"{value:.3f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows._"
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
    out.extend("| " + " | ".join(str(cell).replace("|", "/") for cell in row) + " |" for row in rows)
    return "\n".join(out)


def _arrow(scope: str) -> str:
    return str(scope).replace("->", " → ")


def _jersey_lookup(tables: dict[str, pd.DataFrame]) -> dict[tuple[str, str], int]:
    """(team, player) -> shirt number, from the tables that carry both."""
    lookup: dict[tuple[str, str], int] = {}
    for name in ("window-positions", "pressure-touches"):
        frame = tables.get(name)
        if frame is None or frame.empty or not {"team_name", "player", "jersey"}.issubset(frame.columns):
            continue
        for team, player, jersey in frame[["team_name", "player", "jersey"]].drop_duplicates().itertuples(index=False):
            if not _blank(jersey) and (team, player) not in lookup:
                lookup[(team, player)] = int(float(jersey))
    return lookup


def _player(team: str, name: str, jerseys: dict[tuple[str, str], int]) -> str:
    number = jerseys.get((team, name))
    return f"{name} (#{number})" if number is not None else str(name)


def _teams(tables: dict[str, pd.DataFrame], summary: dict[str, Any]) -> list[str]:
    match = summary.get("match") or {}
    named = [match.get("home_team"), match.get("away_team")]
    if all(named):
        return list(named)
    phases = tables["match-phases"]
    if not phases.empty and {"home_team", "away_team"}.issubset(phases.columns):
        return [phases.iloc[0]["home_team"], phases.iloc[0]["away_team"]]
    return list(dict.fromkeys(tables["window-profile"]["team_name"].tolist()))


def _parse_numbers(raw) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if _blank(raw):
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


# --------------------------------------------------------------------------- sections


def _header(summary, teams) -> list[str]:
    match = summary.get("match") or {}
    goals_block = summary.get("score_and_goals") or {}
    home, away = teams[0], teams[1] if len(teams) > 1 else ""
    title = f"{home} {goals_block.get('home_score', '?')}–{goals_block.get('away_score', '?')} {away}"
    meta = [str(match[key])[:10] if key == "date" else str(match[key]) for key in ("competition", "date") if match.get(key)]
    lines = [f"# {title} — insight sheet", ""]
    if meta:
        lines += [" · ".join(meta), ""]
    lines += [
        "> Working material for an article: figures come from the Match Analysis Pack, the questions are "
        "what to settle before a figure becomes a sentence. One match only.",
        "",
    ]
    goals = goals_block.get("goals") or []
    if goals:
        lines += ["## Goals", ""]
        for goal in goals:
            how = ", ".join(str(goal[key]) for key in ("attack_type", "decisive_mechanism") if not _blank(goal.get(key)))
            creator = f", created by {goal['creator']}" if not _blank(goal.get("creator")) else ""
            lines.append(f"- {goal.get('minute', '?')}' {goal.get('scorer', '?')} ({goal.get('team', '?')}){' — ' + how if how else ''}{creator}")
        lines.append("")
    return lines


def _phases_section(tables, teams, jerseys) -> list[str]:
    phases = tables["match-phases"]
    rows = []
    for _, row in phases.iterrows():
        subs = "; ".join(
            f"{team}: " + ", ".join(_player(team, name.strip(), jerseys) for name in str(row[column]).split(";"))
            for team, column in zip(teams, ("home_subs_on", "away_subs_on"))
            if not _blank(row.get(column))
        )
        start = f"{int(row['home_score'])}–{int(row['away_score'])}"
        end = f"{int(row['home_score_end'])}–{int(row['away_score_end'])}"
        rows.append([
            f"{row['label']}{' (short)' if _flag(row.get('short')) else ''}",
            _num(row["play_minutes"]),
            start if start == end else f"{start} → {end}",
            str(row["trigger"]),
            f"{row['home_state']} / {row['away_state']}",
            subs or "–",
        ])
    return [
        "## How the match split",
        "",
        f"State is {teams[0]} / {teams[1] if len(teams) > 1 else ''}. Minutes are minutes of play.",
        "",
        _md_table(["Phase", "Play min", "Score", "Starts at", "State", "Players on"], rows),
        "",
    ]


def _insights_section(tables, teams, jerseys, only_team) -> list[str]:
    insights = tables["insight-candidates"]
    if only_team:
        insights = insights[insights["team_name"] == only_team]
    lines = ["## Insight candidates", ""]
    if insights.empty:
        return lines + ["_The rules found nothing worth flagging._", ""]
    lines += [
        f"{len(insights)} candidate(s), strongest first as ranked in the pack. A candidate is a number worth a "
        "look, not a conclusion.",
        "",
    ]
    caveats: list[str] = []
    for position, (_, row) in enumerate(insights.iterrows(), start=1):
        numbers = _parse_numbers(row.get("numbers"))
        lines += [f"### {position}. [{row['strength']}] {row['team_name']} — {str(row['kind']).replace('_', ' ')} ({_arrow(row['scope'])})", "", str(row["statement"]), ""]
        names = numbers.get("players") or ([numbers["player"]] if numbers.get("player") else [])
        if names:
            lines += ["Players: " + ", ".join(_player(row["team_name"], name, jerseys) for name in names), ""]
        evidence = " · ".join(
            f"{key} {value}" for key, value in numbers.items() if key != "player" and not isinstance(value, (list, dict))
        )
        if evidence:
            lines += [f"Evidence: {evidence}", ""]
        check = CHECK_BEFORE_WRITING.get(row["kind"])
        if check:
            lines += [f"Settle before writing: {check}", ""]
        caveat = row.get("caveat")
        if not _blank(caveat):
            if caveat not in caveats:
                caveats.append(caveat)
            lines += [f"Caveat [{caveats.index(caveat) + 1}]", ""]
    if caveats:
        lines += ["Caveats:", ""] + [f"{index}. {text}" for index, text in enumerate(caveats, start=1)] + [""]
    return lines


def _window_section(tables, teams, only_team) -> list[str]:
    profile = tables["window-profile"]
    lines = ["## Each team, phase by phase", ""]
    for team in teams:
        if only_team and team != only_team:
            continue
        mine = profile[profile["team_name"] == team].sort_values("phase_id")
        rows = []
        for _, row in mine.iterrows():
            label = "Whole match" if row["phase_label"] == MATCH_LABEL else str(row["phase_label"])
            star = "*" if _flag(row.get("low_sample")) else ""
            rows.append([
                label + star,
                _num(row["play_minutes"]),
                f"{int(row['touches'])} ({_num(row['touch_share_pct'], 0, '%')})",
                _num(row["own_half_touch_pct"], 0, "%"),
                _num(row["final_third_per_10min"]),
                _num(row["box_touches_per_10min"]),
                _num(row["shots"], 0),
                _num(row["block_height_m"], 0),
                _num(row["team_length_m"], 0),
                _num(row["pressed_touch_pct"], 1, "%"),
            ])
        lines += [
            f"### {team}",
            "",
            _md_table(
                ["Phase", "Min", "Touches (share)", "Own half", "Final third /10'", "Box touches /10'", "Shots", "Block m", "Length m", "Pressed"],
                rows,
            ),
            "",
        ]
    lines += ["Touch share is of both teams' touches in the phase. Block height and length are in metres from the team's own goal.", ""]
    return lines


def _ci(row) -> str:
    return f"{_num(row['pressed_pct'])}% [{_num(row['ci_low_pct'])}–{_num(row['ci_high_pct'])}]"


def _pressure_section(tables, teams, jerseys, only_team) -> list[str]:
    pressure = tables["pressure-profile"]
    whole = pressure[pressure["phase_label"] == MATCH_LABEL]
    lines = [
        "## Pressure",
        "",
        "Share of touches followed by an opponent intervention, with the 95% interval in brackets. "
        "`p` compares the row with the rest of the team (Fisher exact). Only outfield players' touches are "
        "counted here, so the totals are lower than the touches in the phase table.",
        "",
    ]
    for team in teams:
        if only_team and team != only_team:
            continue
        mine = whole[whole["team_name"] == team]
        total = mine[mine["group_type"] == "all"]
        if mine.empty:
            continue
        lines += [f"### {team}", ""]
        if not total.empty:
            row = total.iloc[0]
            lines += [f"Whole match: {int(row['pressed'])} of {int(row['touches'])} touches pressed, {_ci(row)}.", ""]
        for group_type, order, heading in (("lane", pz.LANES, "By lane"), ("third", pz.THIRDS, "By third")):
            part = mine[mine["group_type"] == group_type].copy()
            if part.empty:
                continue
            rank = {name: index for index, name in enumerate(order)}
            part = part.sort_values("group", key=lambda column: column.map(lambda name: rank.get(name, len(rank))))
            rows = [
                [str(r["group"]).replace("_", " ") + ("*" if _flag(r["low_sample"]) else ""), int(r["touches"]), int(r["pressed"]), _ci(r), _p_value(r["vs_rest_p"])]
                for _, r in part.iterrows()
            ]
            lines += [f"{heading}:", "", _md_table(["Group", "Touches", "Pressed", "Share [95%]", "p"], rows), ""]
        players = mine[mine["group_type"] == "player"]
        enough = players[~players["low_sample"].map(_flag)].sort_values(["pressed_pct", "touches"], ascending=[False, False])
        rows = [
            [_player(team, r["group"], jerseys), int(r["touches"]), int(r["pressed"]), _ci(r), _p_value(r["vs_rest_p"])]
            for _, r in enough.head(MAX_PLAYER_ROWS).iterrows()
        ]
        hidden = len(players) - len(enough)
        lines += ["By player (20+ touches):", "", _md_table(["Player", "Touches", "Pressed", "Share [95%]", "p"], rows), ""]
        if len(enough) > MAX_PLAYER_ROWS:
            lines += [f"{len(enough) - MAX_PLAYER_ROWS} more player(s) with 20+ touches are in `tables/pressure-profile.csv`.", ""]
        if hidden:
            lines += [f"{hidden} player(s) with fewer than 20 touches are left out.", ""]
    return lines


def _droughts_section(tables, teams, only_team) -> list[str]:
    droughts = tables["shot-droughts"]
    if only_team:
        droughts = droughts[droughts["team_name"] == only_team]
    lines = ["## Droughts", ""]
    if droughts.empty:
        return lines + ["_No stretch of 15 minutes or more without a shot or a box touch._", ""]
    rows = []
    for _, row in droughts.iterrows():
        rows.append([
            row["team_name"],
            "shots" if row["kind"] == "shots" else "box touches",
            f"{row['start_clock']}–{row['end_clock']}" + ("" if _flag(row["ends_with_event"]) else " (to the whistle)"),
            _num(row["play_minutes"]),
            int(row["final_third_touches"]),
            int(row["box_touches"]),
            int(row["crosses"]),
            int(row["opponent_shots"]),
        ])
    lines += [
        "Stretches of 15 play minutes or more between a team's consecutive shots (or box touches), "
        "with what the team and the opponent did meanwhile.",
        "",
        _md_table(["Team", "Without", "From–to", "Play min", "Final-third touches", "Box touches", "Crosses", "Opponent shots"], rows),
        "",
    ]
    return lines


def _closing_section(tables) -> list[str]:
    lines = ["## Using the other tables", ""]
    if "window-positions" in tables:
        lines.append(
            "- `tables/window-positions.csv`: every player's median position per phase in the team's own frame and "
            "in a shared frame (one team rotated 180°), ready for a head-to-head pitch."
        )
    if "pressure-touches" in tables:
        lines.append("- `tables/pressure-touches.csv`: one row per touch with who pressed it, to place pressure on a pitch.")
    lines += ["", "## Definitions", ""] + [f"- {text}" for text in DEFINITIONS] + [""]
    return lines


def render_insight_sheet(
    tables: dict[str, pd.DataFrame],
    summary: dict[str, Any] | None = None,
    *,
    team: str | None = None,
) -> str:
    """Render the markdown sheet; ``team`` limits the per-team sections and the candidates to one team."""
    summary = summary or {}
    missing = [name for name in REQUIRED_TABLES if name not in tables]
    if missing:
        raise PackTooOldError("missing tables: " + ", ".join(missing))
    teams = _teams(tables, summary)
    if team and team not in teams:
        raise ValueError(f"team {team!r} is not in this match ({', '.join(teams)})")
    jerseys = _jersey_lookup(tables)
    lines: list[str] = []
    lines += _header(summary, teams)
    lines += _phases_section(tables, teams, jerseys)
    lines += _insights_section(tables, teams, jerseys, team)
    lines += _window_section(tables, teams, team)
    lines += _pressure_section(tables, teams, jerseys, team)
    lines += _droughts_section(tables, teams, team)
    lines += _closing_section(tables)
    return "\n".join(lines).rstrip() + "\n"
