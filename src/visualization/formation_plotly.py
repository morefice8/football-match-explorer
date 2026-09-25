# In un nuovo file, es: src/visualization/formation_plotly.py
import logging
logger = logging.getLogger(__name__)

import plotly.graph_objects as go
import pandas as pd

from src.utils.formation_layouts import get_formation_layout_coords, get_formation_name
from src.visualization.plotly_branding import (
    MATCH_COMPARE_HEIGHT,
    MATCH_PITCH_MUTED,
    MATCH_PITCH_TEXT,
    MATCH_WARNING,
    add_attacking_direction,
    add_zero_state,
    apply_match_pitch_layout,
    get_team_palette,
)
from src.visualization import pitch_plots

# PLOT-02 — synchronized interactive formation timeline
# ---------------------------------------------------------------------------

FORMATION_TIMELINE_HEIGHT = 510

_FORMATION_EVENT_CODES = {
    "starting_xi": "XI",
    "goal": "G",
    "substitution": "SUB",
    "formation_change": "FORM",
    "dismissal": "RC",
}


def _timeline_flag_is_true(value):
    """Return True only for explicit truthy Opta qualifier values."""
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {
        "1", "true", "yes", "y", "red", "rc",
    }


def _timeline_row_flag(row, aliases):
    return any(
        column in row.index
        and _timeline_flag_is_true(row.get(column))
        for column in aliases
    )


def _timeline_safe_int(value, default=None):
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _timeline_id(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            pass
    return text or None


def _timeline_event_seconds(row):
    minute = _timeline_safe_int(row.get("timeMin"), 0)
    second = _timeline_safe_int(row.get("timeSec"), 0)
    if minute is None:
        return None
    return int(max(minute, 0) * 60 + max(second or 0, 0))


def _timeline_time_label(seconds):
    minute, second = divmod(int(max(seconds or 0, 0)), 60)
    return (
        f"{minute}′ {second:02d}″"
        if second
        else f"{minute}′"
    )


def _timeline_formation_id(value, default=None):
    value = _timeline_safe_int(value, default)
    return default if value is None else int(value)


def _timeline_formation_name(formation_id):
    if formation_id is None:
        return "Unknown shape"
    try:
        name = get_formation_name(int(formation_id))
    except (TypeError, ValueError, KeyError):
        name = None
    return str(name) if name else f"Formation {formation_id}"


def _timeline_extract_player_positions(event):
    """Extract {player_id: Opta slot} from type 34 / 40 rows."""
    players_raw = event.get("Involved")
    positions_raw = event.get("Team player formation")
    if (
        players_raw is None
        or positions_raw is None
        or pd.isna(players_raw)
        or pd.isna(positions_raw)
    ):
        return {}

    player_ids = [
        _timeline_id(value)
        for value in str(players_raw).split(",")
    ]
    positions = [
        _timeline_safe_int(value)
        for value in str(positions_raw).split(",")
    ]
    if len(player_ids) != len(positions):
        return {}

    return {
        player_id: int(position)
        for player_id, position in zip(player_ids, positions)
        if player_id and position is not None and position > 0
    }


def _timeline_copy_state(state):
    return {
        "formation_id": _timeline_formation_id(
            state.get("formation_id")
        ),
        "players": {
            str(player_id): int(position)
            for player_id, position in (
                state.get("players", {}) or {}
            ).items()
            if player_id is not None and position is not None
        },
    }


def _timeline_player_data(df):
    player_map = {}
    if df is None or df.empty or "playerId" not in df.columns:
        return player_map

    for _, row in df.iterrows():
        player_id = _timeline_id(row.get("playerId"))
        if not player_id:
            continue

        current = player_map.setdefault(
            player_id,
            {"name": "Unknown", "jersey": "?"},
        )

        name = row.get("playerName")
        if name is not None and not pd.isna(name):
            name = str(name).strip()
            if name:
                current["name"] = name

        jersey = _timeline_safe_int(
            row.get("Mapped Jersey Number")
        )
        if jersey is not None:
            current["jersey"] = str(jersey)

    return player_map


def _timeline_player_label(player_map, player_id, fallback_name=None):
    player_id = _timeline_id(player_id)
    info = (player_map or {}).get(player_id, {})
    name = info.get("name")
    if not name or name == "Unknown":
        name = (
            str(fallback_name)
            if fallback_name is not None
            and not pd.isna(fallback_name)
            else "Unknown"
        )
    jersey = info.get("jersey", "?")
    return (
        f"#{jersey} {name}"
        if jersey and jersey != "?"
        else str(name)
    )


def _timeline_team_side(
    row,
    *,
    home_id,
    away_id,
    home_team,
    away_team,
):
    contestant_id = _timeline_id(row.get("contestantId"))
    if contestant_id and contestant_id == home_id:
        return "home"
    if contestant_id and contestant_id == away_id:
        return "away"

    team_name = row.get("team_name")
    if team_name == home_team:
        return "home"
    if team_name == away_team:
        return "away"
    return None


def _timeline_is_dismissal(row):
    return (
        _timeline_row_flag(row, ("Red card", "Red Card"))
        or _timeline_row_flag(
            row,
            ("Second yellow", "Second Yellow", "Second yellow card"),
        )
    )


def _timeline_dismissal_label(row):
    if _timeline_row_flag(
        row,
        ("Second yellow", "Second Yellow", "Second yellow card"),
    ):
        return "Second yellow"
    return "Red card"


def _timeline_is_own_goal(row):
    return _timeline_row_flag(
        row,
        ("Own goal", "Own Goal"),
    )


def _timeline_resolve_sub_on(df, sub_off):
    """
    Pair typeId 18 (Player Off) with typeId 19 (Player On).
    related_eventId is authoritative; same-time matching is fallback only.
    """
    event_id = _timeline_id(sub_off.get("eventId"))
    player_on = df[
        pd.to_numeric(df["typeId"], errors="coerce").eq(19)
    ].copy()

    if player_on.empty:
        return None

    if event_id and "related_eventId" in player_on.columns:
        linked = player_on.loc[
            player_on["related_eventId"].map(_timeline_id).eq(event_id)
        ]
        if not linked.empty:
            return linked.sort_values(
                "event_sequence_index",
                kind="stable",
            ).iloc[0]

    contestant_id = _timeline_id(sub_off.get("contestantId"))
    candidates = player_on.copy()
    if contestant_id and "contestantId" in candidates.columns:
        candidates = candidates[
            candidates["contestantId"].map(_timeline_id).eq(
                contestant_id
            )
        ]

    off_time = _timeline_event_seconds(sub_off)
    if off_time is not None and not candidates.empty:
        candidate_times = candidates.apply(
            _timeline_event_seconds,
            axis=1,
        )
        delta = candidate_times.sub(off_time).abs()
        nearby = candidates.loc[delta.le(2)]
        if not nearby.empty:
            return nearby.sort_values(
                "event_sequence_index",
                kind="stable",
            ).iloc[0]

    return None


def _timeline_event_code(events):
    counts = {}
    for event in events or []:
        code = _FORMATION_EVENT_CODES.get(event.get("kind"))
        if code:
            counts[code] = counts.get(code, 0) + 1

    labels = []
    for code in ("XI", "G", "SUB", "FORM", "RC"):
        count = counts.get(code, 0)
        if count:
            labels.append(f"{code}×{count}" if count > 1 else code)
    return "·".join(labels)


def _timeline_event_payload(
    kind,
    team,
    description,
    *,
    row=None,
    seconds=None,
):
    """Store readable exact event time inside a minute-level match moment."""
    exact_seconds = seconds

    if exact_seconds is None and row is not None:
        exact_seconds = _timeline_event_seconds(row)

    return {
        "kind": kind,
        "team": team,
        "description": description,
        "event_time_label": (
            _timeline_time_label(exact_seconds)
            if exact_seconds is not None
            else ""
        ),
    }


def _timeline_snapshot(
    *,
    seconds,
    score_home,
    score_away,
    home_state,
    away_state,
    home_highlights,
    away_highlights,
    events,
):
    home_copy = _timeline_copy_state(home_state)
    away_copy = _timeline_copy_state(away_state)

    return {
        "time_seconds": int(seconds),
        "time_label": _timeline_time_label(seconds),
        "score_home": int(score_home),
        "score_away": int(score_away),
        "score": f"{int(score_home)} – {int(score_away)}",
        "events": list(events or []),
        "home_state": home_copy,
        "away_state": away_copy,
        "home_highlights": sorted(
            str(value) for value in (home_highlights or set())
        ),
        "away_highlights": sorted(
            str(value) for value in (away_highlights or set())
        ),
        "home_formation_name": _timeline_formation_name(
            home_copy.get("formation_id")
        ),
        "away_formation_name": _timeline_formation_name(
            away_copy.get("formation_id")
        ),
    }


def build_formation_timeline_model(df_processed, match_info):
    """
    Build one compact synchronized timeline for Home and Away.

    Moments include Starting XI, goals, substitutions, formation changes and
    dismissals. Ordinary yellow cards are intentionally excluded.
    """
    if df_processed is None or df_processed.empty:
        return {
            "home_team": "Home",
            "away_team": "Away",
            "home_id": None,
            "away_id": None,
            "player_data": {},
            "moments": [],
            "match_end_seconds": 0,
        }

    df = df_processed.copy()
    if "event_sequence_index" not in df.columns:
        df = df.reset_index().rename(
            columns={"index": "event_sequence_index"}
        )

    sequence_values = pd.to_numeric(
        df["event_sequence_index"],
        errors="coerce",
    )
    fallback_sequence = pd.Series(
        range(len(df)),
        index=df.index,
        dtype=float,
    )
    df["event_sequence_index"] = sequence_values.fillna(
        fallback_sequence
    )

    type_ids = pd.to_numeric(df["typeId"], errors="coerce")

    start_events = df.loc[
        type_ids.eq(34)
    ].sort_values(
        "event_sequence_index",
        kind="stable",
    )

    if len(start_events) < 2:
        raise ValueError(
            "Could not find starting formation events for both teams."
        )

    match_info = match_info or {}
    home_team = match_info.get("hteamName") or "Home"
    away_team = match_info.get("ateamName") or "Away"

    team_series = (
        start_events["team_name"]
        if "team_name" in start_events.columns
        else pd.Series(index=start_events.index, dtype=object)
    )

    home_candidates = start_events[team_series.eq(home_team)]
    away_candidates = start_events[team_series.eq(away_team)]

    home_start = (
        home_candidates.iloc[0]
        if not home_candidates.empty
        else start_events.iloc[0]
    )

    if not away_candidates.empty:
        away_start = away_candidates.iloc[0]
    else:
        remaining = start_events.drop(
            index=home_start.name,
            errors="ignore",
        )
        away_start = (
            remaining.iloc[0]
            if not remaining.empty
            else start_events.iloc[1]
        )

    if home_team == "Home":
        value = home_start.get("team_name")
        if value is not None and not pd.isna(value):
            home_team = str(value)

    if away_team == "Away":
        value = away_start.get("team_name")
        if value is not None and not pd.isna(value):
            away_team = str(value)

    home_id = _timeline_id(home_start.get("contestantId"))
    away_id = _timeline_id(away_start.get("contestantId"))

    home_state = {
        "formation_id": _timeline_formation_id(
            home_start.get("Team formation")
        ),
        "players": _timeline_extract_player_positions(home_start),
    }
    away_state = {
        "formation_id": _timeline_formation_id(
            away_start.get("Team formation")
        ),
        "players": _timeline_extract_player_positions(away_start),
    }

    player_map = _timeline_player_data(df)
    score_home = 0
    score_away = 0

    moments = [
        _timeline_snapshot(
            seconds=0,
            score_home=0,
            score_away=0,
            home_state=home_state,
            away_state=away_state,
            home_highlights=set(),
            away_highlights=set(),
            events=[
                _timeline_event_payload(
                    "starting_xi",
                    "both",
                    "Starting XI and opening team structures.",
                    seconds=0,
                )
            ],
        )
    ]

    relevant = df.loc[
        type_ids.isin([16, 17, 18, 40])
    ].copy()

    if not relevant.empty:
        relevant["_timeline_seconds"] = relevant.apply(
            _timeline_event_seconds,
            axis=1,
        )
        relevant = relevant[
            relevant["_timeline_seconds"].notna()
        ].sort_values(
            ["_timeline_seconds", "event_sequence_index"],
            kind="stable",
        )

        # The analytical timeline works at football-minute resolution.
        # Multiple feed events a few seconds apart during the same stoppage
        # therefore become one navigable match moment, while each event keeps
        # its exact timestamp in the detail panel.
        relevant["_timeline_minute"] = (
            relevant["_timeline_seconds"]
            .floordiv(60)
            .astype(int)
            .mul(60)
        )

        grouped_relevant = relevant.groupby(
            "_timeline_minute",
            sort=True,
        )
    else:
        grouped_relevant = []

    for seconds, group in grouped_relevant:
        seconds = int(seconds)
        home_highlights = set()
        away_highlights = set()
        moment_events = []

        for _, event in group.iterrows():
            type_id = _timeline_safe_int(event.get("typeId"))

            side = _timeline_team_side(
                event,
                home_id=home_id,
                away_id=away_id,
                home_team=home_team,
                away_team=away_team,
            )

            team_name = (
                home_team
                if side == "home"
                else away_team
                if side == "away"
                else str(event.get("team_name", "Unknown team"))
            )

            state = (
                home_state
                if side == "home"
                else away_state
                if side == "away"
                else None
            )

            highlight_set = (
                home_highlights
                if side == "home"
                else away_highlights
                if side == "away"
                else set()
            )

            if type_id == 16:
                own_goal = _timeline_is_own_goal(event)
                scoring_side = side
                if own_goal:
                    scoring_side = (
                        "away"
                        if side == "home"
                        else "home"
                        if side == "away"
                        else None
                    )

                if scoring_side == "home":
                    score_home += 1
                elif scoring_side == "away":
                    score_away += 1

                scorer = _timeline_player_label(
                    player_map,
                    event.get("playerId"),
                    event.get("playerName"),
                )
                description = (
                    f"Own goal · {scorer} ({team_name})"
                    if own_goal
                    else f"Goal · {scorer} ({team_name})"
                )
                moment_events.append(
                    _timeline_event_payload(
                        "goal",
                        side,
                        description,
                        row=event,
                    )
                )

            elif type_id == 18:
                sub_on = _timeline_resolve_sub_on(df, event)
                player_off_id = _timeline_id(
                    event.get("playerId")
                )
                player_on_id = (
                    _timeline_id(sub_on.get("playerId"))
                    if sub_on is not None
                    else None
                )

                if state is not None:
                    position = (
                        state["players"].pop(player_off_id, None)
                        if player_off_id
                        else None
                    )
                    if player_on_id and position is not None:
                        state["players"][player_on_id] = int(position)
                        highlight_set.add(player_on_id)

                player_off = _timeline_player_label(
                    player_map,
                    player_off_id,
                    event.get("playerName"),
                )

                if sub_on is not None:
                    player_on = _timeline_player_label(
                        player_map,
                        player_on_id,
                        sub_on.get("playerName"),
                    )
                    description = (
                        f"{team_name} · {player_on} on for {player_off}"
                    )
                else:
                    description = (
                        f"{team_name} · {player_off} off "
                        "(incoming player link unavailable)"
                    )

                moment_events.append(
                    _timeline_event_payload(
                        "substitution",
                        side,
                        description,
                        row=event,
                    )
                )

            elif type_id == 40:
                if state is None:
                    continue

                previous_players = dict(state.get("players", {}))
                new_formation_id = _timeline_formation_id(
                    event.get("Team formation"),
                    state.get("formation_id"),
                )
                extracted_players = _timeline_extract_player_positions(
                    event
                )

                if new_formation_id is not None:
                    state["formation_id"] = new_formation_id

                if extracted_players:
                    state["players"] = extracted_players
                    for player_id, position in extracted_players.items():
                        if previous_players.get(player_id) != position:
                            highlight_set.add(player_id)

                formation_name = _timeline_formation_name(
                    state.get("formation_id")
                )

                moment_events.append(
                    _timeline_event_payload(
                        "formation_change",
                        side,
                        (
                            f"{team_name} · shape change to "
                            f"{formation_name}"
                        ),
                        row=event,
                    )
                )

            elif type_id == 17 and _timeline_is_dismissal(event):
                player_id = _timeline_id(event.get("playerId"))

                if state is not None and player_id:
                    state["players"].pop(player_id, None)

                player = _timeline_player_label(
                    player_map,
                    player_id,
                    event.get("playerName"),
                )
                card_label = _timeline_dismissal_label(event)
                moment_events.append(
                    _timeline_event_payload(
                        "dismissal",
                        side,
                        (
                            f"{card_label} · {player} ({team_name})"
                        ),
                        row=event,
                    )
                )

        # A group containing only normal yellow cards is ignored.
        if not moment_events:
            continue

        snapshot = _timeline_snapshot(
            seconds=seconds,
            score_home=score_home,
            score_away=score_away,
            home_state=home_state,
            away_state=away_state,
            home_highlights=home_highlights,
            away_highlights=away_highlights,
            events=moment_events,
        )

        if seconds == 0:
            snapshot["events"] = moments[0]["events"] + snapshot["events"]
            moments[0] = snapshot
        else:
            moments.append(snapshot)

    all_times = df.apply(_timeline_event_seconds, axis=1)
    valid_times = [
        int(value)
        for value in all_times.tolist()
        if value is not None and not pd.isna(value)
    ]
    last_moment = moments[-1]["time_seconds"] if moments else 0

    return {
        "home_team": str(home_team),
        "away_team": str(away_team),
        "home_id": home_id,
        "away_id": away_id,
        "player_data": player_map,
        "moments": moments,
        "match_end_seconds": int(
            max(valid_times + [last_moment, 1])
        ),
    }


def get_formation_timeline_moment(model, selected_time):
    moments = (model or {}).get("moments", []) or []
    if not moments:
        return None

    try:
        target = float(selected_time)
    except (TypeError, ValueError):
        target = float(moments[0]["time_seconds"])

    return min(
        moments,
        key=lambda moment: abs(
            float(moment.get("time_seconds", 0)) - target
        ),
    )


def build_formation_slider_marks(model):
    # Every analytical moment remains selectable. Rail text is deliberately
    # sparse: one semantic label per moment, with a fixed priority so mixed
    # moments never become strings such as "SUB·FORM".
    moments = (model or {}).get("moments", []) or []

    priority = {
        "dismissal": (4, "RC"),
        "goal": (3, "G"),
        "formation_change": (2, "FORM"),
        "starting_xi": (1, "XI"),
    }

    marks = {}
    labelled = []

    for moment in moments:
        seconds = int(moment.get("time_seconds", 0))
        events = moment.get("events", []) or []
        kinds = {
            event.get("kind")
            for event in events
        }

        best_kind = None
        best_priority = 0
        best_code = ""

        for kind in kinds:
            kind_priority, code = priority.get(
                kind,
                (0, ""),
            )

            if kind_priority > best_priority:
                best_kind = kind
                best_priority = kind_priority
                best_code = code

        marks[seconds] = {
            "label": "",
            "style": {
                "fontWeight": "700",
            },
        }

        if best_kind:
            labelled.append(
                {
                    "seconds": seconds,
                    "priority": best_priority,
                    "code": best_code,
                }
            )

    # Labels closer than two football minutes are hard to read on a compact
    # 90-minute rail. Keep every dot, but show only the strongest nearby label.
    # If priorities tie, keep the later moment because it reflects the most
    # recent tactical state.
    min_gap_seconds = 120
    visible = []

    for candidate in labelled:
        if (
            not visible
            or candidate["seconds"] - visible[-1]["seconds"]
            >= min_gap_seconds
        ):
            visible.append(candidate)
            continue

        previous = visible[-1]

        replace_previous = (
            candidate["priority"] > previous["priority"]
            or (
                candidate["priority"] == previous["priority"]
                and candidate["seconds"] > previous["seconds"]
            )
        )

        if replace_previous:
            visible[-1] = candidate

    for item in visible:
        seconds = item["seconds"]
        marks[seconds]["label"] = (
            f"{seconds // 60}′ {item['code']}"
        )

    return marks


def plot_formation_timeline_state(
    state,
    player_data_map,
    *,
    is_away=False,
    highlighted_players=None,
):
    """
    Render one team state using the shared Match Plot Design System.

    Home and Away deliberately share the canonical team-relative
    left-to-right orientation for direct shape comparison.
    """
    palette = get_team_palette(is_away=is_away)
    fig = go.Figure()

    pitch_shapes = pitch_plots.get_plotly_pitch_shapes(
        "rgba(255,255,255,0.24)",
        "rgba(255,255,255,0.82)",
    )

    state = state or {}
    formation_id = _timeline_formation_id(
        state.get("formation_id")
    )
    highlighted = {
        str(value)
        for value in (highlighted_players or [])
    }

    player_ids = []
    x_values = []
    y_values = []
    jerseys = []
    surnames = []
    customdata = []

    for player_id, position in (
        state.get("players", {}) or {}
    ).items():
        player_id = str(player_id)
        try:
            x, y = get_formation_layout_coords(
                formation_id,
                int(position),
            )
        except (TypeError, ValueError, KeyError):
            x, y = None, None

        if x is None or y is None:
            continue

        info = (player_data_map or {}).get(player_id, {})
        name = str(info.get("name", "Unknown"))
        jersey = str(info.get("jersey", "?"))
        surname = _mean_positions_compact_name(
            name
        )

        player_ids.append(player_id)
        x_values.append(float(x))
        y_values.append(float(y))
        jerseys.append(jersey)
        surnames.append(surname)
        customdata.append([name, jersey, int(position)])

    if x_values:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers+text",
                marker=dict(
                    size=34,
                    color=palette["primary"],
                    line=dict(
                        color="rgba(255,255,255,0.92)",
                        width=1.8,
                    ),
                ),
                text=jerseys,
                textposition="middle center",
                textfont=dict(
                    color="#ffffff",
                    size=11,
                    family="Inter, Arial, sans-serif",
                ),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b>"
                    "<br>Jersey: %{customdata[1]}"
                    "<br>Formation slot: %{customdata[2]}"
                    "<extra></extra>"
                ),
                showlegend=False,
                name="Players",
            )
        )

        # High-contrast player labels are easier to scan than free-floating
        # white text, especially when two lines are vertically compact.
        for x, y, surname in zip(
            x_values,
            y_values,
            surnames,
        ):
            label = str(surname)
            if len(label) > 14:
                label = label[:13] + "…"

            place_above = y < 14

            fig.add_annotation(
                x=x,
                y=y,
                text=f"<b>{label}</b>",
                showarrow=False,
                xanchor="center",
                yanchor=(
                    "bottom"
                    if place_above
                    else "top"
                ),
                yshift=(
                    23
                    if place_above
                    else -23
                ),
                font=dict(
                    family="Inter, Arial, sans-serif",
                    color="#ffffff",
                    size=10.5,
                ),
                bgcolor="rgba(16,47,69,0.92)",
                bordercolor="rgba(255,255,255,0.20)",
                borderwidth=1,
                borderpad=2,
                opacity=0.98,
            )

        highlight_x = []
        highlight_y = []
        for player_id, x, y in zip(
            player_ids,
            x_values,
            y_values,
        ):
            if player_id in highlighted:
                highlight_x.append(x)
                highlight_y.append(y)

        if highlight_x:
            fig.add_trace(
                go.Scatter(
                    x=highlight_x,
                    y=highlight_y,
                    mode="markers",
                    marker=dict(
                        size=42,
                        color="rgba(0,0,0,0)",
                        line=dict(
                            color=MATCH_WARNING,
                            width=4,
                        ),
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                    name="Changed player",
                )
            )
    else:
        add_zero_state(
            fig,
            "Formation positions unavailable",
        )

    fig.add_annotation(
        x=3,
        y=97,
        text=f"<b>{_timeline_formation_name(formation_id)}</b>",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(
            family="Inter, Arial, sans-serif",
            color=MATCH_PITCH_TEXT,
            size=13,
        ),
        bgcolor="rgba(16,47,69,0.72)",
        borderpad=4,
    )

    apply_match_pitch_layout(
        fig,
        pitch_shapes=pitch_shapes,
        height=FORMATION_TIMELINE_HEIGHT,
        showlegend=False,
        header=False,
    )
    add_attacking_direction(fig, dark=True)

    return fig

# PLOT-03 — median positions + spatial dispersion
# ---------------------------------------------------------------------------

MEAN_POSITIONS_PLOT_HEIGHT = 560



_MEAN_POSITION_NAME_PARTICLES = {
    "da",
    "das",
    "de",
    "del",
    "della",
    "di",
    "dos",
    "du",
    "la",
    "le",
    "van",
    "von",
    "der",
    "den",
}


def _mean_positions_compact_name(
    full_name,
):
    """
    Keep surname particles in pitch labels.

    Examples:
      * Giovanni Di Lorenzo -> Di Lorenzo
      * Kevin De Bruyne -> De Bruyne
      * Donny van de Beek -> van de Beek
    """
    text = str(
        full_name or ""
    ).strip()

    if not text:
        return "Unknown"

    parts = [
        part
        for part in text.split()
        if part
    ]

    if len(parts) == 1:
        return parts[0]

    start = len(
        parts
    ) - 1

    while (
        start > 0
        and parts[
            start - 1
        ].lower()
        in _MEAN_POSITION_NAME_PARTICLES
    ):
        start -= 1

    return " ".join(
        parts[
            start:
        ]
    )


def _mean_positions_rgba(
    hex_color,
    alpha,
):
    color = str(hex_color).lstrip("#")

    if len(color) != 6:
        return (
            f"rgba(21,151,194,{alpha})"
        )

    try:
        r = int(
            color[0:2],
            16,
        )
        g = int(
            color[2:4],
            16,
        )
        b = int(
            color[4:6],
            16,
        )
    except ValueError:
        return (
            f"rgba(21,151,194,{alpha})"
        )

    return (
        f"rgba({r},{g},{b},{alpha})"
    )


def _mean_positions_marker_size(
    touch_share,
):
    share = pd.to_numeric(
        touch_share,
        errors="coerce",
    )

    if pd.isna(share):
        share = 0.0

    # Touch share remains visible, but the range is intentionally restrained
    # because Full Match can contain 14-16 eligible players.
    return max(
        26.0,
        min(
            44.0,
            26.0 + float(share) * 1.25,
        ),
    )


def _mean_positions_label_offsets(
    df_player_profile,
):
    """
    Spread surname labels around dense median-location clusters.

    The player marker stays at the true median position; only the annotation
    box moves. This improves readability without changing any metric.
    """
    if (
        df_player_profile is None
        or df_player_profile.empty
    ):
        return []

    points = [
        (
            float(row["median_x"]),
            float(row["median_y"]),
        )
        for _, row in df_player_profile.iterrows()
    ]

    clusters = []
    visited = set()

    for index in range(len(points)):
        if index in visited:
            continue

        cluster = []
        queue = [index]
        visited.add(index)

        while queue:
            current = queue.pop(0)
            cluster.append(current)

            x_current, y_current = points[current]

            for other in range(len(points)):
                if other in visited:
                    continue

                x_other, y_other = points[other]

                if (
                    abs(x_current - x_other) <= 9.0
                    and abs(y_current - y_other) <= 9.0
                ):
                    visited.add(other)
                    queue.append(other)

        clusters.append(cluster)

    # Pixel offsets around the real marker location.
    candidates = [
        {
            "xshift": 0,
            "yshift": -24,
            "xanchor": "center",
            "yanchor": "top",
        },
        {
            "xshift": 0,
            "yshift": 24,
            "xanchor": "center",
            "yanchor": "bottom",
        },
        {
            "xshift": -22,
            "yshift": 0,
            "xanchor": "right",
            "yanchor": "middle",
        },
        {
            "xshift": 22,
            "yshift": 0,
            "xanchor": "left",
            "yanchor": "middle",
        },
        {
            "xshift": -18,
            "yshift": 20,
            "xanchor": "right",
            "yanchor": "bottom",
        },
        {
            "xshift": 18,
            "yshift": -20,
            "xanchor": "left",
            "yanchor": "top",
        },
    ]

    offsets = [
        None
        for _ in points
    ]

    for cluster in clusters:
        ordered = sorted(
            cluster,
            key=lambda idx: (
                points[idx][1],
                points[idx][0],
            ),
        )

        for local_index, player_index in enumerate(ordered):
            offsets[player_index] = dict(
                candidates[
                    local_index
                    % len(candidates)
                ]
            )

    return offsets


def plot_mean_positions_profile_plotly(
    df_player_profile,
    team_summary,
    *,
    is_away=False,
):
    """
    Territorial occupation profile, not a formation.

    Centre = player median touch location.
    Ellipse = x/y interquartile touch footprint.
    Marker size = share of the team's selected-period touch events.
    """
    palette = get_team_palette(
        is_away=is_away
    )
    primary = palette["primary"]

    fig = go.Figure()

    pitch_shapes = (
        pitch_plots.get_plotly_pitch_shapes(
            "rgba(255,255,255,0.24)",
            "rgba(255,255,255,0.82)",
        )
    )

    profile = (
        df_player_profile.copy()
        if df_player_profile is not None
        else pd.DataFrame()
    )

    summary = (
        dict(team_summary)
        if team_summary
        else {}
    )

    # ---------------------------------------------------------
    # TEAM FOOTPRINT
    # ---------------------------------------------------------

    required_bounds = (
        "structural_min_x",
        "structural_max_x",
        "structural_min_y",
        "structural_max_y",
    )

    if all(
        summary.get(key) is not None
        for key in required_bounds
    ):
        pitch_shapes.append(
            dict(
                type="rect",
                x0=summary[
                    "structural_min_x"
                ],
                x1=summary[
                    "structural_max_x"
                ],
                y0=summary[
                    "structural_min_y"
                ],
                y1=summary[
                    "structural_max_y"
                ],
                line=dict(
                    color=_mean_positions_rgba(
                        primary,
                        0.62,
                    ),
                    width=1.4,
                    dash="dot",
                ),
                fillcolor=_mean_positions_rgba(
                    primary,
                    0.055,
                ),
                layer="below",
            )
        )

    centroid_x = summary.get(
        "centroid_x"
    )
    centroid_y = summary.get(
        "centroid_y"
    )

    if (
        centroid_x is not None
        and centroid_y is not None
    ):
        pitch_shapes.append(
            dict(
                type="line",
                x0=float(
                    centroid_x
                ),
                x1=float(
                    centroid_x
                ),
                y0=0,
                y1=100,
                line=dict(
                    color=_mean_positions_rgba(
                        primary,
                        0.55,
                    ),
                    width=1.4,
                    dash="dash",
                ),
                layer="below",
            )
        )

    # ---------------------------------------------------------
    # INDIVIDUAL IQR FOOTPRINTS
    # ---------------------------------------------------------

    if not profile.empty:
        for _, row in profile.iterrows():
            median_x = float(
                row["median_x"]
            )
            median_y = float(
                row["median_y"]
            )

            half_x = max(
                float(
                    row.get(
                        "iqr_x",
                        0.0,
                    )
                ) / 2.0,
                1.4,
            )
            half_y = max(
                float(
                    row.get(
                        "iqr_y",
                        0.0,
                    )
                ) / 2.0,
                1.4,
            )

            pitch_shapes.append(
                dict(
                    type="circle",
                    x0=median_x - half_x,
                    x1=median_x + half_x,
                    y0=median_y - half_y,
                    y1=median_y + half_y,
                    line=dict(
                        color=_mean_positions_rgba(
                            primary,
                            0.22,
                        ),
                        width=1,
                    ),
                    fillcolor=_mean_positions_rgba(
                        primary,
                        0.045,
                    ),
                    layer="below",
                )
            )

        jersey_text = []
        surname_text = []
        marker_sizes = []
        customdata = []

        for _, row in profile.iterrows():
            jersey = row.get(
                "Mapped Jersey Number"
            )

            if (
                jersey is None
                or pd.isna(jersey)
            ):
                jersey_label = ""
            else:
                jersey_label = str(
                    int(jersey)
                )

            name = str(
                row.get(
                    "playerName",
                    "Unknown",
                )
            )
            surname = _mean_positions_compact_name(
                name
            )

            if len(surname) > 14:
                surname = (
                    surname[:13]
                    + "…"
                )

            jersey_text.append(
                jersey_label
            )
            surname_text.append(
                surname
            )
            marker_sizes.append(
                _mean_positions_marker_size(
                    row.get(
                        "touch_share",
                        0.0,
                    )
                )
            )
            customdata.append(
                [
                    name,
                    jersey_label,
                    row.get(
                        "positional_role",
                        "Unknown",
                    ),
                    float(
                        row.get(
                            "minutes_played",
                            0.0,
                        )
                    ),
                    int(
                        row.get(
                            "touch_count",
                            0,
                        )
                    ),
                    float(
                        row.get(
                            "touch_share",
                            0.0,
                        )
                    ),
                    float(
                        row.get(
                            "dispersion_m",
                            0.0,
                        )
                    ),
                ]
            )

        fig.add_trace(
            go.Scatter(
                x=profile[
                    "median_x"
                ],
                y=profile[
                    "median_y"
                ],
                mode="markers+text",
                marker=dict(
                    size=marker_sizes,
                    color=primary,
                    line=dict(
                        color=(
                            "rgba(255,255,255,0.94)"
                        ),
                        width=1.7,
                    ),
                ),
                text=jersey_text,
                textposition=(
                    "middle center"
                ),
                textfont=dict(
                    family=(
                        "Inter, Arial, sans-serif"
                    ),
                    color="#ffffff",
                    size=10.5,
                ),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b>"
                    "<br>Jersey: %{customdata[1]}"
                    "<br>Role: %{customdata[2]}"
                    "<br>Minutes: %{customdata[3]:.1f}"
                    "<br>Touches: %{customdata[4]}"
                    "<br>Touch share: %{customdata[5]:.1f}%"
                    "<br>Median dispersion: %{customdata[6]:.1f} m"
                    "<extra></extra>"
                ),
                showlegend=False,
                name="Median locations",
            )
        )

        label_offsets = (
            _mean_positions_label_offsets(
                profile
            )
        )

        for (
            x,
            y,
            surname,
            offset,
        ) in zip(
            profile["median_x"],
            profile["median_y"],
            surname_text,
            label_offsets,
        ):
            offset = (
                offset
                or {
                    "xshift": 0,
                    "yshift": -24,
                    "xanchor": "center",
                    "yanchor": "top",
                }
            )

            fig.add_annotation(
                x=float(x),
                y=float(y),
                text=(
                    f"<b>{surname}</b>"
                ),
                showarrow=False,
                xanchor=offset[
                    "xanchor"
                ],
                yanchor=offset[
                    "yanchor"
                ],
                xshift=offset[
                    "xshift"
                ],
                yshift=offset[
                    "yshift"
                ],
                font=dict(
                    family=(
                        "Inter, Arial, sans-serif"
                    ),
                    color="#ffffff",
                    size=9.5,
                ),
                bgcolor=(
                    "rgba(16,47,69,0.92)"
                ),
                bordercolor=(
                    "rgba(255,255,255,0.18)"
                ),
                borderwidth=1,
                borderpad=2,
            )

    else:
        add_zero_state(
            fig,
            "No players meet the selected minutes threshold",
        )

    # ---------------------------------------------------------
    # CENTROID
    # ---------------------------------------------------------

    if (
        centroid_x is not None
        and centroid_y is not None
    ):
        fig.add_trace(
            go.Scatter(
                x=[
                    float(
                        centroid_x
                    )
                ],
                y=[
                    float(
                        centroid_y
                    )
                ],
                mode="markers",
                marker=dict(
                    size=16,
                    color=(
                        "rgba(255,255,255,0)"
                    ),
                    symbol="x",
                    line=dict(
                        color="#ffffff",
                        width=2,
                    ),
                ),
                hovertemplate=(
                    "<b>Team centre</b>"
                    "<br>Reference point for the outfield "
                    "territorial profile"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    apply_match_pitch_layout(
        fig,
        pitch_shapes=pitch_shapes,
        height=MEAN_POSITIONS_PLOT_HEIGHT,
        showlegend=False,
        header=False,
        x_range=(-2, 102),
        y_range=(-5, 107),
    )

    add_attacking_direction(
        fig,
        dark=True,
        x=0.985,
        y=0.025,
        xanchor="right",
        yanchor="bottom",
    )

    return fig
