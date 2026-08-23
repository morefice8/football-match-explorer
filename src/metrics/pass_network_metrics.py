"""Reliable, period-aware Pass Network metrics (PLOT-04)."""

from __future__ import annotations

import pandas as pd


PASS_NETWORK_MIN_MINUTES = 15.0
PASS_NETWORK_MIN_CONNECTION = 5
PASS_NETWORK_TOP_CONNECTIONS = 8


def _period_mask(df: pd.DataFrame, period: str) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=bool)

    key = str(period or "full").strip().lower()
    if key in {"full", "full_match", "match"}:
        return pd.Series(True, index=df.index)

    if "periodId" not in df.columns:
        return pd.Series(False, index=df.index)

    period_ids = pd.to_numeric(df["periodId"], errors="coerce")
    if key in {"1h", "1", "first_half"}:
        return period_ids.eq(1)
    if key in {"2h", "2", "second_half"}:
        return period_ids.eq(2)
    raise ValueError("period must be 'full', '1h' or '2h'")


def _seconds(df: pd.DataFrame) -> pd.Series:
    minutes = pd.to_numeric(df.get("timeMin"), errors="coerce").fillna(0.0)
    seconds = pd.to_numeric(df.get("timeSec"), errors="coerce").fillna(0.0)
    return minutes * 60.0 + seconds


def _truthy(value) -> bool:
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
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _window(df_team: pd.DataFrame, period: str) -> tuple[float, float]:
    if df_team is None or df_team.empty:
        return 0.0, 0.0

    secs = _seconds(df_team)
    key = str(period or "full").strip().lower()
    if key in {"full", "full_match", "match"}:
        return 0.0, max(float(secs.max()), 90.0 * 60.0)

    selected = secs[_period_mask(df_team, key)]
    if selected.empty:
        return 0.0, 0.0

    # Real event feeds contain events throughout each half. Small buffers
    # avoid shaving minutes from players whose first/last recorded action is
    # not exactly the half boundary.
    if key in {"1h", "1", "first_half"}:
        return min(float(selected.min()), 0.0), max(float(selected.max()), 45.0 * 60.0)
    return min(float(selected.min()), 45.0 * 60.0), max(float(selected.max()), 90.0 * 60.0)


def player_minutes_in_window(
    df_processed: pd.DataFrame,
    team_name: str,
    period: str,
) -> pd.DataFrame:
    """Best-effort lineup/substitution minutes for Full/1H/2H."""
    columns = ["playerName", "minutes", "is_starter", "status"]
    if (
        df_processed is None
        or df_processed.empty
        or "team_name" not in df_processed.columns
        or "playerName" not in df_processed.columns
    ):
        return pd.DataFrame(columns=columns)

    team = df_processed[df_processed["team_name"].eq(team_name)].copy()
    if team.empty:
        return pd.DataFrame(columns=columns)

    team["_pn_seconds"] = _seconds(team)
    start, end = _window(team, period)
    if end <= start:
        return pd.DataFrame(columns=columns)

    type_ids = pd.to_numeric(
        team.get("typeId", pd.Series(index=team.index, dtype=float)),
        errors="coerce",
    )

    starter_column = None
    for candidate in ("Is Starter", "is_starter", "isStarter"):
        if candidate in team.columns:
            starter_column = candidate
            break

    records = []
    for player_name in team["playerName"].dropna().astype(str).unique():
        rows = team[team["playerName"].astype(str).eq(player_name)].copy()

        is_starter = False
        if starter_column:
            is_starter = any(_truthy(value) for value in rows[starter_column])

        entry_candidates = []
        if is_starter:
            entry_candidates.append(0.0)

        sub_on = rows[type_ids.loc[rows.index].eq(19)]["_pn_seconds"].dropna()
        if not sub_on.empty:
            entry_candidates.append(float(sub_on.min()))

        # Fallback for processed feeds where the substitute-on administrative
        # event is not attached to the incoming player's own row.
        if not entry_candidates:
            observed = rows["_pn_seconds"].dropna()
            if not observed.empty:
                entry_candidates.append(float(observed.min()))

        if not entry_candidates:
            continue

        exit_candidates = []
        player_off = rows[type_ids.loc[rows.index].eq(18)]["_pn_seconds"].dropna()
        if not player_off.empty:
            exit_candidates.append(float(player_off.min()))

        # Only explicit red/second-yellow fields terminate the interval.
        for card_col in (
            "Red card", "Red Card", "Second yellow", "Second Yellow",
            "Second yellow card", "card_type", "cardType",
        ):
            if card_col not in rows.columns:
                continue
            values = rows[card_col]
            mask = values.apply(
                lambda value: _truthy(value)
                or str(value).strip().lower() in {
                    "red", "rc", "second yellow", "2nd yellow"
                }
            )
            dismissed = rows.loc[mask, "_pn_seconds"].dropna()
            if not dismissed.empty:
                exit_candidates.append(float(dismissed.min()))

        player_on = min(entry_candidates)
        player_off_time = min(exit_candidates) if exit_candidates else end
        overlap = max(min(player_off_time, end) - max(player_on, start), 0.0)

        records.append({
            "playerName": player_name,
            "minutes": overlap / 60.0,
            "is_starter": bool(is_starter),
            "status": "Starter" if is_starter else "Substitute",
        })

    return pd.DataFrame(records, columns=columns)


def build_pass_network_profile(
    passes_df: pd.DataFrame,
    df_processed: pd.DataFrame,
    team_name: str,
    *,
    period: str = "full",
    min_minutes: float = PASS_NETWORK_MIN_MINUTES,
    min_connection: int = PASS_NETWORK_MIN_CONNECTION,
    top_n: int = PASS_NETWORK_TOP_CONNECTIONS,
):
    """
    Build one compact network.

    Only successful passes with reliable receiver inference can create an
    edge. Node location is median pass involvement location (pass origins +
    reception endpoints), node volume is sent + received passes, both edge
    directions are retained, and only top-N threshold-qualified edges leave
    this metric layer.
    """
    edge_columns = [
        "player1", "player2", "player1_to_player2",
        "player2_to_player1", "pass_count", "pass_avg_x",
        "pass_avg_y", "pass_avg_x_end", "pass_avg_y_end",
    ]
    node_columns = [
        "playerName", "pass_avg_x", "pass_avg_y", "pass_sent",
        "pass_received", "pass_involvement", "jersey_number",
        "minutes", "is_starter", "status",
    ]
    summary = {
        "period": str(period or "full").lower(),
        "min_minutes": float(min_minutes),
        "min_connection": int(min_connection),
        "top_n": int(top_n),
        "successful_passes": 0,
        "reliable_passes": 0,
        "included_passes": 0,
        "eligible_players": 0,
        "qualifying_connections": 0,
        "shown_connections": 0,
    }

    empty_edges = pd.DataFrame(columns=edge_columns)
    empty_nodes = pd.DataFrame(columns=node_columns)
    if passes_df is None or passes_df.empty:
        return empty_edges, empty_nodes, summary

    required = {
        "team_name", "playerName", "receiver", "outcome",
        "x", "y", "end_x", "end_y", "receiver_is_reliable",
    }
    if not required.issubset(passes_df.columns):
        return empty_edges, empty_nodes, summary

    passes = passes_df[passes_df["team_name"].eq(team_name)].copy()
    passes = passes[_period_mask(passes, period)].copy()
    passes = passes[passes["outcome"].eq("Successful")].copy()
    summary["successful_passes"] = int(len(passes))

    passes = passes[
        passes["receiver_is_reliable"].fillna(False).astype(bool)
    ].copy()
    passes = passes.dropna(subset=["playerName", "receiver"]).copy()
    passes["playerName"] = passes["playerName"].astype(str)
    passes["receiver"] = passes["receiver"].astype(str)
    passes = passes[passes["playerName"].ne(passes["receiver"])].copy()
    summary["reliable_passes"] = int(len(passes))

    minutes = player_minutes_in_window(df_processed, team_name, period)
    eligible_minutes = minutes[minutes["minutes"].ge(float(min_minutes))].copy()
    eligible = set(eligible_minutes["playerName"].astype(str))

    passes = passes[
        passes["playerName"].isin(eligible)
        & passes["receiver"].isin(eligible)
    ].copy()

    for col in ("x", "y", "end_x", "end_y"):
        passes[col] = pd.to_numeric(passes[col], errors="coerce")
    passes = passes.dropna(subset=["x", "y", "end_x", "end_y"]).copy()
    summary["included_passes"] = int(len(passes))
    if passes.empty:
        return empty_edges, empty_nodes, summary

    points = pd.concat([
        pd.DataFrame({
            "playerName": passes["playerName"],
            "x": passes["x"],
            "y": passes["y"],
        }),
        pd.DataFrame({
            "playerName": passes["receiver"],
            "x": passes["end_x"],
            "y": passes["end_y"],
        }),
    ], ignore_index=True)

    nodes = points.groupby("playerName", as_index=False).agg(
        pass_avg_x=("x", "median"),
        pass_avg_y=("y", "median"),
    )
    sent = passes.groupby("playerName").size().rename("pass_sent")
    received = passes.groupby("receiver").size().rename("pass_received")
    nodes = nodes.merge(sent, left_on="playerName", right_index=True, how="left")
    nodes = nodes.merge(received, left_on="playerName", right_index=True, how="left")
    nodes[["pass_sent", "pass_received"]] = (
        nodes[["pass_sent", "pass_received"]].fillna(0).astype(int)
    )
    nodes["pass_involvement"] = nodes["pass_sent"] + nodes["pass_received"]

    jersey_lookup = {}
    if (
        df_processed is not None
        and not df_processed.empty
        and "Mapped Jersey Number" in df_processed.columns
    ):
        jersey_lookup = (
            df_processed[df_processed["team_name"].eq(team_name)]
            .dropna(subset=["playerName"])
            .drop_duplicates("playerName")
            .set_index("playerName")["Mapped Jersey Number"]
            .to_dict()
        )
    nodes["jersey_number"] = nodes["playerName"].map(jersey_lookup)
    nodes = nodes.merge(eligible_minutes, on="playerName", how="left")
    nodes = nodes[node_columns].sort_values(
        ["pass_involvement", "playerName"], ascending=[False, True]
    ).reset_index(drop=True)
    summary["eligible_players"] = int(len(nodes))

    directional = (
        passes.groupby(["playerName", "receiver"]).size()
        .rename("count").reset_index()
    )
    pairs = {}
    for row in directional.itertuples(index=False):
        p1, p2 = sorted((str(row.playerName), str(row.receiver)))
        key = (p1, p2)
        record = pairs.setdefault(key, {
            "player1": p1,
            "player2": p2,
            "player1_to_player2": 0,
            "player2_to_player1": 0,
        })
        if str(row.playerName) == p1:
            record["player1_to_player2"] += int(row.count)
        else:
            record["player2_to_player1"] += int(row.count)

    edges = pd.DataFrame(list(pairs.values()))
    if edges.empty:
        return empty_edges, nodes, summary

    edges["pass_count"] = (
        edges["player1_to_player2"] + edges["player2_to_player1"]
    )
    edges = edges[edges["pass_count"].ge(int(min_connection))].copy()
    summary["qualifying_connections"] = int(len(edges))
    if edges.empty:
        return empty_edges, nodes, summary

    node_index = nodes.set_index("playerName")
    edges["pass_avg_x"] = edges["player1"].map(node_index["pass_avg_x"])
    edges["pass_avg_y"] = edges["player1"].map(node_index["pass_avg_y"])
    edges["pass_avg_x_end"] = edges["player2"].map(node_index["pass_avg_x"])
    edges["pass_avg_y_end"] = edges["player2"].map(node_index["pass_avg_y"])
    edges = (
        edges.dropna(subset=[
            "pass_avg_x", "pass_avg_y", "pass_avg_x_end", "pass_avg_y_end"
        ])
        .sort_values(["pass_count", "player1", "player2"], ascending=[False, True, True])
        .head(max(int(top_n), 0))
        .reset_index(drop=True)
    )
    summary["shown_connections"] = int(len(edges))
    return edges[edge_columns], nodes, summary
