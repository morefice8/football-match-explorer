# src/metrics/cross_metrics.py
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html

def analyze_crosses(
    df_processed,
    team_name,
    *,
    post_cross_seconds=10.0,
    max_followup_events=8,
):
    # PLOT-08 outcome contract:
    # - Outcome keeps the raw Opta completion label.
    # - Retained = completed delivery OR same-team control/second ball
    #   inside the post-cross window before opponent control.
    # - Shot Generated = same-team shot inside that same window.
    # - Canonical is_key_pass/is_assist flags also imply a generated shot.
    # - The scan never crosses a period boundary.

    if (
        df_processed is None
        or df_processed.empty
    ):
        return pd.DataFrame()

    events = df_processed.copy()

    required_columns = {
        "team_name",
        "type_name",
        "cross",
    }

    if not required_columns.issubset(
        events.columns
    ):
        return pd.DataFrame()

    events["_cross_source_order"] = range(
        len(events)
    )

    sort_columns = []

    for column in (
        "periodId",
        "timeMin",
        "timeSec",
    ):
        if column in events.columns:
            events[column] = pd.to_numeric(
                events[column],
                errors="coerce",
            )
            sort_columns.append(
                column
            )

    sort_columns.append(
        "_cross_source_order"
    )

    events = (
        events
        .sort_values(
            sort_columns,
            kind="stable",
        )
        .reset_index(drop=True)
    )

    cross_numeric = pd.to_numeric(
        events["cross"],
        errors="coerce",
    ).fillna(0)

    cross_positions = events.index[
        (events["team_name"] == team_name)
        & (events["type_name"] == "Pass")
        & (cross_numeric == 1)
    ].tolist()

    if not cross_positions:
        return pd.DataFrame()

    shot_types = {
        "goal",
        "miss",
        "attempt saved",
        "post",
    }

    ignored_types = {
        "",
        "card",
        "error",
        "formation change",
        "formation set",
        "substitution off",
        "substitution on",
        "player off",
        "player on",
        "deleted event",
        "start",
        "end",
    }

    terminal_same_team_types = {
        "offside",
    }

    def truthy(value):
        if value is None:
            return False

        if isinstance(value, bool):
            return value

        try:
            if pd.isna(value):
                return False
        except Exception:
            pass

        if isinstance(value, (int, float)):
            return value == 1

        return (
            str(value)
            .strip()
            .casefold()
            in {
                "1",
                "true",
                "yes",
            }
        )

    def event_clock_seconds(row):
        minute = pd.to_numeric(
            row.get("timeMin"),
            errors="coerce",
        )

        second = pd.to_numeric(
            row.get("timeSec"),
            errors="coerce",
        )

        if pd.isna(minute):
            return None

        if pd.isna(second):
            second = 0.0

        return (
            float(minute) * 60.0
            + float(second)
        )

    def same_period(
        first,
        second,
    ):
        if "periodId" not in events.columns:
            return True

        first_period = first.get(
            "periodId"
        )

        second_period = second.get(
            "periodId"
        )

        if (
            pd.isna(first_period)
            or pd.isna(second_period)
        ):
            return True

        return (
            int(first_period)
            == int(second_period)
        )

    def post_cross_outcomes(
        cross_position,
        cross_row,
    ):
        retained = (
            str(
                cross_row.get(
                    "outcome",
                    "",
                )
            )
            == "Successful"
        )

        shot_generated = (
            truthy(
                cross_row.get(
                    "is_key_pass"
                )
            )
            or truthy(
                cross_row.get(
                    "is_assist"
                )
            )
        )

        start_clock = (
            event_clock_seconds(
                cross_row
            )
        )

        meaningful_events = 0

        for position in range(
            cross_position + 1,
            len(events),
        ):
            event = events.iloc[
                position
            ]

            if not same_period(
                cross_row,
                event,
            ):
                break

            event_clock = (
                event_clock_seconds(
                    event
                )
            )

            if (
                start_clock is not None
                and event_clock is not None
            ):
                delta = (
                    event_clock
                    - start_clock
                )

                if delta < 0:
                    continue

                if (
                    delta
                    > float(
                        post_cross_seconds
                    )
                ):
                    break

            event_type = (
                str(
                    event.get(
                        "type_name",
                        "",
                    )
                )
                .strip()
                .casefold()
            )

            if event_type in ignored_types:
                continue

            event_team = event.get(
                "team_name"
            )

            if (
                event_team is None
                or pd.isna(
                    event_team
                )
            ):
                continue

            if event_team != team_name:
                # Opponent foul leaves the attacking side with the
                # restart, but terminates the open-play post-cross window.
                if event_type == "foul":
                    retained = True
                break

            meaningful_events += 1

            if event_type in shot_types:
                retained = True
                shot_generated = True
                break

            if (
                event_type
                in terminal_same_team_types
            ):
                break

            retained = True

            if (
                meaningful_events
                >= int(
                    max_followup_events
                )
            ):
                break

        return (
            bool(retained),
            bool(shot_generated),
        )

    def get_pitch_zone(
        x,
        y,
    ):
        x = pd.to_numeric(
            x,
            errors="coerce",
        )

        y = pd.to_numeric(
            y,
            errors="coerce",
        )

        if (
            pd.isna(x)
            or pd.isna(y)
        ):
            return "Unknown"

        if y > 67:
            side = "Left"
        elif y < 33:
            side = "Right"
        else:
            side = "Center"

        if x > 80:
            area = "Deep"
        elif x > 60:
            area = "Advanced"
        else:
            area = "Midfield"

        return (
            f"{side} {area}"
        )

    analyzed_data = []

    for cross_position in cross_positions:
        cross = events.iloc[
            cross_position
        ]

        play_type = "Open Play"

        if truthy(
            cross.get(
                "Corner taken"
            )
        ):
            play_type = "From Corner"

        elif truthy(
            cross.get(
                "Freekick taken"
            )
        ):
            play_type = "From Free Kick"

        foot = "Unknown"

        if truthy(
            cross.get(
                "Right footed"
            )
        ):
            foot = "Right"

        elif truthy(
            cross.get(
                "Left footed"
            )
        ):
            foot = "Left"

        swing = "N/A"

        if truthy(
            cross.get(
                "In-swinger"
            )
        ):
            swing = "In-swinger"

        elif truthy(
            cross.get(
                "Out-swinger"
            )
        ):
            swing = "Out-swinger"

        elif truthy(
            cross.get(
                "Straight"
            )
        ):
            swing = "Straight"

        origin_zone = get_pitch_zone(
            cross.get("x"),
            cross.get("y"),
        )

        destination_zone = get_pitch_zone(
            cross.get("end_x"),
            cross.get("end_y"),
        )

        outcome = (
            "Completed"
            if (
                cross.get(
                    "outcome"
                )
                == "Successful"
            )
            else "Incomplete"
        )

        retained, shot_generated = (
            post_cross_outcomes(
                cross_position,
                cross,
            )
        )

        analyzed_data.append({
            "cross_id":
                cross.get(
                    "eventId",
                    cross.get(
                        "id",
                        cross_position,
                    ),
                ),
            "playerName":
                cross.get(
                    "playerName",
                    "Unknown",
                ),
            "Play Type":
                play_type,
            "Foot":
                foot,
            "Swing":
                swing,
            "Origin Zone":
                origin_zone,
            "Destination Zone":
                destination_zone,
            "Outcome":
                outcome,
            "Retained":
                retained,
            "Shot Generated":
                shot_generated,
            "x":
                cross.get("x"),
            "y":
                cross.get("y"),
            "end_x":
                cross.get("end_x"),
            "end_y":
                cross.get("end_y"),
            "periodId":
                cross.get(
                    "periodId"
                ),
            "timeMin":
                cross.get(
                    "timeMin"
                ),
            "timeSec":
                cross.get(
                    "timeSec"
                ),
        })

    return pd.DataFrame(
        analyzed_data
    )

def create_cross_summary_cards(df_analyzed, active_filter=None):
    """Creates a full set of detailed, interactive summary cards for crosses."""
    if df_analyzed.empty:
        return html.Div()

    # La funzione helper interna ora è corretta
    def create_card(title, data_dict, filter_type):
        items = []
        if not data_dict: return None
        sorted_items = sorted(data_dict.items(), key=lambda item: item[1], reverse=True)
        # Usiamo l'active_filter definito nello scope esterno
        for value, count in sorted_items:
            active = active_filter is not None and str(active_filter.get(filter_type)) == str(value)

            items.append(dbc.ListGroupItem(
                [html.Div(value), dbc.Badge(f"{count}", color="light", className="ms-auto")],
                id={'type': 'cross-filter', 'filter_type': filter_type, 'value': value},
                action=True, n_clicks=0, active=active,
                className="d-flex justify-content-between align-items-center"
            ))
        return dbc.Card(
            [
                dbc.CardHeader(
                    title,
                    className="cross-filter-card-header",
                ),

                dbc.ListGroup(
                    items,
                    flush=True,
                ),
            ],
            className="cross-filter-card",
        )

    stats = {
        'origin': df_analyzed['Origin Zone'].value_counts().to_dict(),
        'destination': df_analyzed['Destination Zone'].value_counts().to_dict(),
        'swing': df_analyzed[df_analyzed['Swing'] != 'N/A']['Swing'].value_counts().to_dict(),
        'outcome': df_analyzed['Outcome'].value_counts().to_dict(),
        'feet': df_analyzed[df_analyzed['Foot'] != 'Unknown']['Foot'].value_counts().to_dict(),
        'takers': df_analyzed['playerName'].value_counts().to_dict(),
        'play_type': df_analyzed['Play Type'].value_counts().to_dict()
    }

    active_filter = active_filter or {}

    # --- CHIAMATE CORRETTE (con 3 argomenti) ---
    cards = [
        create_card("Origin Zone", stats.get('origin'), 'origin'),
        create_card("Destination Zone", stats.get('destination'), 'destination'),
        create_card("Play Type", stats.get('play_type'), 'play_type'),
        create_card("Swing Type", stats.get('swing'), 'swing'),
        create_card("Taker Foot", stats.get('feet'), 'foot'),
        create_card("Outcome", stats.get('outcome'), 'outcome'),
    ]

    # Creiamo la card dei crossatori a parte per gestire il filter_type 'taker'
    takers_stats = stats.get('takers')
    if takers_stats:
        takers_items = []
        # Mostra solo i primi 5 per non affollare
        for player, count in sorted(takers_stats.items(), key=lambda item: item[1], reverse=True)[:5]:
            active = active_filter.get('taker') == player
            takers_items.append(dbc.ListGroupItem(
                [html.Div(player), dbc.Badge(f"{count}", color="light", className="ms-auto")],
                id={'type': 'cross-filter', 'filter_type': 'taker', 'value': player},
                action=True, n_clicks=0, active=active,
                className="d-flex justify-content-between align-items-center"
            ))
        cards.append(
            dbc.Card(
                [
                    dbc.CardHeader(
                        "Top Crossers",
                        className="cross-filter-card-header",
                    ),

                    dbc.ListGroup(
                        takers_items,
                        flush=True,
                    ),
                ],
                className="cross-filter-card",
            )
        )

    valid_cards = [
        card
        for card in cards
        if card is not None
    ]

    return html.Div(
        valid_cards,
        className="cross-filter-grid",
    )

def build_cross_flow_profile(
    df_analyzed,
    limit=8,
):
    # Route profile with event completion kept separately from
    # post-cross retention and shot generation.

    default_summary = {
        "total_crosses": 0,
        "retained_crosses": 0,
        "retention_pct": 0.0,
        "shot_crosses": 0,
        "shot_rate_pct": 0.0,
        "top_crosser": "N/A",
        "top_crosser_count": 0,
        "top_route": "N/A",
        "top_route_count": 0,
        "top_route_pct": 0.0,
        "top_three_pct": 0.0,
    }

    columns = [
        "Origin Zone",
        "Destination Zone",
        "Crosses",
        "Share %",
        "Completed",
        "Completion %",
        "Retained",
        "Retention %",
        "Shots",
        "Shot Rate %",
    ]

    if (
        df_analyzed is None
        or df_analyzed.empty
    ):
        return (
            default_summary,
            pd.DataFrame(
                columns=columns
            ),
        )

    df = df_analyzed.copy()

    for column in (
        "Retained",
        "Shot Generated",
    ):
        if column not in df.columns:
            df[column] = False

        df[column] = (
            df[column]
            .fillna(False)
            .astype(bool)
        )

    total_crosses = len(df)

    retained_crosses = int(
        df["Retained"].sum()
    )

    shot_crosses = int(
        df["Shot Generated"].sum()
    )

    top_crosser = "N/A"
    top_crosser_count = 0

    if (
        "playerName" in df.columns
        and df["playerName"]
            .notna()
            .any()
    ):
        crosser_counts = (
            df["playerName"]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
        )

        if not crosser_counts.empty:
            top_crosser = str(
                crosser_counts.index[0]
            )

            top_crosser_count = int(
                crosser_counts.iloc[0]
            )

    routes = (
        df.groupby(
            [
                "Origin Zone",
                "Destination Zone",
            ],
            dropna=False,
        )
        .agg(
            Crosses=(
                "Origin Zone",
                "size",
            ),
            Completed=(
                "Outcome",
                lambda values: int(
                    (
                        values
                        == "Completed"
                    ).sum()
                ),
            ),
            Retained=(
                "Retained",
                "sum",
            ),
            Shots=(
                "Shot Generated",
                "sum",
            ),
        )
        .reset_index()
    )

    for column in (
        "Crosses",
        "Completed",
        "Retained",
        "Shots",
    ):
        routes[column] = (
            pd.to_numeric(
                routes[column],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )

    routes["Share %"] = (
        routes["Crosses"]
        / total_crosses
        * 100.0
    )

    routes["Completion %"] = (
        routes["Completed"]
        / routes["Crosses"]
        * 100.0
    )

    routes["Retention %"] = (
        routes["Retained"]
        / routes["Crosses"]
        * 100.0
    )

    routes["Shot Rate %"] = (
        routes["Shots"]
        / routes["Crosses"]
        * 100.0
    )

    routes = (
        routes
        .sort_values(
            [
                "Crosses",
                "Shots",
                "Retained",
                "Completed",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    if routes.empty:
        return (
            default_summary,
            pd.DataFrame(
                columns=columns
            ),
        )

    top_route = routes.iloc[0]

    top_three_count = int(
        routes
        .head(3)["Crosses"]
        .sum()
    )

    summary = {
        "total_crosses":
            int(
                total_crosses
            ),
        "retained_crosses":
            retained_crosses,
        "retention_pct":
            (
                retained_crosses
                / total_crosses
                * 100.0
            ),
        "shot_crosses":
            shot_crosses,
        "shot_rate_pct":
            (
                shot_crosses
                / total_crosses
                * 100.0
            ),
        "top_crosser":
            top_crosser,
        "top_crosser_count":
            top_crosser_count,
        "top_route": (
            f"{top_route['Origin Zone']} → "
            f"{top_route['Destination Zone']}"
        ),
        "top_route_count":
            int(
                top_route[
                    "Crosses"
                ]
            ),
        "top_route_pct":
            float(
                top_route[
                    "Share %"
                ]
            ),
        "top_three_pct":
            (
                top_three_count
                / total_crosses
                * 100.0
            ),
    }

    return (
        summary,
        routes
        .head(
            limit
        )[columns]
        .copy(),
    )

