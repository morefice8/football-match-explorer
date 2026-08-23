"""Analyst-facing Event Explorer for Match Analysis."""

from __future__ import annotations

import pandas as pd
import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html


ROW_KEY = "_event_source_row"

TECHNICAL_IDENTIFIER_COLUMNS = {
    "id", "eventId", "event_id", "matchId", "match_id",
    "fixtureId", "fixture_id", "teamId", "team_id",
    "playerId", "player_id", "receiver_player_id",
    "receiver_event_id", "relatedEventId", "related_event_id",
}

PERIOD_CANDIDATES = (
    "periodId",
    "period_id",
    "Period",
    "period_name",
    "period",
)

DEFAULT_COLUMN_LABELS = {
    "event_time": "Time",
    "event_period": "Period",
    "team_name": "Team",
    "playerName": "Player",
    "Mapped Jersey Number": "#",
    "positional_role": "Role",
    "type_name": "Event",
    "outcome": "Outcome",
    "start_location": "Start",
    "end_location": "End",
}

DEFAULT_COLUMN_ORDER = tuple(DEFAULT_COLUMN_LABELS)


def _number(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def format_event_timestamp(minute, second):
    minute_value = _number(minute)
    second_value = _number(second)

    if minute_value is None:
        return "—"

    minute_int = max(0, int(minute_value))
    second_int = (
        0
        if second_value is None
        else max(0, min(59, int(second_value)))
    )
    return f"{minute_int:02d}:{second_int:02d}"


def _normalise_period(value, minute=None):
    if value is not None:
        try:
            if not pd.isna(value):
                numeric = int(float(value))
                mapping = {
                    1: "1H",
                    2: "2H",
                    3: "ET1",
                    4: "ET2",
                    5: "Pens",
                }
                if numeric in mapping:
                    return mapping[numeric]

                # Unknown numeric feed codes are not canonical football periods.
                # Fall back to match minute instead of exposing values such as
                # 14 or 16 in the Period filter.
                value = None
        except (TypeError, ValueError):
            pass

        text = "" if value is None else str(value).strip()
        aliases = {
            "first half": "1H",
            "1st half": "1H",
            "first": "1H",
            "second half": "2H",
            "2nd half": "2H",
            "second": "2H",
            "extra time first half": "ET1",
            "extra time second half": "ET2",
            "penalties": "Pens",
            "penalty shootout": "Pens",
        }
        if text.lower() in aliases:
            return aliases[text.lower()]
        if text:
            return text

    minute_value = _number(minute)
    if minute_value is None:
        return "—"
    if minute_value <= 45:
        return "1H"
    if minute_value <= 90:
        return "2H"
    if minute_value <= 105:
        return "ET1"
    if minute_value <= 120:
        return "ET2"
    return "Pens"


def resolve_event_periods(df):
    period_column = next(
        (
            column
            for column in PERIOD_CANDIDATES
            if column in df.columns
        ),
        None,
    )

    minutes = (
        df["timeMin"]
        if "timeMin" in df.columns
        else pd.Series(
            [None] * len(df),
            index=df.index,
        )
    )

    if period_column is None:
        values = [
            _normalise_period(None, minute)
            for minute in minutes
        ]
    else:
        values = [
            _normalise_period(period, minute)
            for period, minute
            in zip(df[period_column], minutes)
        ]

    return pd.Series(
        values,
        index=df.index,
        dtype="object",
    )


def format_location(x, y):
    x_value = _number(x)
    y_value = _number(y)
    if x_value is None or y_value is None:
        return "—"
    return f"{x_value:.1f}, {y_value:.1f}"


def prepare_event_explorer_dataframe(df):
    """Build the compact analyst view without mutating the raw event feed."""
    if df is None:
        return pd.DataFrame(
            columns=list(DEFAULT_COLUMN_ORDER)
        )

    source = df.copy(deep=True)
    display = pd.DataFrame(index=source.index)
    display[ROW_KEY] = range(len(source))

    minutes = source.get(
        "timeMin",
        pd.Series(
            [None] * len(source),
            index=source.index,
        ),
    )
    seconds = source.get(
        "timeSec",
        pd.Series(
            [None] * len(source),
            index=source.index,
        ),
    )

    display["event_time"] = [
        format_event_timestamp(minute, second)
        for minute, second
        in zip(minutes, seconds)
    ]
    display["event_period"] = resolve_event_periods(source)

    for column in (
        "team_name",
        "playerName",
        "Mapped Jersey Number",
        "positional_role",
        "type_name",
        "outcome",
    ):
        if column in source.columns:
            display[column] = source[column]

    if "x" in source.columns and "y" in source.columns:
        display["start_location"] = [
            format_location(x, y)
            for x, y in zip(source["x"], source["y"])
        ]

    if "end_x" in source.columns and "end_y" in source.columns:
        display["end_location"] = [
            format_location(x, y)
            for x, y in zip(source["end_x"], source["end_y"])
        ]

    # Dash DataTable understands a record-level id as a stable row id.
    # It is not included in the visible columns.
    display["id"] = display[ROW_KEY].astype(str)
    return display


def event_explorer_visible_columns(display_df):
    return [
        column
        for column in DEFAULT_COLUMN_ORDER
        if column in display_df.columns
    ]


def filter_event_explorer_dataframe(
    display_df,
    *,
    team=None,
    event=None,
    period=None,
):
    filtered = display_df.copy()

    for column, value in (
        ("team_name", team),
        ("type_name", event),
        ("event_period", period),
    ):
        if (
            value not in (None, "")
            and column in filtered.columns
        ):
            filtered = filtered[
                filtered[column].astype(str)
                == str(value)
            ]

    return filtered


def _dropdown_options(series):
    values = []

    for value in series:
        if pd.isna(value):
            continue

        text = str(value).strip()
        if text:
            values.append(text)

    return [
        {"label": value, "value": value}
        for value in sorted(set(values))
    ]


def event_filter_options(display_df):
    return {
        "teams": _dropdown_options(
            display_df.get(
                "team_name",
                pd.Series(dtype="object"),
            )
        ),
        "events": _dropdown_options(
            display_df.get(
                "type_name",
                pd.Series(dtype="object"),
            )
        ),
        "periods": _dropdown_options(
            display_df.get(
                "event_period",
                pd.Series(dtype="object"),
            )
        ),
    }


def event_explorer_records(display_df):
    safe_df = display_df.copy()
    safe_df = safe_df.where(pd.notna(safe_df), None)
    return safe_df.to_dict("records")


def render_technical_fields(raw_row):
    items = []

    for field, value in raw_row.items():
        if pd.isna(value):
            display_value = "—"
        elif isinstance(value, float):
            display_value = f"{value:.6g}"
        else:
            display_value = str(value)

        classes = "event-technical-row"
        if field in TECHNICAL_IDENTIFIER_COLUMNS:
            classes += " event-technical-row--identifier"

        items.append(
            html.Div(
                [
                    html.Span(
                        str(field),
                        className="event-technical-key",
                    ),
                    html.Code(
                        display_value,
                        className="event-technical-value",
                    ),
                ],
                className=classes,
            )
        )

    return html.Div(
        items,
        className="event-technical-list",
    )


def build_event_explorer(df):
    display_df = prepare_event_explorer_dataframe(df)
    visible_columns = event_explorer_visible_columns(display_df)
    options = event_filter_options(display_df)

    columns = [
        {
            "name": DEFAULT_COLUMN_LABELS[column],
            "id": column,
        }
        for column in visible_columns
    ]
    table = dash_table.DataTable(
        id="overview-datatable",
        data=event_explorer_records(display_df),
        columns=columns,
        page_size=20,
        filter_action="none",
        sort_action="native",
        sort_mode="multi",
        page_action="native",
        fixed_rows={"headers": True},
        cell_selectable=True,
        style_table={
            "overflowX": "auto",
            "maxWidth": "100%",
            "maxHeight": "640px",
        },
        style_cell={
            "backgroundColor": "#ffffff",
            "color": "#233b53",
            "textAlign": "left",
            "minWidth": "72px",
            "maxWidth": "210px",
            "whiteSpace": "normal",
            "border": "0",
            "borderBottom": "1px solid #e7eef3",
            "fontFamily": "Inter, Arial, sans-serif",
            "fontSize": "12px",
            "padding": "11px 10px",
        },
        style_cell_conditional=[
            {
                "if": {"column_id": "event_time"},
                "width": "72px",
                "textAlign": "center",
                "fontWeight": "800",
            },
            {
                "if": {"column_id": "event_period"},
                "width": "60px",
                "textAlign": "center",
            },
            {
                "if": {"column_id": "Mapped Jersey Number"},
                "width": "44px",
                "textAlign": "center",
            },
            {
                "if": {"column_id": "playerName"},
                "minWidth": "145px",
            },
            {
                "if": {"column_id": "type_name"},
                "minWidth": "125px",
            },
            {
                "if": {"column_id": "outcome"},
                "width": "118px",
                "textAlign": "center",
            },
            {
                "if": {"column_id": "start_location"},
                "width": "92px",
                "textAlign": "center",
            },
            {
                "if": {"column_id": "end_location"},
                "width": "92px",
                "textAlign": "center",
            },
        ],
        style_header={
            "backgroundColor": "#f5f9fb",
            "color": "#526b7d",
            "fontWeight": "800",
            "border": "0",
            "borderBottom": "1px solid #dbe7ed",
            "padding": "12px 10px",
            "fontSize": "10px",
            "textTransform": "uppercase",
            "letterSpacing": ".06em",
        },
        style_data_conditional=[
            {
                "if": {
                    "filter_query": '{outcome} = "Successful"',
                    "column_id": "outcome",
                },
                "backgroundColor": "#e8f6f0",
                "color": "#177258",
                "fontWeight": "800",
            },
            {
                "if": {
                    "filter_query": '{outcome} = "Unsuccessful"',
                    "column_id": "outcome",
                },
                "backgroundColor": "#fff0ed",
                "color": "#b94d42",
                "fontWeight": "800",
            },
            {
                "if": {
                    "filter_query": '{outcome} = "Unknown"',
                    "column_id": "outcome",
                },
                "backgroundColor": "#eef3f6",
                "color": "#667d8d",
                "fontWeight": "800",
            },
            {
                "if": {"state": "active"},
                "backgroundColor": "#edf8fb",
                "border": "1px solid #7bc9db",
            },
        ],
    )

    def filter_field(label, component):
        return html.Div(
            [
                html.Label(
                    label,
                    className="event-filter-label",
                ),
                component,
            ],
            className="event-filter-field",
        )

    return html.Div(
        [
            html.Div(
                [
                    filter_field(
                        "Team",
                        dcc.Dropdown(
                            id="event-explorer-team-filter",
                            options=options["teams"],
                            placeholder="All teams",
                            clearable=True,
                            className="event-filter-dropdown",
                        ),
                    ),
                    filter_field(
                        "Event",
                        dcc.Dropdown(
                            id="event-explorer-event-filter",
                            options=options["events"],
                            placeholder="All events",
                            clearable=True,
                            searchable=True,
                            className="event-filter-dropdown",
                        ),
                    ),
                    filter_field(
                        "Period",
                        dcc.Dropdown(
                            id="event-explorer-period-filter",
                            options=options["periods"],
                            placeholder="All periods",
                            clearable=True,
                            className="event-filter-dropdown",
                        ),
                    ),
                    html.Div(
                        [
                            html.Span(
                                "VISIBLE EVENTS",
                                className="event-filter-count-label",
                            ),
                            html.Strong(
                                f"{len(display_df):,}",
                                id="event-explorer-count",
                                className="event-filter-count",
                            ),
                        ],
                        className="event-filter-summary",
                    ),
                ],
                className="event-explorer-filters",
            ),
            html.Div(
                table,
                className="match-event-table",
            ),
            html.Div(
                [
                    html.I(
                        className="fa-solid fa-arrow-pointer"
                    ),
                    html.Span(
                        "Select any row to inspect "
                        "the complete technical event."
                    ),
                ],
                className="event-explorer-row-hint",
            ),
            dbc.Offcanvas(
                html.Div(
                    id="event-explorer-drawer-body"
                ),
                id="event-explorer-drawer",
                title="Event technical fields",
                is_open=False,
                placement="end",
                scrollable=True,
                backdrop=True,
                className="event-explorer-drawer",
            ),
        ],
        className="event-explorer",
    )
