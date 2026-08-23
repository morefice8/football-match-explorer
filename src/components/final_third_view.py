from __future__ import annotations

import pandas as pd
import dash_bootstrap_components as dbc
from dash import dcc, html

from src.components.match_graph_shell import match_graph_panel
from src.visualization import pass_plotly


ENTRY_ALL = "all"
ENTRY_PASSES = "passes"
ENTRY_CARRIES = "carries"
SHOW_ENTRIES = "show"


def controls():
    return html.Section([
        html.Div([
            html.Div([
                html.Span(
                    "ENTRY TYPE",
                    className="match-panel-eyebrow",
                ),
                html.Strong(
                    "Final-third access",
                    className="pass-network-control-title",
                ),
                dbc.RadioItems(
                    id="final-third-entry-type",
                    options=[
                        {"label": "All", "value": ENTRY_ALL},
                        {"label": "Passes", "value": ENTRY_PASSES},
                        {"label": "Inferred carries", "value": ENTRY_CARRIES},
                    ],
                    value=ENTRY_ALL,
                    inline=True,
                    className="pass-network-radio final-third-entry-radio",
                    inputClassName="btn-check",
                    labelClassName="pass-network-radio-label",
                    labelCheckedClassName="is-active",
                ),
            ]),
        ], className="pass-network-control-group"),

        html.Div([
            html.Div([
                html.Span(
                    "DETAIL",
                    className="match-panel-eyebrow",
                ),
                html.Strong(
                    "Entry trajectories",
                    className="pass-network-control-title",
                ),
                html.Small(
                    (
                        "The summary is the default view. "
                        "Individual trajectories are optional."
                    ),
                    className="pass-network-control-note",
                ),
            ]),
            dbc.Checklist(
                id="final-third-show-entries",
                options=[
                    {"label": "Show entries", "value": SHOW_ENTRIES},
                ],
                value=[],
                switch=True,
                className="final-third-show-switch",
            ),
        ], className=(
            "pass-network-control-group "
            "final-third-detail-control"
        )),
    ], className=(
        "pass-network-controls "
        "final-third-controls"
    ))


def workspace(shell_component):
    return html.Div([
        controls(),
        shell_component,
    ], className=(
        "match-tab-body "
        "final-third-workspace"
    ))


def show_entries_enabled(value):
    return SHOW_ENTRIES in (value or [])


def filter_entries(entries_df, entry_type=ENTRY_ALL):
    data = (
        entries_df.copy()
        if entries_df is not None
        else pd.DataFrame()
    )

    if (
        data.empty
        or "entry_type" not in data.columns
        or entry_type == ENTRY_ALL
    ):
        return data

    if entry_type == ENTRY_PASSES:
        return data[data["entry_type"].eq("Pass")].copy()

    if entry_type == ENTRY_CARRIES:
        return data[data["entry_type"].eq("Carry")].copy()

    return data


def summarize_entries(entries_df, base_stats=None):
    data = (
        entries_df.copy()
        if entries_df is not None
        else pd.DataFrame()
    )
    base_stats = base_stats or {}

    type_counts = (
        data["entry_type"].value_counts()
        if not data.empty and "entry_type" in data.columns
        else pd.Series(dtype="int64")
    )
    channel_counts = (
        data["final_third_channel"].value_counts()
        if not data.empty and "final_third_channel" in data.columns
        else pd.Series(dtype="int64")
    )
    destination_counts = (
        data["destination_zone"].value_counts()
        if not data.empty and "destination_zone" in data.columns
        else pd.Series(dtype="int64")
    )

    return {
        "total_final_third": int(len(data)),
        "pass_entries": int(type_counts.get("Pass", 0)),
        "carry_entries": int(type_counts.get("Carry", 0)),
        "channel_left": int(channel_counts.get("Left", 0)),
        "channel_central": int(channel_counts.get("Central", 0)),
        "channel_right": int(channel_counts.get("Right", 0)),
        "zone14": int(destination_counts.get("Zone 14", 0)),
        "hs_left": int(destination_counts.get("Left Half-Space", 0)),
        "hs_right": int(destination_counts.get("Right Half-Space", 0)),
        "other": int(destination_counts.get("Other", 0)),
        "carry_entry_candidates": int(
            base_stats.get("carry_entry_candidates", 0)
        ),
        "carry_entries_excluded_confidence": int(
            base_stats.get("carry_entries_excluded_confidence", 0)
        ),
        "carry_entries_excluded_boundary": int(
            base_stats.get("carry_entries_excluded_boundary", 0)
        ),
        "carry_entries_excluded_total": int(
            base_stats.get("carry_entries_excluded_total", 0)
        ),
    }


def _metric_card(label, value, detail):
    return html.Div([
        html.Span(
            label,
            className="progressive-kpi-label",
        ),
        html.Strong(
            value,
            className="progressive-kpi-value",
        ),
        html.Small(
            detail,
            className="progressive-kpi-detail",
        ),
    ], className="progressive-kpi-card")


def _profile(counts, total):
    denominator = max(int(total), 1)

    return html.Div([
        html.Div([
            html.Div([
                html.Span(
                    label,
                    className="final-third-profile-label",
                ),
                html.Strong(
                    f"{count} · {count / denominator * 100:.0f}%",
                    className="final-third-profile-value",
                ),
            ], className="final-third-profile-meta"),

            html.Div(
                html.Span(
                    style={
                        "width":
                            f"{count / denominator * 100:.1f}%"
                    }
                ),
                className="progressive-channel-track",
            ),
        ], className="final-third-profile-row")
        for label, count in counts.items()
    ], className=(
        "progressive-channel-profile "
        "final-third-profile"
    ))


def _sidebar(summary):
    total = int(summary.get("total_final_third", 0))
    pass_count = int(summary.get("pass_entries", 0))
    carry_count = int(summary.get("carry_entries", 0))

    channel_counts = {
        "Left": int(summary.get("channel_left", 0)),
        "Central": int(summary.get("channel_central", 0)),
        "Right": int(summary.get("channel_right", 0)),
    }

    destination_counts = {
        "Zone 14": int(summary.get("zone14", 0)),
        "Left HS": int(summary.get("hs_left", 0)),
        "Right HS": int(summary.get("hs_right", 0)),
        "Other": int(summary.get("other", 0)),
    }

    if total:
        main_channel = max(channel_counts, key=channel_counts.get)
        main_channel_count = channel_counts[main_channel]
    else:
        main_channel = "—"
        main_channel_count = 0

    carry_candidates = int(
        summary.get("carry_entry_candidates", 0)
    )
    carries_excluded = int(
        summary.get("carry_entries_excluded_total", 0)
    )

    return html.Div([
        html.Div([
            _metric_card(
                "TOTAL ENTRIES",
                str(total),
                "Selected entry type",
            ),
            _metric_card(
                "PASS ENTRIES",
                str(pass_count),
                "Completed boundary crossings",
            ),
            _metric_card(
                "INFERRED CARRIES",
                str(carry_count),
                (
                    f"{carries_excluded} of "
                    f"{carry_candidates} candidates excluded"
                    if carry_candidates
                    else "No carry candidates"
                ),
            ),
            _metric_card(
                "MAIN CHANNEL",
                main_channel,
                f"{main_channel_count} of {total} selected entries",
            ),
        ], className="progressive-kpi-grid"),

        html.Div([
            html.Span(
                "CHANNEL PROFILE",
                className="match-panel-eyebrow",
            ),
            html.H6("Where the final third is entered"),
            _profile(channel_counts, total),
        ], className=(
            "progressive-sidebar-section "
            "final-third-sidebar-section"
        )),

        html.Div([
            html.Span(
                "DESTINATION PROFILE",
                className="match-panel-eyebrow",
            ),
            html.H6("Where the entry finishes"),
            _profile(destination_counts, total),
        ], className=(
            "progressive-sidebar-section "
            "final-third-sidebar-section"
        )),
    ], className=(
        "progressive-sidebar "
        "final-third-sidebar"
    ))


def _scope_label(entry_type):
    if entry_type == ENTRY_PASSES:
        return "pass entries"
    if entry_type == ENTRY_CARRIES:
        return "inferred carry entries"
    return "entries"


def team_panel(
    entries_df,
    base_stats,
    team_name,
    team_color,
    *,
    is_away=False,
    entry_type=ENTRY_ALL,
    show_entries=False,
):
    filtered = filter_entries(
        entries_df,
        entry_type,
    )
    summary = summarize_entries(
        filtered,
        base_stats,
    )

    if show_entries:
        figure = (
            pass_plotly
            .plot_final_third_entries_detail_plotly(
                filtered,
                summary,
                team_name,
                team_color,
                is_away=is_away,
            )
        )
    else:
        figure = pass_plotly.plot_final_third_summary_plotly(
            filtered,
            summary,
            team_name,
            team_color,
            is_away=is_away,
        )

    graph = dcc.Graph(
        figure=figure,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
        className=(
            "progressive-summary-map-graph "
            "final-third-map-graph"
        ),
    )

    total = int(summary.get("total_final_third", 0))
    scope = _scope_label(entry_type)

    empty_message = (
        f"No {scope} for this team."
        if total == 0
        else None
    )

    return match_graph_panel(
        team_name=team_name,
        is_away=is_away,
        sample_size=f"n = {total} {scope}",
        graph=graph,
        sidebar=_sidebar(summary),
        empty_message=empty_message,
    )


def panels(
    home_entries,
    home_stats,
    away_entries,
    away_stats,
    home_team,
    away_team,
    home_color,
    away_color,
    *,
    entry_type=ENTRY_ALL,
    show_entries=False,
):
    return [
        team_panel(
            home_entries,
            home_stats,
            home_team,
            home_color,
            is_away=False,
            entry_type=entry_type,
            show_entries=show_entries,
        ),
        team_panel(
            away_entries,
            away_stats,
            away_team,
            away_color,
            is_away=True,
            entry_type=entry_type,
            show_entries=show_entries,
        ),
    ]
