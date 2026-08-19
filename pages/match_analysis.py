from dash import html
import dash_bootstrap_components as dbc

from src.components.brand_components import app_footer, brand_lockup


def layout(match_id):
    tabs_config = [
        {"label": "Match Overview", "value": "overview", "icon": "fas fa-clipboard-list"},
        {"label": "Formation", "value": "formation", "icon": "fas fa-users"},
        {"label": "Passes", "value": "passes", "icon": "fas fa-exchange-alt"},
        {"label": "Buildup", "value": "buildup", "icon": "fas fa-sitemap"},
        {"label": "Def. Transition", "value": "defensive-transition", "icon": "fas fa-shield-alt"},
        {"label": "Off. Transition", "value": "offensive-transition", "icon": "fas fa-bolt"},
        {"label": "Set Piece", "value": "set-piece", "icon": "fas fa-flag"},
        {"label": "Player Analysis", "value": "player_analysis", "icon": "fas fa-user-astronaut"},
    ]

    sidebar = html.Aside([
        brand_lockup("MATCH ANALYSIS", compact=True),
        html.Div(id="sidebar-match-header", className="match-analysis-fixture"),
        dbc.Nav(
            [
                dbc.NavLink(
                    [html.I(className=f"{tab['icon']}"), html.Span(tab["label"])],
                    href=f"/match/{match_id}?tab={tab['value']}",
                    active="exact",
                    id=f"navlink-{tab['value']}",
                    className="match-analysis-nav-link",
                )
                for tab in tabs_config
            ],
            vertical=True,
            pills=True,
            className="match-analysis-nav",
        ),
        html.Div([
            dbc.Button(
                [html.I(className="fas fa-file-alt me-2"), "Generate Report"],
                id="generate-report-button",
                className="match-analysis-report-btn",
            ),
            dbc.Button(
                [html.I(className="fas fa-home me-2"), "Back to Home"],
                href="/",
                className="match-analysis-home-btn",
            ),
        ], className="match-analysis-sidebar-actions"),
    ], className="match-analysis-sidebar")

    main_content = html.Div([
        html.Header([
            html.Div([
                html.Span("INTERACTIVE REPORT", className="match-analysis-topbar-kicker"),
                html.Strong("Match Intelligence Workspace"),
            ]),
            html.Div([
                html.I(className="fa-solid fa-circle-check"),
                html.Span("Event data loaded"),
            ], className="match-analysis-data-status"),
        ], className="match-analysis-topbar"),
        html.Div(id="match-tab-content", className="match-analysis-content"),
        app_footer("Event-level tactical analysis and interactive match reporting."),
    ], className="match-analysis-main")

    return html.Main([sidebar, main_content], className="match-analysis-page")
