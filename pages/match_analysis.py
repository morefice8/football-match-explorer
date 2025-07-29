# pages/match_analysis.py
from dash import html, dcc
import dash_bootstrap_components as dbc
from src.components.layout_components import app_signature

def layout(match_id):
    sidebar_style = {
        "position": "fixed", "top": 0, "left": 0, "bottom": 0,
        "width": "18rem", "padding": "2rem 1rem", "backgroundColor": "#2E3439",
        "overflowY": "auto", "display": "flex", "flexDirection": "column"
    }
    content_style = {
        "marginLeft": "20rem", "marginRight": "1rem",
        "padding": "2rem 1rem", "maxWidth": "calc(100vw - 22rem)"
    }
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
    sidebar = html.Div([
            html.Div(id="sidebar-match-header", className="text-white text-center mb-3"),
            html.Hr(className="text-white"),
            dbc.Nav(
                [
                    dbc.NavLink(
                        [html.I(className=f"{tab['icon']} me-2"), tab["label"]],
                        href=f"/match/{match_id}?tab={tab['value']}", active="exact",
                        id=f"navlink-{tab['value']}", className="py-2"
                    ) for tab in tabs_config
                ],
                vertical=True, pills=True, className="mb-3 flex-grow-1"
            ),
            html.Img(src="/assets/brand/app_logo.png", style={"width": "120px", "margin": "30px auto 20px auto", "display": "block"}),
            dbc.Button([html.I(className="fas fa-file-alt me-2"), "Generate Report"], id="generate-report-button", color="success", className="mb-2 w-100"),
            dbc.Button([html.I(className="fas fa-home me-2"), "Back to Home"], href="/", color="secondary", className="w-100")
        ], style=sidebar_style)
        
    content_area = html.Div(id="match-tab-content")
    return html.Div([sidebar, content_area, app_signature()], style=content_style)