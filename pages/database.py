import os
from dash import html, dcc
import dash_bootstrap_components as dbc
from src.components.layout_components import app_signature

def get_leagues():
    path = os.path.join("data", "matches")
    if not os.path.exists(path): return []
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def layout():
    """ Layout for browsing the local match database. """
    return dbc.Container([
        dbc.Row(dbc.Col(dbc.Button([html.I(className="fas fa-arrow-left me-2"), "Back to Home"], href="/", color="secondary")), className="my-3"),
        html.H2("Explore Match Database"),
        html.P("Select a league, season, and team to find a specific match.", className="text-muted"),
        dbc.Row([
            dbc.Col([html.Label("League"), dcc.Dropdown(id="dropdown-league", options=[{"label": l, "value": l} for l in get_leagues()])], width=3),
            dbc.Col([html.Label("Season"), dcc.Dropdown(id="dropdown-season")], width=3),
            dbc.Col([html.Label("Team"), dcc.Dropdown(id="dropdown-team-filter")], width=3),
            dbc.Col([html.Label("Round"), dcc.Dropdown(id="dropdown-round")], width=3),
        ], className="mb-4"),
        html.H5("Matches found in database:", className="mt-4 mb-3 text-white"),
        dcc.Loading(type="circle", children=dbc.Row(id="match-list")),
        app_signature()
    ], fluid=True, className="py-4")