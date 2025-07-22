from dash import html
import dash_bootstrap_components as dbc

def layout():
    """ Placeholder layout for the players statistics page. """
    return dbc.Container([
        dbc.Row(dbc.Col(dbc.Button([html.I(className="fas fa-arrow-left me-2"), "Back to Home"], href="/", color="secondary")), className="my-3"),
        html.H2("Players Statistics"),
        dbc.Alert("This feature is under construction. Players-level statistics will be available here soon!", color="info", className="mt-4")
    ], className="mt-5")
