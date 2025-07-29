# pages/upload.py
from dash import html, dcc
import dash_bootstrap_components as dbc
from src.components.layout_components import app_signature

def layout():
    """ Layout for the page dedicated to uploading a single match file. """
    upload_style = {
        'width': '100%', 'height': '150px', 'lineHeight': '150px',
        'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '5px',
        'textAlign': 'center', 'margin': '20px 0', 'borderColor': '#6c757d',
        'color': '#6c757d', 'transition': 'all 0.3s ease-in-out'
    }
    return dbc.Container([
        dbc.Row(dbc.Col(dbc.Button([html.I(className="fas fa-arrow-left me-2"), "Back to Home"], href="/", color="secondary")), className="my-3"),
        html.H2("Analyze a New Match"),
        html.P("Upload a raw JSON match event file to start the analysis.", className="text-muted"),
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select a File')]),
            style=upload_style,
            multiple=False
        ),
        html.Div(id='upload-status-output', className="text-center mt-3"),
        app_signature(),
    ], className="mt-5")