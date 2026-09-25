# src/visualization/pitch_plots.py
"""Shared Plotly pitch-shape helper.

Historically this module held the app's matplotlib pitch/chart
functions. All of them have since been superseded by Plotly
equivalents elsewhere (shot_map_plotly.py, formation_plotly.py, etc.)
and were removed; get_plotly_pitch_shapes is the one piece those
Plotly modules still import from here.
"""


def get_plotly_pitch_shapes(line_color="rgba(0, 0, 0, 0.5)", background_color=None):
    """
    Restituisce una lista di 'shapes' di Plotly per disegnare un campo da calcio.
    Questa funzione NON modifica una figura, ma restituisce solo gli oggetti.
    """
    del background_color

    PITCH_WIDTH_OPTA = 100
    PITCH_HEIGHT_OPTA = 100

    shapes = [
        dict(type="rect", x0=0, y0=0, x1=PITCH_WIDTH_OPTA, y1=PITCH_HEIGHT_OPTA, line=dict(color=line_color, width=2), layer="above"),
        dict(type="line", x0=50, y0=0, x1=50, y1=PITCH_HEIGHT_OPTA, line=dict(color=line_color, width=2), layer="above"),
        dict(type="circle", x0=41.5, y0=41.5, x1=58.5, y1=58.5, line=dict(color=line_color, width=2), layer="above"),
        dict(type="rect", x0=0, y0=21.1, x1=16.5, y1=78.9, line=dict(color=line_color, width=2), layer="above"),
        dict(type="rect", x0=100, y0=21.1, x1=83.5, y1=78.9, line=dict(color=line_color, width=2), layer="above"),
        dict(type="rect", x0=0, y0=36.8, x1=5.5, y1=63.2, line=dict(color=line_color, width=2), layer="above"),
        dict(type="rect", x0=100, y0=36.8, x1=94.5, y1=63.2, line=dict(color=line_color, width=2), layer="above"),
        dict(type="path", path="M 16.5,34.9 C 22.5,42 22.5,58 16.5,65.1", line_color=line_color, layer="above"),
        dict(type="path", path="M 83.5,34.9 C 77.5,42 77.5,58 83.5,65.1", line_color=line_color, layer="above"),
    ]
    return shapes
