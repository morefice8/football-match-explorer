import plotly.graph_objects as go

from src.visualization.coordinate_contract import ATTACKING_DIRECTION_LABEL


DARK_PLOT_BG = '#29343d'
DARK_TEXT = '#ffffff'
DARK_MUTED_TEXT = '#b9c8d2'
DARK_HOVER_BG = '#102f45'


def add_plot_header(
    fig,
    title,
    subtitle=None,
    *,
    dark=True,
):
    """
    Add a consistent title/subtitle block above a Plotly figure.
    """

    title_color = (
        DARK_TEXT
        if dark
        else '#17354d'
    )

    subtitle_color = (
        DARK_MUTED_TEXT
        if dark
        else '#647c8e'
    )

    fig.add_annotation(
        x=0.5,
        y=1.155,
        xref='paper',
        yref='paper',
        text=f"<b>{title}</b>",
        showarrow=False,
        xanchor='center',
        yanchor='middle',
        font=dict(
            size=18,
            color=title_color,
        ),
    )

    if subtitle:
        fig.add_annotation(
            x=0.5,
            y=1.105,
            xref='paper',
            yref='paper',
            text=subtitle,
            showarrow=False,
            xanchor='center',
            yanchor='middle',
            font=dict(
                size=11,
                color=subtitle_color,
            ),
        )

    return fig


def apply_dark_pitch_layout(
    fig,
    *,
    height=660,
    top_margin=125,
    showlegend=True,
):
    """
    Shared dark Match Analysis layout for pitch-based Plotly charts.
    """

    fig.update_layout(
        title=None,

        plot_bgcolor=DARK_PLOT_BG,
        paper_bgcolor=DARK_PLOT_BG,

        height=height,

        margin=dict(
            l=20,
            r=20,
            t=top_margin,
            b=20,
        ),

        showlegend=showlegend,

        hoverlabel=dict(
            bgcolor=DARK_HOVER_BG,
            font_color='white',
            bordercolor='rgba(255,255,255,0.18)',
        ),

        font=dict(
            color=DARK_TEXT,
        ),
    )

    return fig

def add_attacking_direction(
    fig,
    *,
    dark=True,
    x=0.985,
    y=0.985,
):
    """Add the canonical left-to-right attacking-direction label."""
    text_color = DARK_MUTED_TEXT if dark else '#536d80'
    bg_color = 'rgba(16,47,69,0.72)' if dark else 'rgba(255,255,255,0.88)'

    fig.add_annotation(
        x=x,
        y=y,
        xref='paper',
        yref='paper',
        text=f"<b>{ATTACKING_DIRECTION_LABEL}</b>",
        showarrow=False,
        xanchor='right',
        yanchor='top',
        font=dict(
            size=10,
            color=text_color,
        ),
        bgcolor=bg_color,
        borderpad=3,
    )
    return fig
