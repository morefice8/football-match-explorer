# pages/home.py
from dash import html, dcc
import dash_bootstrap_components as dbc

def layout():
    """ The new main landing page with a centered brand identity and large action cards. """
    
    card_style = {
        "transition": "transform 0.2s ease-in-out",
        "cursor": "pointer",
        "height": "250px", # Altezza fissa per renderle quadrate
        "display": "flex",
        "flexDirection": "column",
        "justifyContent": "center",
        "alignItems": "center",
        "padding": "1rem"
    }
    
    icon_style = {"fontSize": "4rem", "marginBottom": "1.5rem", "color": "var(--bs-primary)"}

    def create_action_card(title, description, icon_class, href):
        return dbc.Col(
            dcc.Link(
                dbc.Card(
                    dbc.CardBody([
                        html.I(className=f"{icon_class}", style=icon_style),
                        html.H4(title, className="card-title"),
                        html.P(description, className="card-text text-muted"),
                    ]),
                    className="text-center shadow-sm h-100", # h-100 per farla riempire la colonna
                    style=card_style,
                ),
                href=href,
                className="text-decoration-none"
            ),
            lg=5, # Due card per riga, con un po' di spazio in mezzo
            md=6,
            sm=12,
            className="mb-4"
        )

    return dbc.Container([
        # --- Hero Banner senza sovrapposizione ---
html.Div(
    html.Img(
        src="/assets/brand/brand_banner.png",
        style={
            'width': '100vw',
            'height': '350px',
            'objectFit': 'cover'
        }
    ),
    style={'marginBottom': '2rem'}
),

# --- Titolo e Logo Centrati (fuori dal banner) ---
dbc.Row(
    dbc.Col(
        html.Div([
            html.Img(src="/assets/brand/app_logo.png", style={'height': '60px', 'marginRight': '15px'}),
            html.H1("Il Lab dell'8", className="d-inline-block align-middle text-white")
        ]),
        className="text-center"
    ),
    className="mb-5"
),

        # --- Card di Azione su due righe ---
        dbc.Row(
            [
                create_action_card("Analyze Uploaded Match", "Process a raw JSON event file.", "fas fa-upload", "/upload"),
                create_action_card("Explore Match Database", "Browse and analyze stored matches.", "fas fa-database", "/database"),
            ],
            justify="center",
            className="mb-4" # Spazio tra le due righe di card
        ),
        dbc.Row(
            [
                create_action_card("Team Statistics", "Compare team performance metrics.", "fas fa-chart-bar", "/team-stats"),
                create_action_card("Player Statistics", "Dive into player-specific data.", "fas fa-user-shield", "/player-stats"),
            ],
            justify="center",
        ),
        
    ], fluid=True, className="py-5")