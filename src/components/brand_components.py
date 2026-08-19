from dash import dcc, html


def brand_lockup(context="FOOTBALL INTELLIGENCE", compact=False):
    """Reusable brand mark and wordmark."""
    return html.Div([
        html.Div(
            html.Img(src="/assets/brand/app_logo.svg", alt="Il Lab dell'8"),
            className="app-brand-mark",
        ),
        html.Div([
            html.Strong("Il Lab dell’8"),
            html.Span(context),
        ], className="app-brand-copy"),
    ], className=f"app-brand-lockup {'compact' if compact else ''}")


def stats_page_hero(
    title,
    subtitle,
    eyebrow,
    icon,
    variant,
    back_href="/",
    back_label="Home",
):
    """Compact branded hero for season-level statistics workspaces."""
    return html.Header([
        html.Div([
            html.Nav([
                dcc.Link([
                    html.I(className="fa-solid fa-arrow-left-long"),
                    html.Span(back_label),
                ], href=back_href, className="app-page-back-link"),
                brand_lockup(f"{eyebrow} WORKSPACE", compact=True),
            ], className="app-page-hero-nav"),
            html.Div([
                html.Div([
                    html.Span(eyebrow, className="app-page-eyebrow"),
                    html.H1(title),
                    html.P(subtitle),
                ], className="app-page-hero-copy"),
                html.Div([
                    html.I(className=icon),
                    html.Div([
                        html.Strong("MULTI-SEASON"),
                        html.Span("Top-five league intelligence"),
                    ]),
                ], className="app-page-hero-badge"),
            ], className="app-page-hero-content"),
        ], className="app-page-shell"),
    ], className=f"app-page-hero variant-{variant}")


def app_footer(context="Football analytics and match intelligence"):
    """Shared, non-fixed footer for analytical pages."""
    return html.Footer([
        html.Div([
            html.Div([
                brand_lockup("FOOTBALL INTELLIGENCE", compact=True),
                html.P(context),
            ], className="app-footer-identity"),
            html.Nav([
                dcc.Link("Upload Match", href="/upload"),
                dcc.Link("Team Statistics", href="/team-stats"),
                dcc.Link("Player Statistics", href="/player-stats"),
                html.A(
                    ["About", html.I(className="fa-solid fa-arrow-up-right-from-square")],
                    href="https://www.micheleorefice.com",
                    target="_blank",
                ),
            ], className="app-footer-nav"),
        ], className="app-page-shell app-footer-inner"),
    ], className="app-footer")
