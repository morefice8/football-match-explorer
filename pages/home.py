from dash import dcc, html
import dash_bootstrap_components as dbc


def _workspace_card(title, description, eyebrow, icon, href, accent, featured=False):
    return dbc.Col(
        dcc.Link(
            html.Article([
                html.Div([
                    html.Span(eyebrow, className="landing-card-eyebrow"),
                    html.Div(html.I(className=icon), className="landing-card-icon"),
                ], className="landing-card-topline"),
                html.H3(title),
                html.P(description),
                html.Div([
                    html.Span("Open workspace"),
                    html.I(className="fa-solid fa-arrow-right-long"),
                ], className="landing-card-action"),
            ], className=f"landing-workspace-card accent-{accent} {'featured' if featured else ''}"),
            href=href,
            className="landing-card-link",
        ),
        lg=4,
        md=6,
        sm=12,
        className="mb-4",
    )


def _workflow_step(number, title, description, icon):
    return html.Div([
        html.Div([
            html.Span(number),
            html.I(className=icon),
        ], className="landing-step-marker"),
        html.Div([
            html.H4(title),
            html.P(description),
        ]),
    ], className="landing-workflow-step")


def layout():
    return html.Main([
        html.Section([
            html.Div(className="landing-hero-image"),
            html.Div(className="landing-hero-shade"),
            html.Div([
                html.Nav([
                    html.Div([
                        html.Div(
                            html.Img(src="/assets/brand/app_logo.svg", alt="Il Lab dell'8"),
                            className="landing-brand-mark",
                        ),
                        html.Div([
                            html.Strong("Il Lab dell’8"),
                            html.Span("FOOTBALL INTELLIGENCE"),
                        ], className="landing-brand-copy"),
                    ], className="landing-brand"),
                    html.A(
                        [html.I(className="fa-solid fa-arrow-up-right-from-square me-2"), "About the project"],
                        href="https://www.micheleorefice.com",
                        target="_blank",
                        className="landing-about-link",
                    ),
                ], className="landing-nav"),

                html.Div([
                    html.Div([
                        html.Div([
                            html.Span(className="landing-live-dot"),
                            "MATCH & PERFORMANCE ANALYTICS",
                        ], className="landing-eyebrow"),
                        html.H1([
                            "See the game ",
                            html.Span("through data."),
                        ]),
                        html.P(
                            "From a single match event file to team and player intelligence "
                            "across Europe’s top leagues.",
                            className="landing-hero-copy",
                        ),
                        html.Div([
                            dcc.Link(
                                [html.I(className="fa-solid fa-cloud-arrow-up me-2"), "Upload a match"],
                                href="/upload",
                                className="landing-primary-cta",
                            ),
                            dcc.Link(
                                ["Explore team data", html.I(className="fa-solid fa-arrow-right-long ms-2")],
                                href="/team-stats",
                                className="landing-secondary-cta",
                            ),
                        ], className="landing-hero-actions"),
                        html.Div([
                            html.Span([html.I(className="fa-solid fa-check"), " Manual JSON upload"]),
                            html.Span([html.I(className="fa-solid fa-check"), " Role-adjusted metrics"]),
                            html.Span([html.I(className="fa-solid fa-check"), " Multi-season analysis"]),
                        ], className="landing-capabilities"),
                    ], className="landing-hero-content"),
                ], className="landing-hero-body"),
            ], className="landing-shell landing-hero-inner"),
        ], className="landing-hero"),

        html.Section([
            html.Div([
                html.Span("ANALYSIS WORKSPACES", className="landing-section-kicker"),
                html.H2("Choose where to start"),
                html.P(
                    "Analyze one manually collected match or explore season-level performance.",
                    className="landing-section-copy",
                ),
            ], className="landing-section-heading"),
            dbc.Row([
                _workspace_card(
                    "Upload Match",
                    "Load a manually scraped Opta-style JSON event file and open the complete match-analysis report.",
                    "SINGLE MATCH",
                    "fa-solid fa-cloud-arrow-up",
                    "/upload",
                    "coral",
                    featured=True,
                ),
                _workspace_card(
                    "Team Statistics",
                    "Compare clubs, tactical profiles and direct or derived performance metrics across seasons.",
                    "TEAM INTELLIGENCE",
                    "fa-solid fa-chart-column",
                    "/team-stats",
                    "teal",
                ),
                _workspace_card(
                    "Player Statistics",
                    "Discover leaders, role-adjusted percentiles, player profiles and positional usage.",
                    "PLAYER INTELLIGENCE",
                    "fa-solid fa-user-shield",
                    "/player-stats",
                    "blue",
                ),
            ], className="g-4"),
        ], className="landing-shell landing-workspaces"),

        html.Section([
            html.Div([
                html.Div([
                    html.Span("MANUAL MATCH WORKFLOW", className="landing-section-kicker"),
                    html.H2("From raw events to an analyst-ready report"),
                    html.P(
                        "The app does not depend on a stored match database. Scrape the fixture you need, "
                        "upload it and keep the analysis focused on the match in front of you.",
                    ),
                ], className="landing-workflow-copy"),
                html.Div([
                    _workflow_step("01", "Upload JSON", "Select the event file collected for the fixture.", "fa-solid fa-file-code"),
                    _workflow_step("02", "Validate", "The app parses teams, players, events and qualifiers.", "fa-solid fa-circle-check"),
                    _workflow_step("03", "Analyze", "Open the interactive tactical and statistical report.", "fa-solid fa-chart-line"),
                ], className="landing-workflow-steps"),
            ], className="landing-shell landing-workflow-inner"),
        ], className="landing-workflow"),

        html.Footer([
            html.Div([
                html.Div([
                    html.Img(src="/assets/brand/app_logo.svg", alt=""),
                    html.Span("Il Lab dell’8"),
                ], className="landing-footer-brand"),
                html.Span("Football analytics by Michele Orefice", className="landing-footer-copy"),
            ], className="landing-shell landing-footer-inner"),
        ], className="landing-footer"),
    ], className="landing-page")
