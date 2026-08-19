from dash import dcc, html
import dash_bootstrap_components as dbc


def _check_item(icon, title, description):
    return html.Div([
        html.Div(html.I(className=icon), className="upload-check-icon"),
        html.Div([
            html.Strong(title),
            html.Span(description),
        ]),
    ], className="upload-check-item")


def _process_step(number, title, description):
    return html.Div([
        html.Span(number, className="upload-process-number"),
        html.Div([
            html.Strong(title),
            html.Span(description),
        ]),
    ], className="upload-process-step")


def layout():
    """Page dedicated to uploading and analysing a single match file."""
    return html.Main([
        html.Header([
            html.Div([
                html.Nav([
                    dcc.Link([
                        html.I(className="fa-solid fa-arrow-left-long"),
                        html.Span("Home"),
                    ], href="/", className="upload-back-link"),
                    html.Div([
                        html.Img(src="/assets/brand/app_logo.svg", alt="Il Lab dell'8"),
                        html.Div([
                            html.Strong("Il Lab dell’8"),
                            html.Span("MATCH ANALYSIS WORKSPACE"),
                        ]),
                    ], className="upload-brand"),
                ], className="upload-nav"),

                html.Div([
                    html.Div([
                        html.Div([
                            html.Span(className="upload-live-dot"),
                            "SINGLE MATCH ANALYSIS",
                        ], className="upload-eyebrow"),
                        html.H1([
                            "Turn match events ",
                            html.Span("into insight."),
                        ]),
                        html.P(
                            "Upload the event file for the fixture you want to study. "
                            "The app validates the data and opens the complete interactive report.",
                        ),
                    ], className="upload-hero-copy"),
                    html.Div([
                        html.Div([
                            html.I(className="fa-solid fa-shield-halved"),
                            html.Div([
                                html.Strong("Local analysis flow"),
                                html.Span("No stored match database required"),
                            ]),
                        ], className="upload-hero-note"),
                        html.Div([
                            html.Span([html.I(className="fa-solid fa-file-code"), " JSON event data"]),
                            html.Span([html.I(className="fa-solid fa-bolt"), " Automatic redirect"]),
                        ], className="upload-hero-badges"),
                    ], className="upload-hero-aside"),
                ], className="upload-hero-content"),
            ], className="upload-shell"),
        ], className="upload-hero"),

        html.Section([
            dbc.Row([
                dbc.Col([
                    html.Article([
                        html.Div([
                            html.Div([
                                html.Span("EVENT FILE", className="upload-card-kicker"),
                                html.H2("Upload a match"),
                                html.P("Select one valid JSON file containing the events for a single fixture."),
                            ]),
                            html.Span([
                                html.I(className="fa-solid fa-circle-check"),
                                " Ready",
                            ], className="upload-ready-badge"),
                        ], className="upload-card-header"),

                        dcc.Upload(
                            id="upload-data",
                            children=html.Div([
                                html.Div(
                                    html.I(className="fa-solid fa-cloud-arrow-up"),
                                    className="upload-drop-icon",
                                ),
                                html.H3("Drop your match file here"),
                                html.P([
                                    "or ",
                                    html.Span("browse your computer"),
                                ]),
                                html.Div([
                                    html.Span(".JSON"),
                                    html.Span("ONE FIXTURE"),
                                    html.Span("EVENT-LEVEL DATA"),
                                ], className="upload-file-tags"),
                            ], className="upload-drop-content"),
                            className="upload-dropzone",
                            multiple=False,
                        ),

                        dcc.Loading(
                            html.Div(id="upload-status-output", className="upload-status-output"),
                            type="circle",
                            color="#0b87aa",
                        ),

                        html.Div([
                            html.Div([
                                html.I(className="fa-solid fa-lock"),
                                html.Span("The file is processed in your active app session."),
                            ]),
                            html.Span("Maximum one file per upload", className="upload-limit-copy"),
                        ], className="upload-card-footer"),
                    ], className="upload-card"),
                ], lg=8, md=12),

                dbc.Col([
                    html.Aside([
                        html.Span("BEFORE YOU UPLOAD", className="upload-card-kicker"),
                        html.H2("Quick checklist"),
                        html.P(
                            "A clean input file avoids interruptions and produces a complete report.",
                            className="upload-checklist-intro",
                        ),
                        html.Div([
                            _check_item(
                                "fa-solid fa-code",
                                "Valid JSON structure",
                                "The file must be readable as JSON.",
                            ),
                            _check_item(
                                "fa-solid fa-futbol",
                                "One complete fixture",
                                "Include teams, players and match events.",
                            ),
                            _check_item(
                                "fa-solid fa-list-check",
                                "Event qualifiers",
                                "Keep the available Opta-style qualifiers.",
                            ),
                        ], className="upload-checklist"),
                        html.Div([
                            html.I(className="fa-solid fa-lightbulb"),
                            html.P([
                                html.Strong("Manual collection is supported. "),
                                "Scrape only the match you need, then upload the raw event file here.",
                            ]),
                        ], className="upload-tip"),
                    ], className="upload-side-card"),
                ], lg=4, md=12),
            ], className="g-4"),

            html.Div([
                html.Div([
                    html.Span("WHAT HAPPENS NEXT", className="upload-card-kicker"),
                    html.H2("One file, three steps"),
                ], className="upload-process-heading"),
                html.Div([
                    _process_step("01", "Read", "Parse match metadata and raw events."),
                    _process_step("02", "Validate", "Map players, teams and qualifiers."),
                    _process_step("03", "Analyze", "Open the interactive match report."),
                ], className="upload-process-list"),
            ], className="upload-process-card"),
        ], className="upload-shell upload-content"),

        html.Footer([
            html.Div([
                html.Div([
                    html.Img(src="/assets/brand/app_logo.svg", alt=""),
                    html.Strong("Il Lab dell’8"),
                ], className="upload-footer-brand"),
                html.Span("Where match events become football intelligence."),
            ], className="upload-shell upload-footer-inner"),
        ], className="upload-footer"),
    ], className="upload-page")
