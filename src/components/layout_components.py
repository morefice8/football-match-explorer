from dash import html

def app_signature():
    return html.A([
        html.Img(src="/assets/brand/app_logo.svg", style={
            "height": "40px",
            "marginRight": "8px"
        }),
        html.Span("Il Lab dell’8", style={
            "fontWeight": "bold",
            "fontSize": "18px",
            "color": "white",
            "marginRight": "6px"
        }),
        html.Small("by Michele Orefice", style={
            "fontSize": "16px",
            "color": "rgba(255,255,255,0.6)"
        })
    ],
    href="https://www.micheleorefice.com",
    target="_blank",
    style={
        "position": "fixed",
        "bottom": "12px",
        "right": "20px",
        "display": "flex",
        "alignItems": "center",
        "zIndex": 9999,
        "backgroundColor": "rgba(30,30,30,0.9)",
        "padding": "4px 10px",
        "borderRadius": "10px",
        "boxShadow": "0 2px 6px rgba(0,0,0,0.3)",
        "textDecoration": "none"
    })
