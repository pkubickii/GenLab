from dash import Dash, html, dcc, Input, Output, State


layout = html.Div(
    [
        html.P("Homepage, welcome"),
        dcc.Link("Go to GEO", href="/geo")
    ]
)
