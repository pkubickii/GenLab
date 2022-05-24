from dash import Dash, html, dcc, Input, Output, State


layout = html.Div(
    [
        html.Div([
            html.H1("Genetic Laboratory"),
            dcc.Link("Go to GEO", href="/geo", className='fs-5')
        ], style={
            'margin': 'auto',
            'width': '80%',
        }),
    ]
)
