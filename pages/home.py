from dash import Dash, html, dcc, Input, Output, State


layout = html.Div(
    [
        html.Div([
            html.H1("Genetic Laboratory"),
            html.P(dcc.Link("Go to AG", href="/ag", className='fs-5')),
            html.P(dcc.Link("Go to GEO", href="/geo", className='fs-5')),
            html.P(dcc.Link("Go to HC", href="/hc", className='fs-5')),
            html.P(dcc.Link("Go to PS", href="/ps", className='fs-5')),
        ], style={
            'margin': 'auto',
            'width': '80%',
        }),
    ]
)
