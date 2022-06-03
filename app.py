import dash
from dash import Dash, html, dcc, Input, Output, callback
from pages import home, geo, ag, hc, ps
import dash_bootstrap_components as dbc

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.css"
app = Dash(__name__, external_stylesheets=[
           dbc.themes.QUARTZ, dbc_css], suppress_callback_exceptions=True)

server = app.server
app.title = 'GenLab'

navbar = dbc.NavbarSimple(
    dbc.DropdownMenu(
        [
            dbc.DropdownMenuItem("Home", href="/"),
            dbc.DropdownMenuItem("AG", href="/ag"),
            dbc.DropdownMenuItem("GEO", href="/geo"),
            dbc.DropdownMenuItem("HC", href="/hc"),
            dbc.DropdownMenuItem("PS", href="/ps"),
        ],
        nav=True,
        label="Nawigacja",
        className="fs-5"
    ),
    brand="ISA LAB",
    color="primary",
    dark=True,
    class_name="mb-2 mx-0",
)

app.layout = dbc.Container(
    [
        navbar,
        dcc.Location(id='url', refresh=False),
        html.Div(id='page-content')
    ],
    fluid=True,
    class_name="m-0 p-0",
)


@callback(Output('page-content', 'children'),
          Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/':
        return home.layout
    elif pathname == '/geo':
        return geo.layout
    elif pathname == '/ag':
        return ag.layout
    elif pathname == '/hc':
        return hc.layout
    elif pathname == '/ps':
        return ps.layout
    else:
        return '404'


if __name__ == '__main__':
    app.run_server(debug=True)
