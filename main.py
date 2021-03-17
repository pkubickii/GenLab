import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import compute as cmp
from dash.dependencies import Input, Output, State


def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Genetic Laboratory'),

    html.Div([
        html.Div([
            html.Label('Początek przedziału:'),
            dcc.Input(id='a_value', type='text', placeholder='a', value=-4),
        ]),
        html.Div([
            html.Label('Koniec przedziału:'),
            dcc.Input(id='b_value', type='text', placeholder='b', value=12),
        ], style={'margin-left': '10px'}),
        html.Div([
            html.Label('Ilość osobników:'),
            dcc.Input(id='n_value', type='text', placeholder='n', value=10),
        ], style={'margin-left': '10px'}),
        html.Div([
            html.Label('Dokładność:'),
            dcc.Dropdown(id='d_value',
                         options=[
                             {'label': '10^-2', 'value': (10 ** -2)},
                             {'label': '10^-3', 'value': (10 ** -3)},
                             {'label': '10^-4', 'value': (10 ** -4)},
                             {'label': '10^-5', 'value': (10 ** -5)},
                             {'label': '10^-6', 'value': (10 ** -6)},
                         ], value=(10 ** -3)
                         ),
        ], style={'margin-left': '10px'}),
        html.Div([
            html.Button(id='submit_button', n_clicks=0, children='Generuj populację',
                        style={'margin-left': '10px', 'margin-top': '23px'}),
        ])
    ], style={
        'display': 'flex',
    }),

    html.Br(),
    html.Div([

    ], id='population_table'),
])


@app.callback(Output('population_table', 'children'),
              Input('submit_button', 'n_clicks'),
              State('a_value', 'value'),
              State('b_value', 'value'),
              State('n_value', 'value'),
              State('d_value', 'value'))
def update_table(n_clicks, input_a, input_b, input_n, input_d):
    a = int(input_a)
    b = int(input_b)
    n = int(input_n)
    d = input_d
    length = cmp.compute_length(a, b, d)
    x_reals = cmp.add_precision(cmp.generate_population(a, b, n), d)
    x_ints = [cmp.compute_x_int(float(x_real), length, a, b) for x_real in x_reals]
    x_bins = [cmp.compute_x_bin(x_int, length) for x_int in x_ints]
    x_ints2 = [cmp.x_int_from_x_bin(x_bin) for x_bin in x_bins]
    x_reals2 = cmp.add_precision([cmp.compute_x_real(x_int, length, a, b) for x_int in x_ints], d)
    fxs = cmp.add_precision([cmp.compute_fx(float(x_real2)) for x_real2 in x_reals2], d)
    df = pd.DataFrame({
        "x_real": x_reals,
        "x_int": x_ints,
        "x_bin": x_bins,
        "x_int2": x_ints2,
        "x_real2": x_reals2,
        "f(x)": fxs
    })
    return generate_table(df, max_rows=n)


if __name__ == '__main__':
    app.run_server(debug=True)
