import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output, State, callback, no_update
import utils.compute as cmp
import pandas as pd
import numpy as np
import plotly.express as px


def get_start_vbs(vb):
    start_vbs = []
    for i in range(len(vb)):
        vb_temp = list(vb)
        if vb_temp[i] == '0':
            vb_temp[i] = '1'
        else:
            vb_temp[i] = '0'
        start_vbs.append(''.join(vb_temp))
    return start_vbs


def get_ranks(tau, length):
    ranks = np.zeros(length)
    for r in range(length):
        ranks[r] = (r + 1) ** -tau
    return ranks


def get_mut_indices(ranked_index, ranks):
    randoms = np.random.uniform(0, 1, len(ranked_index))
    mutation_indices = []
    for i in range(len(ranked_index)):
        if randoms[i] < ranks[i]:
            mutation_indices.append(ranked_index[i])
    return mutation_indices


def get_mutated_vb(vb, mut_indices):
    vb_temp = list(vb)
    for mi in mut_indices:
        if vb_temp[mi] == '0':
            vb_temp[mi] = '1'
        else:
            vb_temp[mi] = '0'
    return ''.join(vb_temp)


def vbs_best_in_t(vbs_fx):
    vbs_best = []
    vbs_best.append(vbs_fx[0])
    for i in range(1, len(vbs_fx)):
        if vbs_fx[i] > vbs_best[i - 1]:
            vbs_best.append(vbs_fx[i])
        else:
            vbs_best.append(vbs_best[i-1])
    return vbs_best


def geo(vb, time_t, a, b, length, d, tau):
    vbs_best = []
    ranks = get_ranks(tau, length)
    for t in range(time_t):
        start_vbs = get_start_vbs(vb)
        start_rvbs = cmp.add_precision(
            cmp.compute_xreals_from_xbins(a, b, length, start_vbs), d)
        start_fxs = [cmp.compute_fx(float(x)) for x in start_rvbs]
        df = pd.DataFrame(
            {
                "fx": start_fxs,
            })
        df = df.sort_values("fx", ascending=False)
        ranked_index = df.index.to_list()
        mutation_indices = get_mut_indices(ranked_index, ranks)
        vb = get_mutated_vb(vb, mutation_indices)
        vbs_best.append(vb)
    return vbs_best


form = dbc.Form(
    [
        dbc.Row(
            [
                dbc.Col([
                        dbc.InputGroup([
                            dbc.InputGroupText("a:"),
                            dbc.Input(id='a_value', type='number',
                                      placeholder='wprowadź a', value=-4),
                        ], className='me-3'),
                        ], width=2),
                dbc.Col([
                        dbc.InputGroup([
                            dbc.InputGroupText("b:"),
                            dbc.Input(id='b_value', type='number',
                                      placeholder='wprowadź b', value=12),
                        ], className='me-3'),
                        ], width=2),
                dbc.Col([
                        dbc.InputGroup([
                            dbc.InputGroupText("d:"),
                            dbc.Select(id='d_value',
                                       options=[
                                           {'label': '10^-2',
                                               'value': (10 ** -2)},
                                           {'label': '10^-3',
                                               'value': (10 ** -3)},
                                           {'label': '10^-4',
                                               'value': (10 ** -4)},
                                           {'label': '10^-5',
                                               'value': (10 ** -5)},
                                           {'label': '10^-6',
                                               'value': (10 ** -6)},
                                       ],
                                       value=(10 ** -3),
                                       ),
                        ], className='me-3'),
                        ], width=2),
                dbc.Col([
                        dbc.InputGroup([
                            dbc.InputGroupText("tau:"),
                            dbc.Input(id='tau_value', type='number', min=0,
                                      max=5, placeholder='tau', value=1.4, step=0.1),
                        ], className='me-3'),
                        ], width=2),
                dbc.Col([
                        dbc.InputGroup([
                            dbc.InputGroupText("T:"),
                            dbc.Input(id='t_value', type='number', min=1,
                                      max=5000, placeholder='T', value=300),
                        ], className='me-3'),
                        ], width=2),
                dbc.Col([
                        dbc.Button("Uruchom GEO", outline=True, color='info',
                                   size='lg', id='submit_button', n_clicks=0),
                        ], width=2)
            ])
    ])


layout = html.Div(
    [
        html.Div([
            html.H1("Generalized Extremal Optimization",
                    className='m-1 mb-3'),
            form,
            html.Div([], id='geo_error_msg'),
            html.Div([], id='download_button', style={'display': 'none'}),
            html.Div([], id='population_table', style={'display': 'none'}),
            html.Div([], id='n_value', style={'display': 'none'}),
            html.Div([], id='ag_graph', style={'display': 'none'}),
            html.Div([], id='error_msg', style={'display': 'none'}),
            html.Div([], id='pk_value', style={'display': 'none'}),
            html.Div([], id='pm_value', style={'display': 'none'}),
            html.Div([], id='elite_value', style={'display': 'none'}),
            html.Div([], id='generate_table', style={'display': 'none'}),
            html.Div([], id='download_populations', style={'display': 'none'}),
            html.Br(),
        ], style={
            'margin': 'auto',
            'width': '90%',
        }),
        html.Div(
            [
                dcc.Graph(id='result_fig'),
                html.Br(),
                dcc.Loading(children=[
                    html.Div([
                    ], id='result_best', className="fs-5 font-monospace")
                ]),
                html.Br(),
                html.Div([
                    dbc.Button("Wynik testów", id="test_button", outline=True, color="success",
                               size="lg", n_clicks=0),
                ], style={
                    'textAlign': 'center'
                }),
                html.Br(),
                html.Br(),
                html.Div([], id="test_graph"),
                html.Br(),
                html.Br(),
                html.Br(),
            ], style={
                'margin': 'auto',
                'width': '90%',
            }),
        html.Br(),
    ]
)


@callback(Output('test_graph', 'children'),
          Input('test_button', 'n_clicks'),
          prevent_initial_call=True)
def test_button(button_click):
    df = pd.read_csv("pages/assets/geo/geotest_2000_500.csv")
    test_fig = px.line(df,
                       x="tau",
                       y="fxs",
                       title="Wykres fx_avg(vb) dla różnych tau przy T = 500 oraz ilości prób = 2000 na jedną wartość tau",
                       markers="true")
    return dcc.Graph(id="test_graph", figure=test_fig)


@callback(Output('result_best', 'children'),
          Output('result_fig', 'figure'),
          Output('geo_error_msg', 'children'),
          Input('submit_button', 'n_clicks'),
          State('a_value', 'value'),
          State('b_value', 'value'),
          State('d_value', 'value'),
          State('tau_value', 'value'),
          State('t_value', 'value'),
          prevent_initial_call=True)
def get_table(n_clicks, input_a, input_b, input_d, input_tau, input_t):
    if None in [input_a, input_b, input_d, input_tau, input_t]:
        return no_update, no_update, html.Div(
            "Pola wypełniamy wartościami numerycznymi.", style={'color': 'red'})
    elif int(np.ma.round(input_a)) == int(np.ma.round(input_b)):
        return no_update, no_update, html.Div(
            "Przedział jest zerowy! Podaj prawidłowy przedział za pomocą liczb całkowitych.", style={'color': 'red'})
    elif input_a < -10000000 or input_a > 10000000 or input_b < -10000000 or input_b > 10000000:
        return no_update, no_update, html.Div(
            "Przedział jest za duży! Podaj prawidłowy przedział z zakresu [-10M: 10M]",
            style={'color': 'red'})
    if input_a > input_b:
        a = int(np.ma.round(input_b))
        b = int(np.ma.round(input_a))
    else:
        a = int(np.ma.round(input_a))
        b = int(np.ma.round(input_b))

    d = float(input_d)
    time_t = int(input_t)
    tau = float(input_tau)
    length = cmp.compute_length(a, b, d)
    dot_places = cmp.compute_precision(d)
    rand_float = np.random.uniform(a, b)
    vb_real = f'{rand_float:.{dot_places}f}'
    vb_int = cmp.compute_x_int(float(vb_real), length, a, b)
    vb = cmp.compute_x_bin(vb_int, length)

    vbs_best = geo(vb, time_t, a, b, length, d, tau)

    vbrs_best = cmp.add_precision(
        cmp.compute_xreals_from_xbins(a, b, length, vbs_best), d)
    fxs_best = [cmp.compute_fx(float(x)) for x in vbrs_best]
    vbs_fx_best = vbs_best_in_t(fxs_best)

    df = pd.DataFrame(
        {
            "vb_real": vbrs_best,
            "vb": vbs_best,
            "fx": fxs_best,
            "fx_best_in_t": vbs_fx_best,
            "okres_t": np.arange(1, time_t + 1)
        }
    )

    geo_fig = px.line(df,
                      x="okres_t",
                      y=["fx", "fx_best_in_t"],
                      title="Wykres przebiegu fx(vb)",
                      labels={"pokolenie": f'Pokolenia dla T={time_t}',
                              "value": "Wartości f(x)"},
                      markers=True)

    df = df.sort_values(["fx", "fx_best_in_t"], ascending=[False, True])
    df.index = np.arange(0, time_t)
    df = df.truncate(after=0)
    df = df.drop(columns=["fx_best_in_t"])
    return dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True), geo_fig, ""
