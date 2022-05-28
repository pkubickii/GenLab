import numpy as np
import pandas as pd
import utils.compute as cmp
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output, State, callback, no_update
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def get_neighbours(vc):
    start_vcs = []
    for i in range(len(vc)):
        vc_temp = list(vc)
        if vc_temp[i] == '0':
            vc_temp[i] = '1'
        else:
            vc_temp[i] = '0'
        start_vcs.append(''.join(vc_temp))
    return start_vcs


def vc_real_from_vc(vc, length, a, b, p):
    vc_int = cmp.x_int_from_x_bin(vc)
    return f'{cmp.compute_x_real(vc_int, length, a, b):.{p}f}'


def hill_climbing(a, b, d, length, dot_places):
    stop = False
    rand_float = np.random.uniform(a, b)
    vc_real = f'{rand_float:.{dot_places}f}'
    vc_int = cmp.compute_x_int(float(vc_real), length, a, b)
    vc = cmp.compute_x_bin(vc_int, length)
    vcfx_results = [cmp.compute_fx(float(vc_real))]
    vcr_results = [vc_real]
    vc_results = [vc]
    while not stop:
        vcs_neighbours = get_neighbours(vc)
        vcrs_neighbours = cmp.add_precision(
            cmp.compute_xreals_from_xbins(a, b, length, vcs_neighbours), d)
        fxs_neighbours = [cmp.compute_fx(float(x)) for x in vcrs_neighbours]
        vc_fx = cmp.compute_fx(float(vc_real))
        df = pd.DataFrame({
            "vcs_neighbours": vcs_neighbours,
            "fxs_neighbours": fxs_neighbours,
        })
        df = df.sort_values("fxs_neighbours", ascending=False)
        if vc_fx < df.iloc[0][1]:
            vc = df.iloc[0][0]
            vc_real = vc_real_from_vc(vc, length, a, b, dot_places)
            vc_results.append(df.iloc[0][0])
            vcr_results.append(vc_real)
            vcfx_results.append(df.iloc[0][1])
        else:
            stop = True
    rdf = pd.DataFrame({
        "vc_climb": vc_results,
        "vcr_climb": vcr_results,
        "vcfx_climb": vcfx_results,
    })
    return rdf


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
                            dbc.InputGroupText("T:"),
                            dbc.Input(id='t_value', type='number', min=1,
                                      max=5000, placeholder='T', value=5),
                        ], className='me-3'),
                        ], width=2),
                dbc.Col([
                        dbc.Button("Uruchom HC", outline=True, color='info',
                                   size='lg', id='submit_button', n_clicks=0),
                        ], width=2)
            ])
    ])


layout = html.Div(
    [
        html.Div([
            html.H1("Hill Climbing Algorithm",
                    className='m-1 mb-3'),
            form,
            html.Div([], id='hc_error_msg'),
            html.Br(),
        ], style={
            'margin': 'auto',
            'width': '90%',
        }),
        html.Div(
            [
                dcc.Graph(id='hc_result_fig'),
                html.Br(),
                dcc.Loading(children=[
                    html.Div([
                    ], id='hc_result_table', className="fs-5 font-monospace")
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
    ]
)


@callback(Output('hc_result_table', 'children'),
          Output('hc_result_fig', 'figure'),
          Output('hc_error_msg', 'children'),
          Input('submit_button', 'n_clicks'),
          State('a_value', 'value'),
          State('b_value', 'value'),
          State('d_value', 'value'),
          State('t_value', 'value'),
          prevent_initial_call=True)
def get_table(n_clicks, input_a, input_b, input_d, input_t):
    if None in [input_a, input_b, input_d, input_t]:
        return no_update, no_update, html.Div(
            "Pola wypełniamy wartościami numerycznymi.", style={'color': 'red'})
    elif int(np.ma.round(input_a)) == int(np.ma.round(input_b)):
        return no_update, no_update, html.Div(
            "Przedział jest zerowy! Podaj prawidłowy przedział za pomocą liczb całkowitych.", style={'color': 'red'})
    elif input_a < -10000000 or input_a > 10000000 or input_b < -10000000 or input_b > 10000000:
        return no_update, no_update, html.Div(
            "Przedział jest za duży! Podaj prawidłowy przedział z zakresu [-10M: 10M]",
            style={'color': 'red'})
    # params init:
    if input_a > input_b:
        a = int(np.ma.round(input_b))
        b = int(np.ma.round(input_a))
    else:
        a = int(np.ma.round(input_a))
        b = int(np.ma.round(input_b))

    d = float(input_d)
    time_t = int(input_t)
    length = cmp.compute_length(a, b, d)
    dot_places = cmp.compute_precision(d)
    df_climbs = []

    # main loop:
    for t in range(time_t):
        df_climbs.append(hill_climbing(a, b, d, length, dot_places))

    # chart stuff:
    times = np.arange(len(df_climbs))
    df = pd.concat(df_climbs, keys=times)
    df.insert(loc=0, column="period", value=df.index.get_level_values(0))
    t_steps = []
    for i in range(time_t):
        one_period = df.loc[i]["period"].to_list()
        t_steps.append(np.around(np.linspace(
            one_period[0]+0.0, one_period[0]+1.0, num=len(one_period)+2), 2)[1:-1])
    t_steps_flat = np.concatenate(t_steps).tolist()

    fxs_climb = [df.loc[0]["vcfx_climb"].to_list()[0]]
    vcr_climb = [df.loc[0]["vcr_climb"].to_list()[0]]
    fxs_all = df["vcfx_climb"].to_list()
    vcr_all = df["vcr_climb"].to_list()
    for i in range(1, df.shape[0]):
        if fxs_climb[i-1] < fxs_all[i]:
            fxs_climb.append(fxs_all[i])
            vcr_climb.append(vcr_all[i])
        else:
            fxs_climb.append(fxs_climb[i-1])
            vcr_climb.append(vcr_climb[i-1])

    hc_result_fig = make_subplots(
        rows=1,
        cols=1,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.0,
        horizontal_spacing=0.0,
    )
    for i in range(time_t):
        vcr = df.loc[i]["vcr_climb"].to_list()
        hc_result_fig.add_trace(go.Scatter(
            x=t_steps[i],
            y=df.loc[i]["vcfx_climb"].to_list(),
            line={'dash': 'longdashdot'},
            marker={'size': 12},
            hovertemplate='<br>fx = <b>%{y}</b><br>' +
            'vcr = <b>%{text}</b><br>',
            text=vcr,
            name=f'okres {i}'
        ), row=1, col=1)

    hc_result_fig.add_trace(go.Scatter(
        x=t_steps_flat,
        y=fxs_climb,
        name='fx max',
        hovertemplate='<br>fx = <b>%{y}</b><br>' +
        'vcr = <b>%{text}</b>',
        text=vcr_climb,
        line={'dash': 'solid', 'color': 'red', 'width': 5}
    ), row=1, col=1)

    hc_result_fig.update_layout(
        autosize=True,
        height=800,
        title_text="Hill Climbing Plot",
        hovermode='x unified'
    )
    return dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True), hc_result_fig, ""
