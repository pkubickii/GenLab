import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output, State, callback, no_update
import plotly.express as px
import numpy as np
import pandas as pd
import utils.compute as cmp


def get_v(x, v, bl, bg, c1, c2, c3):
    r = np.random.uniform(0, 1, 3)
    return c1 * r[0] * float(v) + c2 * r[1] * (float(bl) - float(x)) + c3 * r[2] * (float(bg) - float(x))


def particle_swarm(a, b, n, d, time_t, c1, c2, c3):
    particles = cmp.add_precision(cmp.generate_population(a, b, n), d)
    vs = cmp.add_precision(cmp.generate_population(a, b, n), d)
    best_locals = list(particles)
    bounds = ['-4.000', '12.000']

    df = pd.DataFrame({
        "particle": particles,
        "fp": [cmp.compute_fx(float(x)) for x in particles],
    })

    fxs_sorted = df.sort_values("fp", ascending=False)
    best_global = particles[fxs_sorted.index[0]]
    results = [df]

    for t in range(time_t):
        p_fxs = [cmp.compute_fx(float(x)) for x in particles]
        bl_fxs = [cmp.compute_fx(float(x)) for x in best_locals]
        bg_fx = cmp.compute_fx(float(best_global))
        new_particles = []
        new_vs = np.zeros(len(particles)).tolist()
        for i in range(len(particles)):
            if p_fxs[i] > bl_fxs[i]:
                best_locals[i] = particles[i]
            elif p_fxs[i] > bg_fx:
                best_global = particles[i]
        for i in range(len(particles)):
            new_vs[i] = get_v(
                particles[i],
                vs[i],
                best_locals[i],
                best_global,
                c1, c2, c3)
            new_particle = float(particles[i]) + new_vs[i]
            if np.round(new_particle, 3) < float(bounds[0]):
                new_particle = float(bounds[0])
            elif np.round(new_particle, 3) > float(bounds[1]):
                new_particle = float(bounds[1])
            new_particles.append(new_particle)
        new_particles = cmp.add_precision(new_particles, d)
        particles = list(new_particles)
        vs = list(new_vs)
        results.append(pd.DataFrame({
            "particle": particles,
            "fp": p_fxs,
        }))
    return results




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
                            dbc.InputGroupText("N:"),
                            dbc.Input(id='n_value', type='number', min=1,
                                      max=5000, placeholder='N', value=50),
                        ], className='me-3'),
                        ], width=2),
                dbc.Col([
                        dbc.InputGroup([
                            dbc.InputGroupText("T:"),
                            dbc.Input(id='t_value', type='number', min=1,
                                      max=5000, placeholder='T', value=100),
                        ], className='me-3'),
                        ], width=2),
                dbc.Col([
                        dbc.Button("Uruchom PC", outline=True, color='info',
                                   size='lg', id='submit_button', n_clicks=0),
                        ], width=2)
            ])
    ])


layout = html.Div(
    [
        html.Div([
            html.H1("Particle Swarm Algorithm",
                    className='m-1 mb-3'),
            form,
            html.Div([], id='ps_error_msg'),
            html.Br(),
        ], style={
            'margin': 'auto',
            'width': '90%',
        }),
        html.Div(
            [
                # dcc.Graph(id='ps_result_fig'),
                html.Div([], id="ps_graph"),
                html.Br(),
                dcc.Loading(children=[
                    html.Div([
                    ], id='ps_result_table', className="fs-5 font-monospace")
                ]),
                html.Br(),
                html.Div([
                    dbc.Button("Wynik testów", id="ps_test_button", outline=True, color="success",
                               size="lg", n_clicks=0),
                ], style={
                    'textAlign': 'center'
                }),
                html.Br(),
                html.Br(),
                html.Div([], id="ps_test_fig"),
                html.Br(),
                html.Br(),
                html.Div([], id='ps_test_table'),
                html.Br(),
            ], style={
                'margin': 'auto',
                'width': '90%',
            }),
        html.Br(),
        html.Div([], id='download_button', style={'display': 'none'}),
        html.Div([], id='population_table', style={'display': 'none'}),
        html.Div([], id='ag_graph', style={'display': 'none'}),
        html.Div([], id='error_msg', style={'display': 'none'}),
        html.Div([], id='pk_value', style={'display': 'none'}),
        html.Div([], id='pm_value', style={'display': 'none'}),
        html.Div([], id='elite_value', style={'display': 'none'}),
        html.Div([], id='generate_table', style={'display': 'none'}),
        html.Div([], id='download_populations', style={'display': 'none'}),
    ]


)


def get_size(fxs):
    fmin = min(fxs)
    return [int(round(fx - fmin, 1) * 10 + 1) for fx in fxs]


@ callback(Output('ps_result_table', 'children'),
           Output('ps_graph', 'children'),
           Output('ps_error_msg', 'children'),
           Input('submit_button', 'n_clicks'),
           State('a_value', 'value'),
           State('b_value', 'value'),
           State('d_value', 'value'),
           State('n_value', 'value'),
           State('t_value', 'value'),
           prevent_initial_call=True)
def get_ps(n_clicks, input_a, input_b, input_d, input_n, input_t):
    if None in [input_a, input_b, input_d, input_n, input_t]:
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
    n = int(input_n)
    time_t = int(input_t)
    c1 = 0.8
    c2 = 1.0
    c3 = 1.2
    df_list = particle_swarm(a, b, n, d, time_t, c1, c2, c3)
    zeros = np.zeros(n * time_t)
    times = np.arange(time_t)
    df = pd.DataFrame()
    df = pd.concat(df_list, keys=times)
    df = df.reset_index()
    df.columns = ["time_t", "p_id", "particle", "fp"]
    df.insert(loc=0, column="zeros", value=zeros)
    df['particle'] = df['particle'].astype(float)

    size = get_size(df["fp"]) 
    df.insert(loc=5, column="size", value=size)
    df.to_csv("./results/psresult.csv")
    ps_fig = px.scatter(
        df,
        x="particle",
        y="fp",
        color="p_id",
        size="size",
        size_max=20,
        animation_frame="time_t",
        animation_group="p_id",
        range_x=[-4, 12],
        range_y=[-2, 2],
    )
    ps_fig.update_layout({'xaxis.autorange': True, 'yaxis.autorange': True})
    return no_update, dcc.Graph("ps_result_graph", figure=ps_fig, config={
        'autosizable': True,
        'responsive': True,
    }), ""
