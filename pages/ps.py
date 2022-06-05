import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output, State, callback, no_update
import plotly.express as px
import numpy as np
import pandas as pd
import utils.compute as cmp
import os


def get_v(x, v, bl, bg, c1, c2, c3):
    r = np.random.uniform(0, 1, 3)
    return c1 * r[0] * float(v) + c2 * r[1] * (float(bl) - float(x)) + c3 * r[2] * (float(bg) - float(x))


def get_bg_for(p_id: int, particles: pd.DataFrame, vicinity: float):
    df = particles.copy()
    pdf = df.iloc[[p_id]]
    dist = []
    p = float(df["particle"][p_id])
    df = df.drop(index=p_id)
    for x in df["particle"]:
        dist.append(abs(p - float(x)))
    df.insert(loc=4, column="dist", value=dist)
    df = df.sort_values("dist")
    how_many = round(vicinity / 100 * df.shape[0])
    nbhood = pd.concat([pdf, df.iloc[:how_many]])
    nbhood = nbhood.sort_values("bg_fxs", ascending=False)
    nbhood = nbhood.reset_index()
    return nbhood["bg"][0]


def particle_swarm(a, b, n, d, time_t, c1, c2, c3, vicinity):
    dot_places = cmp.compute_precision(d)
    particles = cmp.add_precision(cmp.generate_population(a, b, n), d)
    vs = cmp.add_precision(cmp.generate_population(a, b, n), d)
    best_locals = list(particles)
    best_globals = list(particles)
    bounds = [f'{a:.{dot_places}f}', f'{b:.{dot_places}f}']
    p_fxs = [cmp.compute_fx(float(x)) for x in particles]
    bl_fxs = [cmp.compute_fx(float(x)) for x in best_locals]
    bg_fxs = [cmp.compute_fx(float(x)) for x in best_globals]
    df = pd.DataFrame({
        "particle": particles,
        "p_fxs": p_fxs,
        "bl": best_locals,
        "bl_fxs": bl_fxs,
        "bg": best_globals,
        "bg_fxs": bg_fxs,
    })
    results = [df]

    for t in range(time_t):
        best_globals = [get_bg_for(i, df, vicinity) for i in range(n)]
        bg_fxs = [cmp.compute_fx(float(x)) for x in best_globals]
        new_particles = []
        new_vs = np.zeros(n).tolist()
        for i in range(n):
            if p_fxs[i] > bl_fxs[i]:
                best_locals[i] = particles[i]
            elif p_fxs[i] > bg_fxs[i]:
                best_globals[i] = particles[i]
        for i in range(n):
            new_vs[i] = get_v(
                particles[i],
                vs[i],
                best_locals[i],
                best_globals[i],
                c1, c2, c3)
            new_particle = float(particles[i]) + new_vs[i]
            if round(new_particle, dot_places) < round(float(bounds[0]), dot_places):
                new_particle = float(bounds[0])
            elif round(new_particle, dot_places) > round(float(bounds[1]), dot_places):
                new_particle = float(bounds[1])
            new_particles.append(new_particle)
        new_particles = cmp.add_precision(new_particles, d)
        particles = list(new_particles)
        vs = list(new_vs)
        p_fxs = [cmp.compute_fx(float(x)) for x in particles]
        bl_fxs = [cmp.compute_fx(float(x)) for x in best_locals]
        bg_fxs = [cmp.compute_fx(float(x)) for x in best_globals]
        df = pd.DataFrame({
            "particle": particles,
            "p_fxs": p_fxs,
            "bl": best_locals,
            "bl_fxs": bl_fxs,
            "bg": best_globals,
            "bg_fxs": bg_fxs,
        })
        results.append(df)
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
                                      max=5000, placeholder='N', value=100),
                        ], className='me-3'),
                        ], width=2),
                dbc.Col([
                        dbc.InputGroup([
                            dbc.InputGroupText("T:"),
                            dbc.Input(id='t_value', type='number', min=1,
                                      max=5000, placeholder='T', value=100),
                        ], className='me-3'),
                        ], width=2),
            ]),
        dbc.Row([
                dbc.Col([
                    dbc.InputGroup([
                        dbc.InputGroupText("c1:"),
                        dbc.Input(id='c1_value', type='number', min=0, max=3,
                                  placeholder='c1', value=0.8),
                    ], className='me-3'),
                ], width=2),
                dbc.Col([
                    dbc.InputGroup([
                        dbc.InputGroupText("c2:"),
                        dbc.Input(id='c2_value', type='number', min=0, max=3,
                                  placeholder='c2', value=1.0),
                    ], className='me-3'),
                ], width=2),
                dbc.Col([
                    dbc.InputGroup([
                        dbc.InputGroupText("c3:"),
                        dbc.Input(id='c3_value', type='number', min=0, max=3,
                                  placeholder='c3', value=1.2),
                    ], className='me-3'),
                ], width=2),
                dbc.Col([
                    dbc.InputGroup([
                        dbc.InputGroupText("v:"),
                        dbc.Input(id='v_value', type='number', min=1, max=100,
                                  placeholder='vicinity', value=50),
                        dbc.InputGroupText("%"),
                    ], className='me-3'),
                ], width=2),
                dbc.Col([
                    html.Div([
                        dbc.Button("Start PS", outline=True, color='info',
                                   size='lg', id='submit_button', n_clicks=0),
                    ], style={'textAlign': 'center'}),
                ], width=2),
                ], className='mt-2'),
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
                html.Br(),
                dcc.Loading(children=[
                    html.Div([], id="ps_graph"),
                    html.Br(),
                    html.Div([], id='ps_mmm_graph'),
                    html.Br(),
                    html.Div([], id='ps_result_table',
                             className="fs-5 font-monospace"),
                    html.Br(),
                ]),
                html.Br(),
                html.Br(),
                html.Div([
                    dbc.Button("Test results", id="ps_test_button", outline=True, color="success",
                               size="lg", n_clicks=0, style={'display': 'none'}),
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
           Output('ps_mmm_graph', 'children'),
           Output('ps_error_msg', 'children'),
           Input('submit_button', 'n_clicks'),
           State('a_value', 'value'),
           State('b_value', 'value'),
           State('d_value', 'value'),
           State('n_value', 'value'),
           State('t_value', 'value'),
           State('c1_value', 'value'),
           State('c2_value', 'value'),
           State('c3_value', 'value'),
           State('v_value', 'value'),
           prevent_initial_call=True)
def get_ps(n_clicks, input_a, input_b, input_d, input_n, input_t, input_c1, input_c2, input_c3, input_v):
    if None in [input_a, input_b, input_d, input_n, input_t, input_c1, input_c2, input_c3, input_v]:
        return no_update, no_update, no_update, html.Div(
            "Pola wypełniamy wartościami numerycznymi.", style={'color': 'red'})
    elif int(np.ma.round(input_a)) == int(np.ma.round(input_b)):
        return no_update, no_update, no_update, html.Div(
            "Przedział jest zerowy! Podaj prawidłowy przedział za pomocą liczb całkowitych.", style={'color': 'red'})
    elif input_a < -10000000 or input_a > 10000000 or input_b < -10000000 or input_b > 10000000:
        return no_update, no_update, no_update, html.Div(
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
    dot_places = cmp.compute_precision(d)
    n = int(input_n)
    time_t = int(input_t)
    c1 = round(float(input_c1), 1)
    c2 = round(float(input_c2), 1)
    c3 = round(float(input_c3), 1)
    vicinity = round(float(input_v))
    # PCA
    df_list = particle_swarm(a, b, n, d, time_t, c1, c2, c3, vicinity)

    zeros = np.zeros(n * time_t)
    times = np.arange(time_t)
    df = pd.DataFrame()
    df = pd.concat(df_list, keys=times)
    df = df.reset_index()
    df.columns = ["time_t", "p_id", "particle",
                  "p_fxs", "bl", "bl_fxs", "bg", "bg_fxs"]
    df.insert(loc=0, column="zeros", value=zeros)
    size = get_size(df["p_fxs"])
    df.insert(loc=9, column="size", value=size)

    rdf = df.copy()
    df['particle'] = df['particle'].astype(float)
    ps_fig = px.scatter(
        df,
        x="particle",
        y="zeros",
        labels={"particle": "particle position", "zeros": "reference point"},
        color="p_id",
        size="size",
        size_max=20,
        animation_frame="time_t",
        animation_group="p_id",
        range_x=[-4, 12],
        range_y=[-2, 2],
        hover_data={
            "p_fxs": f':.{dot_places}f',
            "bl": True,
            "bl_fxs": f':.{dot_places}f',
            "bg": True,
            "bg_fxs": f':.{dot_places}f',
            "zeros": False,
            "time_t": False,
            "size": False,
        }
    )
    ps_fig.update_layout({'xaxis.autorange': True, 'yaxis.autorange': True})

    path = './results'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    df = df.drop(columns=["zeros", "size"])
    df.to_csv(f'./results/psresult{n}_{time_t}.csv', index_label="lp")

    rdf = rdf.sort_values(["p_fxs", "time_t"], ascending=[False, True])
    rdf = rdf.reset_index()
    rdf = rdf.drop(columns=["index", "zeros", "bl",
                   "bl_fxs", "bg", "bg_fxs", "size"])
    rdf.columns = ["period T", "particle id", "particle", "value of fx(max)"]
    df = df.groupby(by="time_t").agg({"p_fxs": ['max', 'mean', 'min']})
    df.columns = df.columns.droplevel(0)
    df.columns = ["fx_max", "fx_avg", "fx_min"]
    ps_mmm_fig = px.line(df,
                         y=["fx_max", "fx_avg", "fx_min"],
                         title="Plot of f_max(particle), f_avg(particle) oraz f_min(particle)",
                         markers=True)
    return dbc.Table.from_dataframe(rdf.iloc[[0]], striped=True, bordered=True, hover=True), \
        dcc.Graph("ps_result_graph", figure=ps_fig, config={
            'autosizable': True,
            'responsive': True,
        }), dcc.Graph("ps_minmax_graph", figure=ps_mmm_fig), ""
