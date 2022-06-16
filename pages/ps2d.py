import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output, State, callback, no_update
import plotly.express as px
import numpy as np
import pandas as pd
import utils.compute as cmp
import os
import math
import json


class Particle:
    def __init__(self, x1=0, x2=0, p=5):
        self.x1 = x1
        self.x2 = x2
        self.p = p
        self.x1 = f'{self.x1:.{self.p}f}'
        self.x2 = f'{self.x2:.{self.p}f}'

    def __str__(self):
        return f'[{self.x1}, {self.x2}]'

    def __repr__(self):
        return f'[{self.x1}, {self.x2}]'

    def get_fx(self):
        return math.pow(float(self.x1) + 2*float(self.x2) - 7, 2) + math.pow(2*float(self.x1) + float(self.x2) - 5, 2)

    def to_list(self, as_float=False):
        if as_float:
            return [float(self.x1), float(self.x2)]
        else:
            return [self.x1, self.x2]

    def move(self, velocity, bounds):
        new_pos = Particle(float(self.x1) + float(velocity.x1),
                           float(self.x2) + float(velocity.x2), self.p)
        if float(new_pos.x1) < bounds[0][0]:
            new_pos.x1 = f'{bounds[0][0]:.{self.p}f}'
        elif float(new_pos.x1) > bounds[0][1]:
            new_pos.x1 = f'{bounds[0][1]:.{self.p}f}'
        if float(new_pos.x2) < bounds[1][0]:
            new_pos.x2 = f'{bounds[1][0]:.{self.p}f}'
        elif float(new_pos.x2) > bounds[1][1]:
            new_pos.x2 = f'{bounds[1][1]:.{self.p}f}'
        return new_pos

    def mul(self, factor: float):
        mullist = [factor * p for p in self.to_list(True)]
        return Particle(mullist[0], mullist[-1])

    def sum(self, particle):
        return Particle(float(self.x1) + float(particle.x1), float(self.x2) + float(particle.x2), self.p)

    def sub(self, particle):
        return Particle(float(self.x1) - float(particle.x1), float(self.x2) - float(particle.x2), self.p)

    def distance(self, particle):
        dist = np.linalg.norm(np.array(self.to_list(
            True)) - np.array(particle.to_list(True)))
        return dist


def generate_particles(a: int, b: int, n: int, p: int) -> Particle:
    x1 = np.random.uniform(a, b, n)
    x2 = np.random.uniform(a, b, n)
    particles = []
    for i in range(n):
        particles.append(Particle(x1[i], x2[i], p))
    return particles


def get_new_velocity(pc, vc, pc_bl, pc_bg, c1, c2, c3):
    r = np.random.uniform(0, 1, 3)

    pc1 = vc.mul(c1 * r[0])
    pc2 = pc_bl.sub(pc).mul(c2 * r[1])
    pc3 = pc_bg.sub(pc).mul(c3 * r[2])
    return pc1.sum(pc2).sum(pc3)


def get_bg_for(p_id: int, swarmdf: pd.DataFrame, vicinity: float):
    df = swarmdf.copy()
    pdf = df.iloc[[p_id]]
    dist = []
    particle = df["particle"][p_id]
    df = df.drop(index=p_id)
    for pc in df["particle"]:
        dist.append(particle.distance(pc))
    df.insert(loc=4, column="dist", value=dist)
    df = df.sort_values("dist")
    how_many = round(vicinity / 100 * df.shape[0])
    nbhood = pd.concat([pdf, df.iloc[:how_many]])
    nbhood = nbhood.sort_values("bg_fxs")
    nbhood = nbhood.reset_index()
    return nbhood["bg"][0]


def particle_swarm(a, b, n, d, time_t, c1, c2, c3, vicinity):
    p = cmp.compute_precision(d)
    particles = generate_particles(a, b, n, p)
    vs = generate_particles(a, b, n, p)
    best_locals = list(particles)
    best_globals = list(particles)
    bounds = [(a, b), (a, b)]
    p_fxs = [particle.get_fx() for particle in particles]
    bl_fxs = list(p_fxs)
    bg_fxs = list(p_fxs)
    df = pd.DataFrame({
        "particle": particles,
        "p_fxs": p_fxs,
        "bl": best_locals,
        "bl_fxs": bl_fxs,
        "bg": best_globals,
        "bg_fxs": bg_fxs,
    })
    results = [df]

    for _ in range(time_t):
        best_globals = [get_bg_for(i, df, vicinity) for i in range(n)]
        bg_fxs = [bg.get_fx() for bg in best_globals]
        new_particles = []
        new_vs = np.zeros(n).tolist()
        for i in range(n):
            if p_fxs[i] < bl_fxs[i]:
                best_locals[i] = particles[i]
            elif p_fxs[i] < bg_fxs[i]:
                best_globals[i] = particles[i]
        for i in range(n):
            new_vs[i] = get_new_velocity(
                particles[i],
                vs[i],
                best_locals[i],
                best_globals[i],
                c1, c2, c3)
            new_particles.append(particles[i].move(new_vs[i], bounds))
        particles = list(new_particles)
        vs = list(new_vs)
        p_fxs = [particle.get_fx() for particle in particles]
        bl_fxs = [particle.get_fx() for particle in best_locals]
        bg_fxs = [particle.get_fx() for particle in best_globals]
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
                                      placeholder='wprowadź a', value=-10),
                        ], className='me-3'),
                        ], width=2),
                dbc.Col([
                        dbc.InputGroup([
                            dbc.InputGroupText("b:"),
                            dbc.Input(id='b_value', type='number',
                                      placeholder='wprowadź b', value=10),
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
                                       value=(10 ** -5),
                                       ),
                        ], className='me-3'),
                        ], width=2),
                dbc.Col([
                        dbc.InputGroup([
                            dbc.InputGroupText("N:"),
                            dbc.Input(id='n_value', type='number', min=1,
                                      max=200, placeholder='N', value=50),
                        ], className='me-3'),
                        ], width=2),
                dbc.Col([
                        dbc.InputGroup([
                            dbc.InputGroupText("T:"),
                            dbc.Input(id='t_value', type='number', min=1,
                                      max=200, placeholder='T', value=50),
                        ], className='me-3'),
                        ], width=2),
            ]),
        dbc.Row([
                dbc.Col([
                    dbc.InputGroup([
                        dbc.InputGroupText("c1:"),
                        dbc.Input(id='c1_value', type='number', min=0, max=3,
                                  placeholder='c1', value=0.7, step=0.1),
                    ], className='me-3'),
                ], width=2),
                dbc.Col([
                    dbc.InputGroup([
                        dbc.InputGroupText("c2:"),
                        dbc.Input(id='c2_value', type='number', min=0, max=3,
                                  placeholder='c2', value=0.8, step=0.1),
                    ], className='me-3'),
                ], width=2),
                dbc.Col([
                    dbc.InputGroup([
                        dbc.InputGroupText("c3:"),
                        dbc.Input(id='c3_value', type='number', min=0, max=3,
                                  placeholder='c3', value=1.2, step=0.1),
                    ], className='me-3'),
                ], width=2),
                dbc.Col([
                    dbc.InputGroup([
                        dbc.InputGroupText("v:"),
                        dbc.Input(id='v_value', type='number', min=1, max=100,
                                  placeholder='vicinity', value=30),
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
            html.H1("Particle Swarm Algorithm 2D (Booth Function)",
                    className='m-1 mb-3'),
            form,
            html.Div([], id='ps2d_error_msg'),
            html.Br(),
        ], style={
            'margin': 'auto',
            'width': '90%',
        }),
        html.Div(
            [
                html.Br(),
                dcc.Loading(children=[
                    html.Div([], id="ps2d_graph"),
                    html.Br(),
                    html.Br(),
                    html.Div([], id='ps2d_mmm_graph'),
                    html.Br(),
                    html.Div([], id='ps2d_result_table',
                             className="fs-5 font-monospace"),
                    html.Br(),
                ]),
                html.Br(),
                html.Br(),
                html.Div([
                    dbc.Button("Test results", id="ps2d_test_button", outline=True, color="success",
                               size="lg", n_clicks=0, style={'display': 'none'}),
                ], style={
                    'textAlign': 'center'
                }),
                html.Br(),
                html.Br(),
                html.Div([], id="ps2d_test_fig"),
                html.Br(),
                html.Br(),
                html.Div([], id='ps2d_test_table'),
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
    result = []
    for fx in fxs:
        if fx > 100:
            result.append(20)
        elif 100 >= fx > 80:
            result.append(19)
        elif 80 >= fx > 70:
            result.append(17)
        elif 70 >= fx > 60:
            result.append(16)
        elif 60 >= fx > 50:
            result.append(15)
        elif 50 >= fx > 40:
            result.append(14)
        elif 40 >= fx > 30:
            result.append(13)
        elif 30 >= fx > 20:
            result.append(12)
        elif 20 >= fx > 10:
            result.append(11)
        elif 10 >= fx > 8:
            result.append(10)
        elif 8 >= fx > 6:
            result.append(9)
        elif 6 >= fx > 4:
            result.append(8)
        elif 4 >= fx > 3:
            result.append(7)
        elif 3 >= fx > 2:
            result.append(6)
        elif 2 >= fx > 1:
            result.append(5)
        elif 1 >= fx > 0.6:
            result.append(4)
        elif 0.6 >= fx > 0.3:
            result.append(3)
        elif 0.3 >= fx > 0.1:
            result.append(2)
        elif fx <= 0.1:
            result.append(1)
    return result


@callback(
    Output('ps2d_result_table', 'children'),
    Output('ps2d_graph', 'children'),
    Output('ps2d_mmm_graph', 'children'),
    Output('ps2d_error_msg', 'children'),
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

    times = np.arange(time_t)
    df = pd.DataFrame()
    df = pd.concat(df_list, keys=times)
    df = df.reset_index()
    df.columns = ["time_t", "p_id", "particle",
                  "p_fxs", "bl", "bl_fxs", "bg", "bg_fxs"]

    particle_list = df['particle'].tolist()
    newlist = json.loads(str(particle_list))
    x1 = []
    x2 = []
    for i in range(len(newlist)):
        x1.append(newlist[i][0])
        x2.append(newlist[i][1])

    df.insert(loc=2, column="x1", value=x1)
    df.insert(loc=3, column="x2", value=x2)
    df = df.drop(columns=["particle", "bl", "bg"])
    rdf = df.copy()
    size = get_size(df["p_fxs"])
    df.insert(loc=7, column="size", value=size)
    ps2d_fig = px.scatter(
        df,
        x="x1",
        y="x2",
        color="p_id",
        size="size",
        size_max=20,
        animation_frame="time_t",
        animation_group="p_id",
        range_x=[-10, 10],
        range_y=[-10, 10],
        height=800,
        hover_data={
            "p_fxs": f':.{dot_places}f',
            "bl_fxs": f':.{dot_places}f',
            "bg_fxs": f':.{dot_places}f',
            "time_t": False,
            "size": False,
        }
    )
    ps2d_fig.update_layout({'xaxis.autorange': True, 'yaxis.autorange': True})

    path = './results'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    df = df.drop(columns=["size"])
    df.to_csv(f'./results/ps2dresult{n}_{time_t}.csv', index_label="lp")

    rdf = rdf.sort_values(["p_fxs", "time_t"], ascending=[True, True])
    rdf = rdf.reset_index()
    rdf = rdf.drop(columns=["index", "bl_fxs", "bg_fxs"])
    rdf.columns = ["period T", "particle id", "x1", "x2", "value of fx(min)"]

    df = df.groupby(by="time_t").agg({"p_fxs": ['min', 'mean', 'max']})
    df.columns = df.columns.droplevel(0)
    df.columns = ["fx_min", "fx_avg", "fx_max"]
    ps2d_mmm_fig = px.line(df,
                           y=["fx_min", "fx_avg", "fx_max"],
                           title="Plot of f_min(particle), f_avg(particle) oraz f_max(particle)",
                           markers=True)

    best_particle = rdf.iloc[[0]]
    x1str = [f'{best_particle["x1"][0]:.{dot_places}f}']
    x2str = [f'{best_particle["x2"][0]:.{dot_places}f}']
    best_particle = best_particle.drop(columns=["x1", "x2"])
    best_particle.insert(loc=2, column="x1", value=x1str)
    best_particle.insert(loc=3, column="x2", value=x2str)
    return dbc.Table.from_dataframe(best_particle, striped=True, bordered=True, hover=True), \
        dcc.Graph("ps2d_result_graph", figure=ps2d_fig, config={
            'autosizable': True,
        }), dcc.Graph("ps2d_minmax_graph", figure=ps2d_mmm_fig), ""
