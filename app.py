import dash
from dash import dcc
from dash import html
import pandas as pd
import numpy as np
import compute as cmp
import selection as sel
import crossover as cross
import mutation as mut
import elite
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
        ], style={'font-family': 'Courier', 'white-space': 'pre'})
    ])


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server
app.title = 'GenLab'

app.layout = html.Div(children=[
    html.H1(children='Genetic Laboratory'),

    html.Div([
        html.Div([
            html.Label('Początek przedziału:'),
            dcc.Input(id='a_value', type='number', placeholder='wprowadź a', value=-4),
        ]),
        html.Div([
            html.Label('Koniec przedziału:'),
            dcc.Input(id='b_value', type='number', placeholder='wprowadź b', value=12),
        ], style={'margin-left': '10px'}),
        html.Div([
            html.Label('Ilość osobników:'),
            dcc.Input(id='n_value', type='number', min=1, max=100, placeholder='wprowadź n', value=10),
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
                         ], value=(10 ** -3), clearable=False
                         ),
        ], style={'margin-left': '10px'}),
        html.Div([
            html.Label('pk:'),
            dcc.Input(id='pk_value', type='number', min=0, max=1, placeholder='pk', value=0.75),
        ], style={'margin-left': '10px'}),
        html.Div([
            html.Label('pm:'),
            dcc.Input(id='pm_value', type='number', min=0, max=1, placeholder='pm', value=0.005),
        ], style={'margin-left': '10px'}),
        html.Div([
            html.Label('elita:'),
            dcc.Checklist(id='elite_value',
                          options=[
                              {'label': 'włączona', 'value': 'on'},
                          ], value=['on'])
        ], style={'margin-left': '10px'}),
        html.Div([
            html.Label('T:'),
            dcc.Input(id='t_value', type='number', min=1, max=10000000, placeholder='T', value=1),
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

    dcc.Input(id='p_value'),

])


@app.callback(Output('population_table', 'children'),
              Output('a_value', 'value'),
              Output('b_value', 'value'),
              Output('n_value', 'value'),
              Output('p_value', 'value'),  # test
              Input('submit_button', 'n_clicks'),
              State('a_value', 'value'),
              State('b_value', 'value'),
              State('n_value', 'value'),
              State('d_value', 'value'),
              State('pk_value', 'value'),
              State('pm_value', 'value'),
              State('elite_value', 'value'),
              State('t_value', 'value'))
def update_table(n_clicks, input_a, input_b, input_n, input_d, input_pk, input_pm, input_elite, input_t):
    if None in [input_a, input_b, input_n, input_pk, input_pm, input_t]:
        return html.Div("Pola wypełniamy wartościami numerycznymi, wartość n w przedziale: [1:100]",
                        style={'color': 'red'}), input_a, input_b, input_n
    elif int(np.ma.round(input_a)) == int(np.ma.round(input_b)):
        return html.Div("Przedział jest zerowy! Podaj prawidłowy przedział za pomocą liczb całkowitych.",
                        style={'color': 'red'}), input_a, input_b, input_n
    elif input_a < -10000000 or input_a > 10000000 or input_b < -10000000 or input_b > 10000000:
        return html.Div("Przedział jest za duży! Podaj prawidłowy przedział z zakresu [-10M: 10M].",
                        style={'color': 'red'}), input_a, input_b, input_n

    if input_a > input_b:
        a = int(np.ma.round(input_b))
        b = int(np.ma.round(input_a))
    else:
        a = int(np.ma.round(input_a))
        b = int(np.ma.round(input_b))

    n = int(np.ma.round(input_n))
    d = input_d
    pk = input_pk
    pm = input_pm

    length = cmp.compute_length(a, b, d)
    x_reals = cmp.add_precision(cmp.generate_population(a, b, n), d)

    sel_reals = []
    sel_bins = []
    parent_bins = []
    cross_points = []
    children_w_cp = []
    pop_after_cross = []
    mutation_indices_formatted = []
    pop_after_mut = []
    x_reals_after_cross_mut = []
    fxs_cross_mutation = []
    p_temp = ''

    elite_memo = elite.get_best(x_reals, [cmp.compute_fx(float(x)) for x in x_reals])
    # main loop:
    for i in range(input_t):
        fxs = [cmp.compute_fx(float(x)) for x in x_reals]
        gxs = sel.compute_gxs(fxs, min(fxs), d)
        pxs = sel.compute_pxs(gxs)
        qxs = sel.compute_qxs(pxs)
        rs = sel.compute_r(n)
        sel_reals = sel.get_new_population(rs, qxs, x_reals)
        sel_fxs = [cmp.compute_fx(float(x)) for x in sel_reals]
        sel_ints = [cmp.compute_x_int(float(x), length, a, b) for x in sel_reals]
        sel_bins = [cmp.compute_x_bin(x, length) for x in sel_ints]
        parent_bins = cross.get_parents(sel_bins, pk)
        cross_points = cross.get_cross_points(parent_bins, length)
        children_bins = cross.get_children(parent_bins, cross_points)
        children_w_cp = cross.get_children_w_cp(children_bins, cross_points)
        pop_after_cross = cross.get_pop_after_cross(children_bins, sel_bins)
        mutation_indices = [mut.get_mutation_indices(length, pm) for _ in range(n)]
        mutation_indices_formatted = [f'{x}' for x in mutation_indices]
        pop_after_mut = mut.mutation(pop_after_cross, mutation_indices)
        x_reals_after_cross_mut = cmp.add_precision(cmp.compute_xreals_from_xbins(a, b, length, pop_after_mut), d)
        p_temp = f'{elite.get_best(sel_reals, sel_fxs)[0]} f(x): {elite.get_best(sel_reals, sel_fxs)[-1]}'
        if input_elite is not None:
            fxs_after_cm = [cmp.compute_fx((float(x))) for x in x_reals_after_cross_mut]
            elite_new = elite.get_best(x_reals_after_cross_mut, fxs_after_cm)
            x_reals_after_cross_mut = elite.inject(elite_memo, x_reals_after_cross_mut)
            elite_memo = elite_new
        fxs_cross_mutation = [cmp.compute_fx(float(x)) for x in x_reals_after_cross_mut]
        x_reals = x_reals_after_cross_mut

    p = p_temp

    df = pd.DataFrame({
        "Lp.": np.arange(1, n + 1),
        "x_real_sel": sel_reals,
        "x_bin_sel": sel_bins,
        "rodzice": parent_bins,
        "pc": cross_points,
        "dzieci": children_w_cp,
        "pop. po krzyżowaniu": pop_after_cross,
        "mutowany gen": mutation_indices_formatted,
        "pop. po mutacji": pop_after_mut,
        "x_real": x_reals_after_cross_mut,
        "f(x)": fxs_cross_mutation,

    })
    return generate_table(df, max_rows=n), a, b, n, p


if __name__ == '__main__':
    app.run_server(debug=True)
