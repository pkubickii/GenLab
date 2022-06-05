import dash
from dash import Dash, html, dcc, dash_table, no_update, callback
import dash_daq as daq
import plotly.express as px
import pandas as pd
import numpy as np
import utils.selection as sel
import utils.crossover as cross
import utils.mutation as mut
import utils.elite as elite
import utils.test as test
import utils.bestparams as bp
import utils.compute as cmp
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc


def generate_table(dataframe, max_rows=10):
    return dbc.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ]),
    ],
        color="primary"
    )


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
                                      max=200, placeholder='N', value=70),
                        ], className='me-3'),
                        ], width=2),
                dbc.Col([
                        dbc.InputGroup([
                            dbc.InputGroupText("T:"),
                            dbc.Input(id='t_value', type='number', min=1,
                                      max=200, placeholder='T', value=140),
                        ], className='me-3'),
                        ], width=2),
            ]),
        dbc.Row([
                dbc.Col([
                    dbc.InputGroup([
                        dbc.InputGroupText("pk:"),
                        dbc.Input(id='pk_value', type='number', min=0, max=1,
                                  placeholder='pk', step=0.05, value=0.8),
                    ], className='me-3'),
                ], width=2),
                dbc.Col([
                    dbc.InputGroup([
                        dbc.InputGroupText("pm:"),
                        dbc.Input(id='pm_value', type='number', min=0, max=1,
                                  placeholder='pm', step=0.0001, value=0.0001),
                    ], className='me-3'),
                ], width=2),
                dbc.Col([
                    html.Div([
                        daq.BooleanSwitch(
                            id="elite_value",
                            on=True,
                            label="elite:",
                            labelPosition='top'
                        )
                    ]),
                ], width=2),
                dbc.Col([
                    html.Div([
                        dbc.Button("Start Genetic Algorithm", outline=True, color='info',
                                   size='lg', id='submit_button', n_clicks=0, class_name='w-100'),
                    ], style={'textAlign': 'center'}),
                ], width=4),
                ], className='mt-2'),
    ]
)


layout = html.Div(
    [
        html.Div([
            html.H1("Genetic Algorithm",
                    className='m-1 mb-3'),
            form,
            html.Div([], id='error_msg'),
            html.Br(),
        ], style={
            'margin': 'auto',
            'width': '90%',
        }),
        html.Br(),
        html.Div([], id="error_msg"),
        html.Br(),
        html.Br(),
        html.Div([
            daq.ToggleSwitch(
                id='btn_toggle',
                value=True,
                label=["Show table", "Hide table"],
                style={'margin': "0", 'padding': "0"},
            ),
        ], style={
            'width': '30%',
            'margin': 'auto',
        }),
        html.Br(),
        html.Div([
            html.Div([

            ], id='population_table'),
        ], id='div_toggle', style={
            'margin': 'auto',
            'width': '90%',
        }),

        html.Br(),
        html.Div([
            dcc.Graph(id="ag_graph"),
        ], style={
            'margin': 'auto',
            'width': '90%',
        }),
        html.Br(),

        html.Div([], id="generate_table", style={
            'margin': 'auto',
            'width': '50%',
        }),
        html.Br(),
        html.Div([
            dbc.Button("Download populations data", outline=True,
                       color="info", id='download_button', n_clicks=0),
            dcc.Download(id="download_populations")
        ], style={
            'textAlign': 'center',
            'margin': 'auto',
        }),
        html.Br(),
        html.Div([
            dbc.Button("Run parameters tests for Genetic Algorithm",
                       outline=True, color="info", id='start_test_button', n_clicks=0),
            html.Br(),
            html.Br(),
            dcc.Loading(children=[
                html.Div([], id="div_test_results", style={
                    'margin': 'auto',
                    'width': '50%'
                }),
                dcc.Download(id="download_top10"),
            ], style={'position': 'relative', 'display': 'flex', 'justify-content': 'center'}),
        ], style={
            'textAlign': 'center',
            'margin': 'auto',
        }),
        html.Br(),
    ])


@callback(Output('div_test_results', 'children'),
          Output('download_top10', 'data'),
          Input('start_test_button', 'n_clicks'),
          prevent_initial_call=True)
def start_test(start_test):
    rdf = test.mini_test()
    top10df = bp.top10_from_df(rdf)
    top10df.reset_index()
    top10df.index = range(1, top10df.shape[0] + 1)
    top10csv = top10df.to_csv(index_label="lp")
    top10df.insert(0, "Lp.", np.arange(1, top10df.shape[0] + 1))
    down_data = dict(content=top10csv, filename="top10.csv")

    return generate_table(top10df), down_data


@callback(Output('div_toggle', 'hidden'),
          Input('btn_toggle', 'value'),
          )
def toggle_table(value):
    return value


@callback(Output('population_table', 'children'),
          Output('a_value', 'value'),
          Output('b_value', 'value'),
          Output('n_value', 'value'),
          Output('ag_graph', 'figure'),
          Output('generate_table', 'children'),
          Output('download_populations', 'data'),
          Output('error_msg', 'children'),
          Input('submit_button', 'n_clicks'),
          Input('download_button', 'n_clicks'),
          State('a_value', 'value'),
          State('b_value', 'value'),
          State('n_value', 'value'),
          State('d_value', 'value'),
          State('pk_value', 'value'),
          State('pm_value', 'value'),
          State('elite_value', 'on'),
          State('t_value', 'value'),
          prevent_initial_call=True)
def update_table(button_submit, button_download, input_a, input_b, input_n, input_d, input_pk, input_pm, input_elite, input_t):
    if None in [input_a, input_b, input_n, input_pk, input_pm, input_t]:
        return no_update, input_a, input_b, input_n, no_update, no_update, no_update, \
            html.Div("Pola wypełniamy wartościami numerycznymi, pk i pm [0:1], N i T [1:200]",
                     style={'color': 'red'})
    elif int(np.ma.round(input_a)) == int(np.ma.round(input_b)):
        return no_update, input_a, input_b, input_n, no_update, no_update, no_update, html.Div(
            "Przedział jest zerowy! Podaj prawidłowy przedział za pomocą liczb całkowitych.",
            style={'color': 'red'})
    elif input_a < -10000000 or input_a > 10000000 or input_b < -10000000 or input_b > 10000000:
        return no_update, input_a, input_b, input_n, no_update, no_update, no_update, html.Div(
            "Przedział jest za duży! Podaj prawidłowy przedział z zakresu [-10M: 10M].",
            style={'color': 'red'})

    if input_a > input_b:
        a = int(np.ma.round(input_b))
        b = int(np.ma.round(input_a))
    else:
        a = int(np.ma.round(input_a))
        b = int(np.ma.round(input_b))

    ctx = dash.callback_context

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
    fx_maxs = []
    fx_mins = []
    fx_avgs = []

    population_txt = ""
    nl = '\n'
    tab = '\t'

    elite_memo = elite.get_best(
        x_reals, [cmp.compute_fx(float(x)) for x in x_reals])

    for i in range(int(input_t)):
        fxs = [cmp.compute_fx(float(x)) for x in x_reals]
        gxs = sel.compute_gxs(fxs, min(fxs), d)
        pxs = sel.compute_pxs(gxs)
        qxs = sel.compute_qxs(pxs)
        rs = sel.compute_r(n)
        sel_reals = sel.get_new_population(rs, qxs, x_reals)
        sel_ints = [cmp.compute_x_int(float(x), length, a, b)
                    for x in sel_reals]
        sel_bins = [cmp.compute_x_bin(x, length) for x in sel_ints]
        parent_bins = cross.get_parents(sel_bins, pk)
        cross_points = cross.get_cross_points(parent_bins, length)
        children_bins = cross.get_children(parent_bins, cross_points)
        children_w_cp = cross.get_children_w_cp(children_bins, cross_points)
        pop_after_cross = cross.get_pop_after_cross(children_bins, sel_bins)
        mutation_indices = [mut.get_mutation_indices(
            length, pm) for _ in range(n)]
        mutation_indices_formatted = [f'{x}' for x in mutation_indices]
        pop_after_mut = mut.mutation(pop_after_cross, mutation_indices)
        x_reals_after_cross_mut = cmp.add_precision(
            cmp.compute_xreals_from_xbins(a, b, length, pop_after_mut), d)
        if input_elite is not False:
            fxs_after_cm = [cmp.compute_fx((float(x)))
                            for x in x_reals_after_cross_mut]
            elite_new = elite.get_best(x_reals_after_cross_mut, fxs_after_cm)
            x_reals_after_cross_mut = elite.inject(
                elite_memo, x_reals_after_cross_mut)
            elite_memo = elite_new
        fxs_cross_mutation = [cmp.compute_fx(
            float(x)) for x in x_reals_after_cross_mut]
        x_reals = x_reals_after_cross_mut
        fx_maxs.append(np.max(fxs_cross_mutation))
        fx_avgs.append(np.average(fxs_cross_mutation))
        fx_mins.append(np.min(fxs_cross_mutation))
        results_zip = [item for item in zip(x_reals, fxs_cross_mutation)]
        population_txt += f'Populacja numer {i+1}{nl}'
        population_txt += f'Lp.{tab} x{tab}{tab}{tab}f(x){nl}'
        for k in range(n):
            population_txt += f'{k+1}{tab}{results_zip[k][0]}{tab}{tab}{results_zip[k][-1]}{nl}'

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

    uniques, counts = np.unique(x_reals, return_counts=True)
    uni_ints = [cmp.compute_x_int(float(x), length, a, b) for x in uniques]
    uni_bins = [cmp.compute_x_bin(x, length) for x in uni_ints]
    uni_fxs = [cmp.compute_fx(float(x)) for x in uniques]

    df_result = pd.DataFrame({
        "x_real": uniques,
        "x_bin": uni_bins,
        "f(x)": uni_fxs,
        "percentage": [f'{float(count) / len(x_reals):.0%}' for count in counts],
    })
    df_result_sorted = df_result.sort_values(by="f(x)", ascending=False)
    df_result_sorted.insert(0, "Lp.", np.arange(1, len(uniques) + 1))

    df_ag = pd.DataFrame({
        "fx_max": fx_maxs,
        "fx_avg": fx_avgs,
        "fx_min": fx_mins,
        "generations": np.arange(1, input_t + 1),
    })

    ag_fig = px.line(df_ag,
                     x="generations",
                     y=["fx_max", "fx_avg", "fx_min"],
                     title="Plot for f_max(x), f_min(x) oraz f_avg(x)",
                     labels={"generations": f'Generations for T={input_t}',
                             "value": "f(x) values"},
                     markers="true")

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'download_button':
        return \
            generate_table(df, max_rows=n), \
            a, b, n, \
            ag_fig, \
            generate_table(df_result_sorted), \
            dict(content=population_txt, filename="population.txt"), \
            ""

    return \
        generate_table(df, max_rows=n), \
        a, b, n, \
        ag_fig, \
        generate_table(df_result_sorted), \
        no_update, \
        ""
