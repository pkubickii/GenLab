import dash_bootstrap_components as dbc
from dash import html
import pandas as pd


def generate_table(dataframe=pd.DataFrame(), max_rows=10):
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
