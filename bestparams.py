import pandas as pd
import numpy as np


def get_best_params(csv):
    df = pd.read_csv(csv, index_col=0)
    df_sorted = df.sort_values("favg", ascending=False)
    return df_sorted


if __name__ == '__main__':
    new_df = get_best_params("res_agg.csv")
    print(new_df.iloc[0:10])
    new_df.iloc[0:10].to_csv("results_top10.csv")
