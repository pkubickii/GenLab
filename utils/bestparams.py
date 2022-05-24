import pandas as pd


def top10_from_csv(csv):
    df = pd.read_csv(csv, index_col=0)
    df_sorted = df.sort_values("favg", ascending=False)
    df_sorted.iloc[0:10].to_csv("./results/results_top10.csv")
    return df_sorted.iloc[0:10]


def top10_from_df(df):
    return df.sort_values("favg", ascending=False).iloc[0:10]


if __name__ == '__main__':
    new_df = top10_from_csv("./results/res_agg.csv")
    print(new_df)
