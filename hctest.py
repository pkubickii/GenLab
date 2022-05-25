import numpy as np
import pandas as pd
from pandas.core.indexes.multi import MultiIndex
import utils.compute as cmp


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


def hill_climbing():
    stop = False
    a = -4
    b = 12
    d = 10 ** -3
    length = cmp.compute_length(a, b, d)
    dot_places = cmp.compute_precision(d)
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
        if vc_fx <= df.iloc[0][1]:
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


if __name__ == '__main__':

    time_t = 3
    dfs = []
    for t in range(time_t):
        df_climbs = hill_climbing()
        dfs.append(df_climbs)
    times = np.arange(len(dfs))
    df = pd.concat(dfs, keys=times)
    print(df)
    # df.index.names = ["one", "two"]
    print(df.index.get_level_values(0))
    df.insert(loc=0, column="period", value=df.index.get_level_values(0))
    print(df)
