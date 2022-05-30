import numpy as np
import pandas as pd
from pandas.core.indexes.multi import MultiIndex
import utils.compute as cmp
import timeit
import json


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


if __name__ == '__main__':
    start = timeit.default_timer()
    a = -4
    b = 12
    d = 10 ** -3
    length = cmp.compute_length(a, b, d)
    dot_places = cmp.compute_precision(d)
    test = 10
    time_t = 100
    df_1000 = []
    for i in range(test):
        df_climbs = []
        for t in range(time_t):
            df_climbs.append(hill_climbing(a, b, d, length, dot_places))
            times = np.arange(len(df_climbs))
        df_1000.append(pd.concat(df_climbs, keys=times))
    tests = np.arange(test)
    df = pd.concat(df_1000, keys=tests)

    test_values = np.zeros(time_t).tolist()
    for i in range(test):
        df_one = df.loc[i]["vcr_climb"]
        if len(df_one[df_one == "10.999"]) < 1:
            continue
        test_values[df_one[df_one == "10.999"].index[0][0]] += 1
    print(test_values)

    stop = timeit.default_timer()
    print(f'the time: {stop-start}')
    df.to_csv("./results/hcall.csv", index_label=["test", "T", "idx"])
    file = open("./results/hcresults.txt", "w")
    file.write(str(test_values))
    file.close()
