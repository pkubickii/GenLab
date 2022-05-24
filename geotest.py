import numpy as np
import pandas as pd
import utils.compute as cmp


def get_start_vbs(vb):
    start_vbs = []
    for i in range(len(vb)):
        vb_temp = list(vb)
        if vb_temp[i] == '0':
            vb_temp[i] = '1'
        else:
            vb_temp[i] = '0'
        start_vbs.append(''.join(vb_temp))
    return start_vbs


def get_ranks(tau, length):
    ranks = np.zeros(length)
    for r in range(length):
        ranks[r] = (r + 1) ** -tau
    return ranks


def get_mut_indices(ranked_index, ranks):
    randoms = np.random.uniform(0, 1, len(ranked_index))
    mutation_indices = []
    for i in range(len(ranked_index)):
        if randoms[i] < ranks[i]:
            mutation_indices.append(ranked_index[i])
    return mutation_indices


def get_mutated_vb(vb, mut_indices):
    vb_temp = list(vb)
    for mi in mut_indices:
        if vb_temp[mi] == '0':
            vb_temp[mi] = '1'
        else:
            vb_temp[mi] = '0'
    return ''.join(vb_temp)


def vbs_best_in_t(vbs_fx):
    vbs_best = []
    vbs_best.append(vbs_fx[0])
    for i in range(1, len(vbs_fx)):
        if vbs_fx[i] > vbs_best[i - 1]:
            vbs_best.append(vbs_fx[i])
        else:
            vbs_best.append(vbs_best[i-1])
    return vbs_best


def geo(vb, time_t, a, b, length, d, tau):
    vbs_best = []
    ranks = get_ranks(tau, length)
    for t in range(time_t):
        start_vbs = get_start_vbs(vb)
        start_rvbs = cmp.add_precision(
            cmp.compute_xreals_from_xbins(a, b, length, start_vbs), d)
        start_fxs = [cmp.compute_fx(float(x)) for x in start_rvbs]
        df = pd.DataFrame(
            {
                "fx": start_fxs,
            })
        df = df.sort_values("fx", ascending=False)
        ranked_index = df.index.to_list()
        mutation_indices = get_mut_indices(ranked_index, ranks)
        vb = get_mutated_vb(vb, mutation_indices)
        vbs_best.append(vb)
    return vbs_best


if __name__ == '__main__':
    a = - 4
    b = 12
    d = 10 ** -3
    time_t = 5
    taus = np.around(np.arange(0.5, 3.1, 0.1), 1)
    length = cmp.compute_length(a, b, d)
    dot_places = cmp.compute_precision(d)
    rand_float = np.random.uniform(a, b)
    vb_real = f'{rand_float:.{dot_places}f}'
    vb_int = cmp.compute_x_int(float(vb_real), length, a, b)
    vb = cmp.compute_x_bin(vb_int, length)
    results_fxs = []
    results_tau = []
    for tau in taus:
        for i in range(1000):
            vbs = geo(vb, time_t, a, b, length, d, tau)
            vbrs = cmp.add_precision(
                cmp.compute_xreals_from_xbins(a, b, length, vbs), d)
            fxs = [cmp.compute_fx(float(x)) for x in vbrs]
            df = pd.DataFrame({"fxs": fxs})
            df = df.sort_values("fxs", ascending=False)
            index_best = df.index.to_list()[0]
            results_fxs.append(fxs[index_best])
            results_tau.append(tau)
            print(fxs[index_best])
    rdf = pd.DataFrame({
        "tau": results_tau,
        "fxs": results_fxs,
    })
    print(rdf)
    rdf = rdf.groupby(by="tau").agg({"fxs": [np.average]})
    rdf.columns = rdf.columns.droplevel(1)
    print(rdf)
    rdf.to_csv('./results/geotest.csv', index_label="tau")
