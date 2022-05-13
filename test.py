import compute as cmp
import selection as sel
import numpy as np
import pandas as pd
import crossover as cross
import mutation as mut
import elite
from multiprocessing import Pool, freeze_support
import timeit
from horology import timed


def run_multiprocessing(func, i, n_proc):
    with Pool(processes=n_proc) as pool:
        return pool.map(func, i)


@timed
def test_func(n_values, pk_values, pm_values, t_values):

    a = -4
    b = 12
    d = 10 ** -3
    length = 14

    ns = []
    pks = []
    pms = []
    ts = []
    fmaxs = []
    favgs = []

    for n in range(len(n_values)):
        for pk in range(len(pk_values)):
            for pm in range(len(pm_values)):
                for t in range(len(t_values)):
                    fxmax_100 = []
                    fxavg_100 = []
                    for omg in range(10):
                        x_reals = cmp.add_precision(cmp.generate_population(a, b, n_values[n]), d)
                        elite_memo = elite.get_best(x_reals, [cmp.compute_fx(float(x)) for x in x_reals])
                        for i in range(t_values[t]):
                            fxs = [cmp.compute_fx(float(x)) for x in x_reals]
                            gxs = sel.compute_gxs(fxs, min(fxs), d)
                            pxs = sel.compute_pxs(gxs)
                            qxs = sel.compute_qxs(pxs)
                            rs = sel.compute_r(n_values[n])
                            sel_reals = sel.get_new_population(rs, qxs, x_reals)
                            sel_ints = [cmp.compute_x_int(float(x), length, a, b) for x in sel_reals]
                            sel_bins = [cmp.compute_x_bin(x, length) for x in sel_ints]
                            parent_bins = cross.get_parents(sel_bins, pk_values[pk])
                            cross_points = cross.get_cross_points(parent_bins, length)
                            children_bins = cross.get_children(parent_bins, cross_points)
                            pop_after_cross = cross.get_pop_after_cross(children_bins, sel_bins)
                            mutation_indices = [mut.get_mutation_indices(length, pm_values[pm]) for _ in
                                                range(n_values[n])]
                            pop_after_mut = mut.mutation(pop_after_cross, mutation_indices)
                            x_reals_after_cross_mut = cmp.add_precision(
                                cmp.compute_xreals_from_xbins(a, b, length, pop_after_mut), d)
                            fxs_after_cm = [cmp.compute_fx((float(x))) for x in x_reals_after_cross_mut]
                            elite_new = elite.get_best(x_reals_after_cross_mut, fxs_after_cm)
                            x_reals_after_cross_mut = elite.inject(elite_memo, x_reals_after_cross_mut)
                            elite_memo = elite_new
                            x_reals = x_reals_after_cross_mut
                        fxs_cross_mutation = [cmp.compute_fx(float(x)) for x in x_reals_after_cross_mut]
                        fxmax_100.append(np.max(fxs_cross_mutation))
                        fxavg_100.append(np.average(fxs_cross_mutation))
                    ns.append(n_values[n])
                    pks.append(pk_values[pk])
                    pms.append(pm_values[pm])
                    ts.append(t_values[t])
                    fmaxs.append(np.average(fxmax_100))
                    favgs.append(np.average(fxavg_100))
                    print(f'N: {n_values[n]} pk: {pk_values[pk]} pm: {pm_values[pm]} T: {t_values[t]} '
                          f'fmax: {fmaxs[-1]} favg: {favgs[-1]}')
    return ns, pks, pms, ts, fmaxs, favgs


if __name__ == '__main__':
    start = timeit.default_timer()
    # test values:
    # n_values = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    # pk_values = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    # pm_values = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    # t_values = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]

    n_values = [30, 35]
    pk_values = [0.5, 0.55]
    pm_values = [0.0001, 0.0005]
    t_values = [50, 60]

    # multiprocessing parameters:
    n_processors = 6

    ns, pks, pms, ts, fmaxs, favgs = test_func(n_values, pk_values, pm_values, t_values)

    stop = timeit.default_timer()
    print(f'Timeit: {stop - start}')
    rdf = pd.DataFrame({
        "n": ns,
        "pk": pks,
        "pm": pms,
        "t": ts,
        "fmax": fmaxs,
        "favg": favgs
    })
    rdf.index = range(1, rdf.shape[0] + 1)
    rdf.to_csv('results_save.csv', index_label="lp")
    file = open("time.txt", "w")
    file.write(str(test_func.interval))
    file.close()
