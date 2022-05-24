import numpy as np


def compute_gxs(fxs, fmin, d):
    return [(fx - fmin + d) for fx in fxs]


def compute_pxs(gxs):
    return [gx/sum(gxs) for gx in gxs]


def compute_qxs(pxs):
    qxs = [pxs[0]]
    for i in range(1, len(pxs)-1):
        qxs.append(qxs[i-1]+pxs[i])
    qxs.append(1.0)
    return qxs


def compute_r(n):
    return np.random.uniform(0, 1, n)


def get_selected_index(r, qxs):
    if r < qxs[0]:
        return 0
    for i in range(1, len(qxs)):
        if qxs[i-1] < r < qxs[i]:
            return i


def get_new_population(randoms, qxs, x_reals):
    return [x_reals[get_selected_index(r, qxs)] for r in randoms]
