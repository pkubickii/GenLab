import numpy as np
import compute as cmp


def get_best(x_reals, fxs):
    return (x_reals[np.argmin(fxs)], np.min(fxs))


def check_and_swap(elite_x_fx, xreals):
    if elite_x_fx[-1] < cmp.compute_fx(float(xreals[0])):
        xreals[0] = elite_x_fx[0]
    return xreals
