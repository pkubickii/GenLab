import numpy as np
import compute as cmp


def get_best(x_reals, fxs):
    return x_reals[np.argmax(fxs)], np.max(fxs)


def check_and_swap(elite_x_fx, xreals):
    if elite_x_fx[-1] < cmp.compute_fx(float(xreals[0])):
        xreals[0] = elite_x_fx[0]
    return xreals


def inject(elite, xreals):
    random_index = np.random.randint(0, len(xreals))
    xreals[random_index] = elite[0]
    return xreals
