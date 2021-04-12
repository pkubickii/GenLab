import math
import numpy as np


def compute_length(a, b, d):
    return math.ceil(math.log((b - a) * (1 / d) + 1, 2))


def compute_x_int(x_real, length, a, b):
    return int(np.ma.round(((1 / (b - a)) * (x_real - a) * (2 ** length - 1))))


def x_int_no_ceil(x_real, length, a, b):
    return (1 / (b - a)) * (x_real - a) * (2 ** length - 1)


def compute_x_real(x_int, length, a, b):
    return a + (((b - a) * x_int) / (2 ** length - 1))


def x_real_to_string(x_real, precision):
    return f'{x_real:.{precision}f}'


def compute_precision(d):
    return -math.floor(math.log(d, 10))


def compute_x_bin(x_int, length):
    return f'{x_int:0{length}b}'


def x_bin_to_int_array(x_bin):
    return [int(x) for x in list(x_bin)]


def x_int_from_x_bin(x_bin):
    return int(x_bin, 2)


def generate_population(a, b, n):
    return np.random.uniform(a, b, n)


def add_precision(population, d):
    p = compute_precision(d)
    return [f'{left_pad(x)}{x:.{p}f}' for x in population]


def left_pad(x):
    if x >= 0:
        return " "
    else:
        return ""


def compute_mod(x):
    return float(f'{math.fmod(x, 1):.3f}')


def compute_fx(x):
    return float(f'{math.fmod(x, 1):.3f}') * (math.cos(20 * math.pi * x) - math.sin(x))


def compute_xreals_from_xbins(a, b, length, x_bins):
    x_ints = [x_int_from_x_bin(x_bin) for x_bin in x_bins]
    return [compute_x_real(x_int, length, a, b) for x_int in x_ints]
