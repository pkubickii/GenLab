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


if __name__ == '__main__':
    # Testy
    a = -4
    b = 12
    d = 10 ** -2
    xr1 = 6.47

    prec = compute_precision(d)
    print(f"Prec num: {compute_precision(d)}")
    chlen = compute_length(a, b, d)
    print(f"l= {chlen}")
    xi1 = compute_x_int(xr1, chlen, a, b)
    xi = x_int_no_ceil(xr1, chlen, a, b)
    print(f"x_int= {xi1}")
    print(f"x_int_no_c= {xi}")
    xr2 = compute_x_real(xi1, chlen, a, b)
    print(f"x_real2= {xr2}")
    print(f"xr_prec= {x_real_to_string(xr2, prec)}")
    xb = compute_x_bin(xi1, chlen)
    print(f"x_bin= {xb}")
    xb_arr = x_bin_to_int_array(xb)
    print(xb_arr)
    xi2 = x_int_from_x_bin(xb)
    print(f"x_int2= {xi2}")
    x_reals = generate_population(-4, 12, 20)
    x_reals_str = add_precision(x_reals, d)
    print(x_reals_str)
