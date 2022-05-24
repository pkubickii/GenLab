import numpy as np


def get_randoms(n):
    return np.random.uniform(0, 1, n)


def get_parents(x_bins, pk=0.75):
    randoms = get_randoms(len(x_bins))
    parents = []
    for i in range(0, len(x_bins)):
        if randoms[i] < pk:
            parents.append(x_bins[i])
        else:
            parents.append(None)
    return make_even_parents(parents)


def make_even_parents(parents):
    k = 0
    tmp_parents = []
    for i in range(len(parents)):
        if parents[i] is not None:
            k += 1
            tmp_parents.append(parents[i])

    if k % 2 != 0:
        for i in range(len(parents)):
            if parents[i] is None:
                parents[i] = tmp_parents[np.random.randint(len(tmp_parents))]
                break
    return parents


def get_cross_points(parents, no_bits):
    cross_points = [None for _ in range(len(parents))]
    for i in range(len(parents)):
        pc = np.random.randint(1, no_bits)
        if parents[i] is not None and cross_points[i] is None:
            cross_points[i] = pc
            for j in range(i + 1, len(parents)):
                if parents[j] is not None and cross_points[j] is None:
                    cross_points[j] = pc
                    break
    return cross_points


def get_children(parents, cross_points):
    children = [None for _ in range(len(parents))]
    i = 0
    while i < len(parents):
        if parents[i] is not None:
            prefix1 = parents[i][:cross_points[i]]
            suffix1 = parents[i][cross_points[i]:]
            for j in range(i + 1, len(parents)):
                if parents[j] is not None:
                    prefix2 = parents[j][:cross_points[j]]
                    suffix2 = parents[j][cross_points[j]:]
                    children[i] = f'{prefix1}{suffix2}'
                    children[j] = f'{prefix2}{suffix1}'
                    i = j
                    break
        i += 1
    return children


def get_children_w_cp(children, cross_points):
    children_w_cp = [None for _ in range(len(children))]
    for i in range(len(children)):
        if children[i] is not None:
            children_w_cp[i] = f'{children[i][:cross_points[i]]} | {children[i][cross_points[i]:]}'
    return children_w_cp


def get_pop_after_cross(children, x_bins):
    for i in range(len(children)):
        if children[i] is None:
            children[i] = x_bins[i]
    return children
