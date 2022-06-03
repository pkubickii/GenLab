from os import write
import numpy as np
import pandas as pd
import utils.compute as cmp
import timeit


def get_v(x, v, bl, bg, c1, c2, c3):
    r = np.random.uniform(0, 1, 3)
    return c1 * r[0] * float(v) + c2 * r[1] * (float(bl) - float(x)) + c3 * r[2] * (float(bg) - float(x))


def particle_swarm(a, b, n, d, time_t, c1, c2, c3):
    particles = cmp.add_precision(cmp.generate_population(a, b, n), d)
    vs = cmp.add_precision(cmp.generate_population(a, b, n), d)
    best_locals = list(particles)
    bounds = [f'{a:.3f}', f'{b:.3f}']

    df = pd.DataFrame({
        "particle": particles,
        "fp": [cmp.compute_fx(float(x)) for x in particles],
    })

    fxs_sorted = df.sort_values("fp", ascending=False)
    best_global = particles[fxs_sorted.index[0]]
    results = [df]

    for t in range(time_t):
        p_fxs = [cmp.compute_fx(float(x)) for x in particles]
        bl_fxs = [cmp.compute_fx(float(x)) for x in best_locals]
        bg_fx = cmp.compute_fx(float(best_global))
        new_particles = []
        new_vs = np.zeros(len(particles)).tolist()
        for i in range(len(particles)):
            if p_fxs[i] > bl_fxs[i]:
                best_locals[i] = particles[i]
            elif p_fxs[i] > bg_fx:
                best_global = particles[i]
        for i in range(len(particles)):
            new_vs[i] = get_v(
                particles[i],
                vs[i],
                best_locals[i],
                best_global,
                c1, c2, c3)
            new_particle = float(particles[i]) + new_vs[i]
            if round(new_particle, 3) < round(float(bounds[0]), 3):
                new_particle = float(bounds[0])
            elif round(new_particle, 3) > round(float(bounds[1]), 3):
                new_particle = float(bounds[1])
            new_particles.append(new_particle)
        new_particles = cmp.add_precision(new_particles, d)
        particles = list(new_particles)
        vs = list(new_vs)
        results.append(pd.DataFrame({
            "particle": particles,
            "fp": p_fxs,
        }))
    return results


def get_size(fxs):
    fmin = min(fxs)
    return [int(round(fx - fmin, 1) * 10 + 1) for fx in fxs]


if __name__ == '__main__':
    start = timeit.default_timer()
    a = -4
    b = 12
    d = 10 ** -3
    c1 = 0.8
    c2 = 1.0
    c3 = 1.2
    n = 5
    time_t = 2

    df_list = particle_swarm(a, b, n, d, time_t, c1, c2, c3)
    times = np.arange(time_t)
    df = pd.concat(df_list, keys=times)
    df = df.reset_index()
    df.columns = ["time_t", "p_id", "particle", "fp"]
    print(df["fp"])
    size = get_size(df["fp"])
    print(size)
    df.insert(loc=4, column="size", value=size)
    df.to_csv("./results/psresult.csv")
    # df['particle'] = df['particle'].astype(float)
    stop = timeit.default_timer()
    print(f'the time: {stop-start}')
    # df.to_csv("./results/psall.csv", index_label=["test", "T", "idx"])
    # file = open("./results/hcresults.txt", "w")
    # file.write(str(test_values))
    # file.close()
