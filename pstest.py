import numpy as np
import pandas as pd
import utils.compute as cmp
import timeit
import os


def get_v(x, v, bl, bg, c1, c2, c3):
    r = np.random.uniform(0, 1, 3)
    return c1 * r[0] * float(v) + c2 * r[1] * (float(bl) - float(x)) + c3 * r[2] * (float(bg) - float(x))


def get_bg_for(p_id: int, particles: pd.DataFrame, vicinity: float):
    df = particles.copy()
    pdf = df.iloc[[p_id]]
    dist = []
    p = float(df["particle"][p_id])
    df = df.drop(index=p_id)
    for x in df["particle"]:
        dist.append(abs(p - float(x)))
    df.insert(loc=4, column="dist", value=dist)
    df = df.sort_values("dist")
    how_many = round(vicinity / 100 * df.shape[0])
    nbhood = pd.concat([pdf, df.iloc[:how_many]])
    nbhood = nbhood.sort_values("bg_fxs", ascending=False)
    nbhood = nbhood.reset_index()
    return nbhood["bg"][0]


def particle_swarm(a, b, n, d, time_t, c1, c2, c3, vicinity):
    dot_places = cmp.compute_precision(d)
    particles = cmp.add_precision(cmp.generate_population(a, b, n), d)
    vs = cmp.add_precision(cmp.generate_population(a, b, n), d)
    best_locals = list(particles)
    best_globals = list(particles)
    bounds = [f'{a:.{dot_places}f}', f'{b:.{dot_places}f}']
    p_fxs = [cmp.compute_fx(float(x)) for x in particles]
    bl_fxs = [cmp.compute_fx(float(x)) for x in best_locals]
    bg_fxs = [cmp.compute_fx(float(x)) for x in best_globals]
    df = pd.DataFrame({
        "particle": particles,
        "p_fxs": p_fxs,
        "bl": best_locals,
        "bl_fxs": bl_fxs,
        "bg": best_globals,
        "bg_fxs": bg_fxs,
    })
    results = [df]

    for t in range(time_t):
        best_globals = [get_bg_for(i, df, vicinity) for i in range(n)]
        bg_fxs = [cmp.compute_fx(float(x)) for x in best_globals]
        new_particles = []
        new_vs = np.zeros(n).tolist()
        for i in range(n):
            if p_fxs[i] > bl_fxs[i]:
                best_locals[i] = particles[i]
            elif p_fxs[i] > bg_fxs[i]:
                best_globals[i] = particles[i]
        for i in range(n):
            new_vs[i] = get_v(
                particles[i],
                vs[i],
                best_locals[i],
                best_globals[i],
                c1, c2, c3)
            new_particle = float(particles[i]) + new_vs[i]
            if round(new_particle, dot_places) < round(float(bounds[0]), dot_places):
                new_particle = float(bounds[0])
            elif round(new_particle, dot_places) > round(float(bounds[1]), dot_places):
                new_particle = float(bounds[1])
            new_particles.append(new_particle)
        new_particles = cmp.add_precision(new_particles, d)
        particles = list(new_particles)
        vs = list(new_vs)
        p_fxs = [cmp.compute_fx(float(x)) for x in particles]
        bl_fxs = [cmp.compute_fx(float(x)) for x in best_locals]
        bg_fxs = [cmp.compute_fx(float(x)) for x in best_globals]
        df = pd.DataFrame({
            "particle": particles,
            "p_fxs": p_fxs,
            "bl": best_locals,
            "bl_fxs": bl_fxs,
            "bg": best_globals,
            "bg_fxs": bg_fxs,
        })
        results.append(df)
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
    time_t = 3
    vicinity = 50.0

    df_list = particle_swarm(a, b, n, d, time_t, c1, c2, c3, vicinity)

    zeros = np.zeros(n * time_t)
    times = np.arange(time_t)
    df = pd.concat(df_list, keys=times)
    df = df.reset_index()
    df.columns = ["time_t", "p_id", "particle",
                  "p_fxs", "bl", "bl_fxs", "bg", "bg_fxs"]
    df.insert(loc=0, column="zeros", value=zeros)
    size = get_size(df["p_fxs"])
    df.insert(loc=9, column="size", value=size)
    path = './results'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    df = df.drop(columns=["zeros", "size"])
    rdf = df.copy()
    rdf = rdf.sort_values("p_fxs", ascending=False)
    rdf = rdf.reset_index()
    print(f'Max: ')
    print(rdf.iloc[[0]])
    df = df.groupby(by="time_t").agg({"p_fxs": ['max', 'mean', 'min']})
    df.columns = df.columns.droplevel(0)
    df.columns = ["fx_max", "fx_avg", "fx_min"]
    df.to_csv(f'./results/psresult{n}_{time_t}.csv', index_label="lp")
    stop = timeit.default_timer()
    print(f'the time: {stop-start}')
    file = open("./results/ps_time.txt", "w")
    file.writelines(["Particle Swarm Time:", str(stop-start)])
    file.close()
