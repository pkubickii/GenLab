import numpy as np
import pandas as pd
import utils.compute as cmp
import timeit
import math
import os
import json


class Particle:
    def __init__(self, x1=0, x2=0, p=5):
        self.x1 = x1
        self.x2 = x2
        self.p = p
        self.x1 = f'{self.x1:.{self.p}f}'
        self.x2 = f'{self.x2:.{self.p}f}'

    def __str__(self):
        return f'[{self.x1}, {self.x2}]'

    def __repr__(self):
        return f'[{self.x1}, {self.x2}]'

    def get_fx(self):
        return math.pow(float(self.x1) + 2*float(self.x2) - 7, 2) + math.pow(2*float(self.x1) + float(self.x2) - 5, 2)

    def to_list(self, as_float=False):
        if as_float:
            return [float(self.x1), float(self.x2)]
        else:
            return [self.x1, self.x2]

    def move(self, velocity, bounds):
        new_pos = Particle(float(self.x1) + float(velocity.x1),
                           float(self.x2) + float(velocity.x2), self.p)
        if float(new_pos.x1) < bounds[0][0]:
            new_pos.x1 = f'{bounds[0][0]:.{self.p}f}'
        elif float(new_pos.x1) > bounds[0][1]:
            new_pos.x1 = f'{bounds[0][1]:.{self.p}f}'
        if float(new_pos.x2) < bounds[1][0]:
            new_pos.x2 = f'{bounds[1][0]:.{self.p}f}'
        elif float(new_pos.x2) > bounds[1][1]:
            new_pos.x2 = f'{bounds[1][1]:.{self.p}f}'
        return new_pos

    def mul(self, factor: float):
        mullist = [factor * p for p in self.to_list(True)]
        return Particle(mullist[0], mullist[-1])

    def sum(self, particle):
        return Particle(float(self.x1) + float(particle.x1), float(self.x2) + float(particle.x2), self.p)

    def sub(self, particle):
        return Particle(float(self.x1) - float(particle.x1), float(self.x2) - float(particle.x2), self.p)

    def distance(self, particle):
        dist = np.linalg.norm(np.array(self.to_list(
            True)) - np.array(particle.to_list(True)))
        return dist


def generate_particles(a: int, b: int, n: int, p: int) -> Particle:
    x1 = np.random.uniform(a, b, n)
    x2 = np.random.uniform(a, b, n)
    particles = []
    for i in range(n):
        particles.append(Particle(x1[i], x2[i], p))
    return particles


def get_new_velocity(pc, vc, pc_bl, pc_bg, c1, c2, c3):
    r = np.random.uniform(0, 1, 3)

    pc1 = vc.mul(c1 * r[0])
    pc2 = pc_bl.sub(pc).mul(c2 * r[1])
    pc3 = pc_bg.sub(pc).mul(c3 * r[2])
    return pc1.sum(pc2).sum(pc3)


def get_bg_for(p_id: int, swarmdf: pd.DataFrame, vicinity: float):
    df = swarmdf.copy()
    pdf = df.iloc[[p_id]]
    dist = []
    particle = df["particle"][p_id]
    df = df.drop(index=p_id)
    for pc in df["particle"]:
        dist.append(particle.distance(pc))
    df.insert(loc=4, column="dist", value=dist)
    df = df.sort_values("dist")
    how_many = round(vicinity / 100 * df.shape[0])
    nbhood = pd.concat([pdf, df.iloc[:how_many]])
    nbhood = nbhood.sort_values("bg_fxs")
    nbhood = nbhood.reset_index()
    return nbhood["bg"][0]


def particle_swarm(a, b, n, d, time_t, c1, c2, c3, vicinity):
    p = cmp.compute_precision(d)
    particles = generate_particles(a, b, n, p)
    vs = generate_particles(a, b, n, p)
    best_locals = list(particles)
    best_globals = list(particles)
    bounds = [(a, b), (a, b)]
    p_fxs = [particle.get_fx() for particle in particles]
    bl_fxs = list(p_fxs)
    bg_fxs = list(p_fxs)
    df = pd.DataFrame({
        "particle": particles,
        "p_fxs": p_fxs,
        "bl": best_locals,
        "bl_fxs": bl_fxs,
        "bg": best_globals,
        "bg_fxs": bg_fxs,
    })
    results = [df]

    for _ in range(time_t):
        best_globals = [get_bg_for(i, df, vicinity) for i in range(n)]
        bg_fxs = [bg.get_fx() for bg in best_globals]
        new_particles = []
        new_vs = np.zeros(n).tolist()
        for i in range(n):
            if p_fxs[i] < bl_fxs[i]:
                best_locals[i] = particles[i]
            elif p_fxs[i] < bg_fxs[i]:
                best_globals[i] = particles[i]
        for i in range(n):
            new_vs[i] = get_new_velocity(
                particles[i],
                vs[i],
                best_locals[i],
                best_globals[i],
                c1, c2, c3)
            new_particles.append(particles[i].move(new_vs[i], bounds))
        particles = list(new_particles)
        vs = list(new_vs)
        p_fxs = [particle.get_fx() for particle in particles]
        bl_fxs = [particle.get_fx() for particle in best_locals]
        bg_fxs = [particle.get_fx() for particle in best_globals]
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
    result = []
    for fx in fxs:
        if fx > 100:
            result.append(20)
        elif 100 >= fx > 80:
            result.append(19)
        elif 80 >= fx > 70:
            result.append(17)
        elif 70 >= fx > 60:
            result.append(16)
        elif 60 >= fx > 50:
            result.append(15)
        elif 50 >= fx > 40:
            result.append(14)
        elif 40 >= fx > 30:
            result.append(13)
        elif 30 >= fx > 20:
            result.append(12)
        elif 20 >= fx > 10:
            result.append(11)
        elif 10 >= fx > 8:
            result.append(10)
        elif 8 >= fx > 6:
            result.append(9)
        elif 6 >= fx > 4:
            result.append(8)
        elif 4 >= fx > 3:
            result.append(7)
        elif 3 >= fx > 2:
            result.append(6)
        elif 2 >= fx > 1:
            result.append(5)
        elif 1 >= fx > 0.6:
            result.append(4)
        elif 0.6 >= fx > 0.3:
            result.append(3)
        elif 0.3 >= fx > 0.1:
            result.append(2)
        elif fx <= 0.1:
            result.append(1)
    return result


if __name__ == '__main__':
    start = timeit.default_timer()
    a = -10
    b = 10
    d = 10 ** -5
    c1 = 0.8
    c2 = 1.0
    c3 = 1.2
    n = 30
    time_t = 50
    vicinity = 50.0
    p = cmp.compute_precision(d)

    df_list = particle_swarm(a, b, n, d, time_t, c1, c2, c3, vicinity)

    times = np.arange(time_t)
    df = pd.concat(df_list, keys=times)
    df = df.reset_index()
    df.columns = ["time_t", "p_id", "particle",
                  "p_fxs", "bl", "bl_fxs", "bg", "bg_fxs"]

    rdf = df.copy()
    size = get_size(df["p_fxs"])
    df.insert(loc=0, column="size", value=size)
    list = df["particle"].tolist()
    newlist = json.loads(str(list))
    x = []
    y = []
    for i in range(len(newlist)):
        x.append(newlist[i][0])
        y.append(newlist[i][1])

    # print(x)
    # print(y)
    df.insert(loc=1, column="x1", value=x)
    df.insert(loc=2, column="x2", value=y)
    # print(df)
    df.to_csv(f'./results/ps2dresult{n}_{time_t}.csv', index_label="lp")

    # print(df.dtypes)
    rdf = rdf.sort_values(["p_fxs", "time_t"], ascending=[True, True])
    rdf = rdf.reset_index()
    rdf = rdf.drop(columns=["index", "bl",
                   "bl_fxs", "bg", "bg_fxs"])
    rdf.columns = ["period T", "particle id", "particle", "value of fx(min)"]
    df = df.groupby(by="time_t").agg({"p_fxs": ['min', 'mean', 'max']})
    df.columns = df.columns.droplevel(0)
    df.columns = ["fx_min", "fx_avg", "fx_max"]
    # print(rdf)
    # print(df)
    path = './results'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    print(f'Max: ')
    print(rdf.iloc[[0]])

    # df.to_csv(f'./results/psresult{n}_{time_t}.csv', index_label="lp")
    # file = open("./results/ps_time.txt", "w")
    # file.writelines(["Particle Swarm Time:", str(stop-start)])
    # file.close()

    stop = timeit.default_timer()
    print(f'the time: {stop-start}')
