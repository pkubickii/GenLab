import numpy as np


def get_randoms(length):
    return np.random.uniform(0, 1, length)


def get_mutation_indices(length, pm=0.005):
    randoms = get_randoms(length)
    mutation_indices = []
    for i in range(length):
        if randoms[i] < pm:
            mutation_indices.append(i)
    return mutation_indices


def mutation(population, mutation_indices):
    pop = np.array(population)
    for i in range(len(pop)):
        pop_list = list(pop[i])
        for x in mutation_indices[i]:
            if pop_list[x] == '1':
                pop_list[x] = '0'
            else:
                pop_list[x] = '1'
        pop[i] = ''.join(pop_list)
    return pop
