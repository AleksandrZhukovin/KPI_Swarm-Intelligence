import numpy as np


class Optimizer:
    def __init__(self, *, fitness_function, bounds, generations=100):
        self._fitness_function = fitness_function
        self._bounds = np.array(bounds)
        self._generations = generations
        self._dim = len(bounds)

    def _cuckoo_search(self, pop_size):
        delta = 0.5
        p_d = 0.3
        nests = np.random.uniform(self._bounds[:, 0], self._bounds[:, 1], (pop_size, self._dim))
        best_nest = nests[np.random.randint(0, pop_size)]
        best_fitness = self._fitness_function(best_nest)
        best_values = []

        for iteration in range(self._generations):
            for i in range(pop_size):
                step_size = delta * (self._bounds[:, 1] - self._bounds[:, 0]) * np.random.standard_cauchy(self._dim)
                new_solution = np.clip(nests[i] + step_size, self._bounds[:, 0], self._bounds[:, 1])
                new_fitness = self._fitness_function(new_solution)

                if new_fitness < self._fitness_function(nests[i]):
                    nests[i] = new_solution

                if new_fitness < best_fitness:
                    best_fitness = new_fitness

                if np.random.rand() < p_d:
                    nests[i] = np.random.uniform(self._bounds[:, 0], self._bounds[:, 1], self._dim)

            delta *= 0.98

            best_values.append(best_fitness)

        return best_values

    def _bat_algorithm(self, pop_size):
        fmin, fmax = 0, 2
        alpha, gamma = 0.9, 0.9
        loudness = np.random.rand(pop_size)
        pulse_rate = np.random.rand(pop_size)
        population = self._initialize_population(pop_size)
        velocity = np.zeros((pop_size, self._dim))
        best_values = []

        fitness = np.array([self._fitness_function(ind) for ind in population])
        global_best = population[np.argmin(fitness)]

        for _ in range(self._generations):
            fitness = np.array([self._fitness_function(ind) for ind in population])
            best_values.append(fitness.min())

            beta = np.random.rand(pop_size, 1)
            frequency = fmin + (fmax - fmin) * beta
            velocity += (population - global_best) * frequency
            new_population = population + velocity
            new_population = np.clip(new_population, self._bounds[:, 0], self._bounds[:, 1])

            mask = np.random.rand(pop_size) > pulse_rate
            new_population[mask] = global_best + 0.001 * np.random.randn(np.sum(mask), self._dim)

            new_fitness = np.array([self._fitness_function(ind) for ind in new_population])
            improved = new_fitness < fitness
            population[improved] = new_population[improved]
            fitness[improved] = new_fitness[improved]

            if fitness.min() < self._fitness_function(global_best):
                global_best = population[np.argmin(fitness)]
                loudness *= alpha
                pulse_rate = pulse_rate * (1 - np.exp(-gamma))

        return best_values

    def _initialize_population(self, pop_size):
        return np.random.uniform(self._bounds[:, 0], self._bounds[:, 1], (pop_size, self._dim))

    def optimize(self, pop_size, method):
        if method == "kukushka":
            return self._cuckoo_search(pop_size)
        elif method == "bat":
            return self._bat_algorithm(pop_size)
        else:
            raise ValueError("ERROR")
