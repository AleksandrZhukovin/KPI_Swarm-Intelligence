import numpy as np


class Optimizer:
    def __init__(self, distance_matrix, generations=100):
        self.graph = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.generations = generations

    def _calculate_total_distance(self, tour):
        return sum(self.graph[tour[i], tour[i+1]] for i in range(len(tour) - 1)) + self.graph[tour[-1], tour[0]]

    def _ant_algorithm(self, alpha=1.0, beta=5.0, rho=0.5, q=100):
        pheromone = np.ones((self.num_cities, self.num_cities))
        visibility = 1 / (self.graph + np.eye(self.num_cities))
        np.fill_diagonal(visibility, 0)
        best_length = float("inf")
        best_tour = None
        best_values = []

        for gen in range(self.generations):
            all_tours = []
            all_lengths = []

            for _ in range(50):
                tour = [np.random.randint(self.num_cities)]
                while len(tour) < self.num_cities:
                    i = tour[-1]
                    probs = []
                    for j in range(self.num_cities):
                        if j not in tour:
                            prob = (pheromone[i][j] ** alpha) * (visibility[i][j] ** beta)
                        else:
                            prob = 0
                        probs.append(prob)
                    probs = np.array(probs)
                    probs /= probs.sum()
                    next_city = np.random.choice(range(self.num_cities), p=probs)
                    tour.append(next_city)

                length = self._calculate_total_distance(tour)
                all_tours.append(tour)
                all_lengths.append(length)

                if length < best_length:
                    best_length = length
                    best_tour = tour

            pheromone *= (1 - rho)
            for tour, length in zip(all_tours, all_lengths):
                for i in range(self.num_cities):
                    a, b = tour[i], tour[(i + 1) % self.num_cities]
                    pheromone[a][b] += q / length
                    pheromone[b][a] += q / length

            best_values.append(best_length)

        return best_values, best_tour

    def _gen_algorithm(self, population_size=50):
        population = [np.random.permutation(self.num_cities).tolist() for _ in range(population_size)]
        best_values = []

        for gen in range(50):
            fitness = [1 / self._calculate_total_distance(tour) for tour in population]
            sorted_indices = np.argsort(fitness)[::-1]
            best_values.append(1 / fitness[sorted_indices[0]])

            selected = [population[i] for i in sorted_indices[:population_size // 2]]
            offspring = []

            while len(offspring) < population_size:
                i1, i2 = np.random.choice(len(selected), 2, replace=False)
                p1, p2 = selected[i1], selected[i2]

                cut = np.random.randint(1, self.num_cities - 1)
                child1 = p1[:cut] + [city for city in p2 if city not in p1[:cut]]
                child2 = p2[:cut] + [city for city in p1 if city not in p2[:cut]]

                offspring.append(child1)
                offspring.append(child2)

            population = offspring[:population_size]

            for i in range(population_size):
                if np.random.rand() < 0.01:
                    a, b = np.random.choice(self.num_cities, 2, replace=False)
                    population[i][a], population[i][b] = population[i][b], population[i][a]

        return best_values

    def optimize(self, method, **kwargs):
        if method == "ant":
            return self._ant_algorithm(**kwargs)
        elif method == "gen":
            return self._gen_algorithm(**kwargs)
        else:
            raise ValueError("Error")
