import numpy as np
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use('TkAgg')


class Optimizer:
    def __init__(self, items, capacity, generations=40):
        self.items = items
        self.capacity = capacity
        self.num_items = len(items)
        self.generations = generations

    def _total_value_and_weight(self, selection):
        value = sum(self.items[i][0] for i in range(self.num_items) if selection[i])
        weight = sum(self.items[i][1] for i in range(self.num_items) if selection[i])
        return value, weight

    def _fitness(self, selection):
        value, weight = self._total_value_and_weight(selection)
        if weight > self.capacity:
            return 0
        return value

    def _histogram(self, fitness, generation):
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(fitness)), fitness)
        plt.title(f"Покоління {generation}")
        plt.xlabel("Хромосоми")
        plt.ylabel("Цінність рюкзака")
        plt.tight_layout()
        plt.show()

    def _convergence(self, best_values):
        plt.figure(figsize=(8, 4))
        plt.plot(best_values, marker='o', linestyle='-')
        plt.xlabel("Покоління")
        plt.ylabel("Цінність рюкзака")
        plt.tight_layout()
        plt.show()

    def _gen_algorithm(self, population_size=50):
        population = [np.random.randint(2, size=self.num_items).tolist() for _ in range(population_size)]
        best_values = []

        for gen in range(self.generations):
            fitness = [self._fitness(sel) for sel in population]
            sorted_indices = np.argsort(fitness)[::-1]
            best_fitness = fitness[sorted_indices[0]]
            best_values.append(best_fitness)

            if gen % 5 == 0 or gen == self.generations - 1:
                self._histogram(fitness, gen)

            selected = [population[i] for i in sorted_indices[:population_size // 2]]
            offspring = []

            while len(offspring) < population_size:
                i1, i2 = np.random.choice(len(selected), 2, replace=False)
                p1, p2 = selected[i1], selected[i2]
                cut = np.random.randint(1, self.num_items - 1)
                child1 = p1[:cut] + p2[cut:]
                child2 = p2[:cut] + p1[cut:]
                offspring.append(child1)
                offspring.append(child2)

            population = offspring[:population_size]

            for i in range(population_size):
                if np.random.rand() < 0.1:
                    mutation_point = np.random.randint(self.num_items)
                    population[i][mutation_point] ^= 1

        self._convergence(best_values)
        return best_values

    def optimize(self, method="gen", **kwargs):
        if method == "gen":
            return self._gen_algorithm(**kwargs)
        elif method == "greedy":
            return self._greedy_algorithm()
        else:
            raise ValueError("ERROR")

    def _greedy_algorithm(self):
        ratio_items = sorted(
            [(i, val / wt) for i, (val, wt) in enumerate(self.items) if wt > 0],
            key=lambda x: x[1],
            reverse=True
        )
        total_value, total_weight = 0, 0
        selection = [0] * self.num_items

        for i, _ in ratio_items:
            val, wt = self.items[i]
            if total_weight + wt <= self.capacity:
                selection[i] = 1
                total_value += val
                total_weight += wt

        return total_value, selection
