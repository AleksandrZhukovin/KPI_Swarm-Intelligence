import numpy as np

from main import Optimizer


def generate_knapsack_problem():
    values = np.random.randint(1, 100, size=20)
    weights = np.random.randint(1, 50, size=20)
    items = list(zip(values, weights))
    capacity = int(np.sum(weights) * 0.5)
    return items, capacity


items, capacity = generate_knapsack_problem()
optimizer = Optimizer(items, capacity)

best_values = optimizer.optimize(method="gen", population_size=100)
print("Gen:", max(best_values))

value, selection = optimizer.optimize(method="greedy")
print("Greedy:", value)
