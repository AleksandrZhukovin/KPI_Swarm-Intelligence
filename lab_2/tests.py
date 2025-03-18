import numpy as np
import matplotlib.pyplot as plt

from main import Optimizer
from fitness_functions import *


def plot_results(best_values, generations, title):
    plt.figure(figsize=(12, 5))

    plt.plot(range(generations), [np.min(vals) for vals in best_values])
    plt.title(title)
    plt.xlabel("покоління")
    plt.ylabel("f(X)")
    plt.show()


dimensions = [2, 5, 10, 15]
pop_size = 50
generations = 100
optimization_methods = ["pso", "bee", "firefly"]

# for n in dimensions:
#     # if n == 2:
#     #     optimizer = Optimizer(fitness_function=mishra_bird, bounds=[(-10, 0), (-6.5, 0)])
#     #     for method in optimization_methods:
#     #         best_values, min_distances = optimizer.optimize(pop_size, method=method)
#     #         plot_results(best_values, min_distances, generations, f"mishra_bird {method.upper()}")
#
#     optimizer = Optimizer(fitness_function=rastrigin, bounds=[(-5.12, 5.12)] * n)
#
#     for method in optimization_methods:
#         best_values, min_distances = optimizer.optimize(pop_size, method=method)
#         plot_results(best_values, generations, f"rastrigin {n}-D {method.upper()}")

reductor_bond = [
    (2.6, 3.6), (0.7, 0.8), (17, 28),
    (7.3, 8.3), (7.8, 8.3), (2.9, 3.9), (5.0, 5.5)
]

optimizer = Optimizer(fitness_function=reductor, bounds=reductor_bond)
for method in optimization_methods:
    best_values = optimizer.optimize(pop_size, method=method)
    plot_results(best_values, generations, f"Редуктор {method}")
