import numpy as np
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use('TkAgg')


def f(X):
    return 10 * len(X) + sum([(x ** 2 - 10 * np.cos(2 * np.pi * x)) for x in X])


def genetic():
    population = np.random.uniform(-5.12, 5.12, (50, 3))
    best_values = []
    min_distances = []

    for gen in range(50):
        fitness = np.array([f(ind) for ind in population])
        sorted_indices = np.argsort(fitness)
        best_values.append(fitness.copy())
        min_distances.append(abs(fitness[sorted_indices[0]] - 0))

        selected = population[sorted_indices[:50 // 2]]
        offspring = []

        while len(offspring) < 50:
            p1, p2 = selected[np.random.choice(len(selected), 2, replace=False)]
            crossover_point = np.random.randint(1, 3)
            child1 = np.concatenate((p1[:crossover_point], p2[crossover_point:]))
            child2 = np.concatenate((p2[:crossover_point], p1[crossover_point:]))
            offspring.extend([child1, child2])

        population = np.array(offspring[:50])
        mutation = np.random.rand(50, 3) < 0.01
        population += mutation * np.random.uniform(-0.5, 0.5, (50, 3))
        population = np.clip(population, -5.12, 5.12)

    return best_values, min_distances


def gwo():
    positions = np.random.uniform(-5.12, 5.12, (50, 3))
    best_values = []
    min_distances = []

    for iter_num in range(50):
        fitness = np.array([f(pos) for pos in positions])
        sorted_indices = np.argsort(fitness)
        best_values.append(fitness.copy())
        min_distances.append(abs(fitness[sorted_indices[0]]))

        lead = positions[sorted_indices[0]]

        for i in range(50):
            pos = lead - np.random.uniform(-1, 1) * abs(lead - positions[i])
            positions[i] = np.clip(pos, -5.12, 5.12)

    return best_values, min_distances


res_g, min_distances_g = genetic()
res_gwo, min_distances_gwo = gwo()


fig, axes = plt.subplots(2, 2, figsize=(10, 6))

axes[0, 0].plot(range(50), [np.mean(values) for values in res_g], label="Середнє")
axes[0, 0].plot(range(50), [np.min(values) for values in res_g], label="Найкраще")
axes[0, 0].set_title("Пристосованість (ген)")
axes[0, 0].legend()

axes[1, 0].plot(range(50), min_distances_g)
axes[1, 0].set_title("Відстань (ген)")

axes[0, 1].plot(range(50), [np.mean(values) for values in res_gwo], label="Середнє")
axes[0, 1].plot(range(50), [np.min(values) for values in res_gwo], label="Найкраще")
axes[0, 1].set_title("Пристосованість (GWO)")
axes[0, 1].legend()

axes[1, 1].plot(range(50), min_distances_gwo)
axes[1, 1].set_title("Відстань (GWO)")

plt.tight_layout()
plt.show()
