import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
# import imageio

from utils import dec2bin

matplotlib.use('TkAgg')


def f(x):
    return x ** 3 * (3 - x) ** 5 * np.sin(10 * np.pi * x)


def genetic_algorithm():
    population = [''.join(random.choice('01') for _ in range(16)) for _ in range(50)]

    x_values = np.linspace(0, 3, 1000)
    y_values = f(x_values)
    # frames = []

    fig, ax = plt.subplots(figsize=(10, 5))

    for gen in range(50):
        decoded_population = np.array([dec2bin(chromo) for chromo in population])
        fitness_values = np.array([f(x) for x in decoded_population])
        selected = [population[i] for i in np.argsort(fitness_values)[:50 // 2]]
        offspring = []

        while len(offspring) < 50:
            p1, p2 = random.sample(selected, 2)
            crossover_point = random.randint(1, 15)
            child1 = p1[:crossover_point] + p2[crossover_point:]
            child2 = p2[:crossover_point] + p1[crossover_point:]
            offspring.extend([child1, child2])

        population = offspring[:50]

        for i in range(len(population)):
            if random.random() < 0.01:
                mutation_point = random.randint(0, 15)
                population[i] = population[i][:mutation_point] + (
                    '1' if population[i][mutation_point] == '0' else '0') + population[i][mutation_point + 1:]

        ax.clear()
        ax.plot(x_values, y_values)
        ax.scatter(decoded_population, fitness_values, label=f"{gen + 1}")
        ax.legend()
        plt.draw()

    #     fig.canvas.draw()
    #     image = np.array(fig.canvas.renderer.buffer_rgba())
    #     frames.append(image)
    #
    # imageio.mimsave("genetic.gif", frames, fps=5)

    best = min(population, key=lambda chromo: f(dec2bin(chromo)))
    return dec2bin(best), f(dec2bin(best))


def gwo():
    positions = np.random.uniform(0, 3, 50)
    lead = min(positions, key=f)

    x_values = np.linspace(0, 3, 1000)
    y_values = f(x_values)
    # frames = []

    fig, ax = plt.subplots(figsize=(10, 5))

    for iter_num in range(50):
        for i in range(50):
            pos = lead - np.random.uniform(-1, 1) * abs(lead - positions[i])
            positions[i] = np.clip(pos, -5.12, 5.12)

        lead = min(positions, key=f)

        ax.clear()
        ax.plot(x_values, y_values)
        ax.scatter(positions, [f(x) for x in positions], label=f"{iter_num + 1}")
        ax.legend()
        plt.draw()

    #     fig.canvas.draw()
    #     image = np.array(fig.canvas.renderer.buffer_rgba())
    #     frames.append(image)
    #
    # imageio.mimsave("gwo.gif", frames, fps=5)

    return lead, f(lead)


res_g, f_g = genetic_algorithm()
res_gwo, f_gwo = gwo()

print(f"Генетичний: x = {res_g}, f = {f_g}")
print(f"GWO: x = {res_gwo}, f = {f_gwo}")

plt.show()
