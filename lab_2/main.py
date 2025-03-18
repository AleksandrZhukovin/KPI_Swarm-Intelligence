import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation


matplotlib.use('TkAgg')


class Optimizer:
    def __init__(self, *, fitness_function, bounds):
        self._fitness_function = fitness_function
        self._bounds = np.array(bounds)
        self._generations = 100
        self._dim = len(bounds)

    def _particle_swarm(self, pop_size):
        w = 0.5
        a1, a2 = 1.5, 1.5
        population = self._initialize_population(pop_size)
        velocity = np.random.uniform(-1, 1, (pop_size, self._dim))
        personal_best = population.copy()
        global_best = population[np.argmin([self._fitness_function(ind) for ind in population])]
        best_values = []

        for _ in range(self._generations):
            fitness = np.array([self._fitness_function(ind) for ind in population])
            best_values.append(fitness.copy())

            better_mask = fitness < np.array([self._fitness_function(ind) for ind in personal_best])
            personal_best[better_mask] = population[better_mask]

            if fitness.min() < self._fitness_function(global_best):
                global_best = population[np.argmin(fitness)]

            r1, r2 = np.random.rand(pop_size, self._dim), np.random.rand(pop_size, self._dim)
            velocity = w * velocity + a1 * r1 * (personal_best - population) + a2 * r2 * (global_best - population)
            population += velocity
            population = np.clip(population, self._bounds[:, 0], self._bounds[:, 1])

        # if self._dim == 2:
        #     self._visualize_optimization(best_values, 'pso')

        return best_values

    def _bee_algorithm(self, pop_size):
        population = self._initialize_population(pop_size)
        best_values = []

        for _ in range(self._generations):
            fitness = np.array([self._fitness_function(ind) for ind in population])
            best_values.append(fitness.copy())

            elite = population[np.argsort(fitness)[:pop_size // 5]]
            new_population = elite.copy()

            for _ in range(pop_size - len(elite)):
                bee = elite[np.random.randint(len(elite))] + np.random.uniform(-0.5, 0.5, self._dim)
                new_population = np.vstack((new_population, bee))

            population = np.clip(new_population, self._bounds[:, 0], self._bounds[:, 1])

        # if self._dim == 2:
        #     self._visualize_optimization(best_values, 'bee')

        return best_values

    def _firefly_algorithm(self, pop_size):
        alpha, beta0, gamma = 0.5, 1.0, 0.1
        population = self._initialize_population(pop_size)
        best_values = []

        for _ in range(self._generations):
            fitness = np.array([self._fitness_function(ind) for ind in population])
            best_values.append(fitness.copy())

            new_population = population.copy()
            for i in range(pop_size):
                for j in range(pop_size):
                    if fitness[j] < fitness[i]:
                        r = np.linalg.norm(population[i] - population[j])
                        beta = beta0 * np.exp(-gamma * r**2)
                        new_population[i] += beta * (population[j] - population[i]) + alpha * (np.random.rand(self._dim) - 0.5)

            population = np.clip(new_population, self._bounds[:, 0], self._bounds[:, 1])

        # if self._dim == 2:
        #     self._visualize_optimization(best_values, 'firefly')

        return best_values

    def _initialize_population(self, pop_size):
        return np.random.uniform(self._bounds[:, 0], self._bounds[:, 1], (pop_size, self._dim))

    def optimize(self, pop_size, method="pso"):
        if method == "pso":
            return self._particle_swarm(pop_size)
        elif method == "bee":
            return self._bee_algorithm(pop_size)
        elif method == "firefly":
            return self._firefly_algorithm(pop_size)
        else:
            raise ValueError("ERROR")

    def _visualize_optimization(self, history, method_name):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = np.linspace(self._bounds[0, 0], self._bounds[0, 1], 100)
        y = np.linspace(self._bounds[1, 0], self._bounds[1, 1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[self._fitness_function(np.array([x, y])) for x, y in zip(X_row, Y_row)] for X_row, Y_row in zip(X, Y)])

        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
        scatter = ax.scatter([], [], [], color='red', s=10)

        def update(frame):
            points = history[frame]
            scatter._offsets3d = (points[:, 0], points[:, 1], [self._fitness_function(p) for p in points])
            return scatter,

        ani = animation.FuncAnimation(fig, update, frames=len(history), interval=100, blit=False)
        ani.save(f"{method_name}_{self._fitness_function.__name__}.gif", writer='pillow', fps=10)
        plt.show()
