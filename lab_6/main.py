import numpy as np
import random


class Optimizer:
    def __init__(self, generations=100):
        self.generations = generations

    def de(self, func, bounds, pop_size=20):
        dim = len(bounds)
        pop = np.random.rand(pop_size, dim)
        for i in range(dim):
            pop[:, i] = bounds[i][0] + pop[:, i] * (bounds[i][1] - bounds[i][0])

        fitness = np.array([func(ind) for ind in pop])
        best_fitness_progress = []

        for gen in range(self.generations):
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + 0.8 * (b - c), [b[0] for b in bounds], [b[1] for b in bounds])
                cross_points = np.random.rand(dim) < 0.9
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                f = func(trial)
                if f < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f
            best_fitness_progress.append(np.min(fitness))

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx], best_fitness_progress

    def pso(self, func, bounds, num_particles=30):
        dim = len(bounds)
        particles = np.random.rand(num_particles, dim)
        for i in range(dim):
            particles[:, i] = bounds[i][0] + particles[:, i] * (bounds[i][1] - bounds[i][0])
        velocities = np.random.uniform(-1, 1, (num_particles, dim))
        pbest = particles.copy()
        pbest_fitness = np.array([func(p) for p in particles])
        gbest_idx = np.argmin(pbest_fitness)
        gbest = pbest[gbest_idx]
        best_fitness_progress = []

        for t in range(self.generations):
            for i in range(num_particles):
                velocities[i] = (0.5 * velocities[i] +
                                 1.5 * random.random() * (pbest[i] - particles[i]) +
                                 1.5 * random.random() * (gbest - particles[i]))
                particles[i] += velocities[i]
                for d in range(dim):
                    if particles[i, d] < bounds[d][0]:
                        particles[i, d] = bounds[d][0]
                    if particles[i, d] > bounds[d][1]:
                        particles[i, d] = bounds[d][1]
                fitness = func(particles[i])
                if fitness < pbest_fitness[i]:
                    pbest[i] = particles[i]
                    pbest_fitness[i] = fitness
                    if fitness < pbest_fitness[gbest_idx]:
                        gbest = particles[i]
                        gbest_idx = i
            best_fitness_progress.append(np.min(pbest_fitness))

        return gbest, pbest_fitness[gbest_idx], best_fitness_progress
