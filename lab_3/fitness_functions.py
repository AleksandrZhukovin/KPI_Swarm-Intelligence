import numpy as np


__all__ = ('rastrigin', 'mishra_bird', 'eq1')


def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def mishra_bird(x):
    x1, x2 = x
    return np.sin(x2) * np.exp((1 - np.cos(x1))**2) + \
           np.cos(x1) * np.exp((1 - np.sin(x2))**2) + \
           (x1 - x2)**2
