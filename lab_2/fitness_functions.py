import numpy as np


__all__ = ('rastrigin', 'mishra_bird', 'reductor',)


def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def mishra_bird(x):
    x1, x2 = x
    return np.sin(x2) * np.exp((1 - np.cos(x1))**2) + \
           np.cos(x1) * np.exp((1 - np.sin(x2))**2) + \
           (x1 - x2)**2


def reductor(x):
    x1, x2, x3, x4, x5, x6, x7 = x
    return (0.7854 * x1 * x2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934)
            - 1.508 * x1 * (x6**2 + x7**2)
            + 7.4777 * (x6**3 + x7**3) + 0.7854 * (x4 * x6**2 + x5 * x7**2))
