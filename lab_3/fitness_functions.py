import numpy as np


__all__ = ('rastrigin', 'mishra_bird', 'eq1')


def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def mishra_bird(x):
    x1, x2 = x
    return np.sin(x2) * np.exp((1 - np.cos(x1))**2) + \
           np.cos(x1) * np.exp((1 - np.sin(x2))**2) + \
           (x1 - x2)**2


def eq1(params):
    x_prime, x, y, t = params
    eq1 = x_prime**2 + y**2 * np.cos(x_prime) - x**2 * t**2
    eq2 = y**4 + x**3 - 3 * np.sin(t * x_prime) - x_prime
    error = eq1**2 + eq2**2
    return error
