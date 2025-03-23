import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.integrate import solve_ivp

from fitness_functions import *
from main import Optimizer


matplotlib.use('TkAgg')


# def test1():
#     o = Optimizer(fitness_function=rastrigin, bounds=[(-5.12, 5.12)] * 20)
#
#     kukushka_best_values = o.optimize(500, 'kukushka')
#     bat_best_values = o.optimize(500, 'bat')
#
#     plt.plot(range(500), kukushka_best_values, label='Кукушка')
#     plt.plot(range(500), bat_best_values, label='Кажан')
#     plt.xlabel("Покоління")
#     plt.ylabel("f(x)")
#     plt.legend()
#     plt.grid()
#     plt.show()
#
#
# def test2():
#     optimizer = Optimizer(fitness_function=eq1, bounds=[
#         (-10, 10),
#         (-10, 10),
#         (-10, 10),
#         (1, 3)
#     ], generations=500)
#
#     k_res = optimizer.optimize(pop_size=50, method="kukushka")
#     b_res = optimizer.optimize(pop_size=50, method="bat")
#     print("Кукушка:", min(k_res))
#     print("Кажан:", min(b_res))
#
#
# test1()


def dif(alpha, target_b, target_B):
    def equations(t, state):
        x, x_dot, y = state
        dx_dt = x_dot
        dx_dot_dt = x ** 2 * t ** 2 - (np.sqrt(y) * np.cos(x_dot))
        dy_dt = y ** 4 + x ** 3 - 3 * np.sin(t) * x_dot
        return [dx_dt, dx_dot_dt, dy_dt]

    initial_state = np.array([1, alpha[0], 1])
    t_span = [1, target_b]
    solution = solve_ivp(equations, t_span, initial_state, dense_output=True)
    x_at_b = solution.y[0, -1]
    return (x_at_b - target_B) ** 2


def test_dif():
    bounds = [(0, 5)]
    optimizer = Optimizer(
        fitness_function=lambda alpha: dif(
            alpha, 3, 2
        ),
        bounds=bounds,
    )

    best_values_kukushka = optimizer.optimize(pop_size=10, method="kukushka")
    best_values_bat = optimizer.optimize(pop_size=10, method="bat")

    plt.plot(best_values_kukushka, label="Кукушка")
    plt.plot(best_values_bat, label="Кажан")
    plt.xlabel("Покоління")
    plt.ylabel("Помилка")
    plt.legend()
    plt.show()


test_dif()
