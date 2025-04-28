import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from main import Optimizer
from fitness_functions import *


matplotlib.use('TkAgg')


data = pd.read_excel('DataRegression.xlsx', sheet_name='Var06')
Y = data.iloc[:, 0].values
X = data.iloc[:, 1].values


indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

split_idx = int(0.75 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
Y_train, Y_val = Y[:split_idx], Y[split_idx:]


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


bounds_model1 = [(100, 500), (0, 1/3)]
bounds_model2 = [(0, 5)] + [(-1/3, 1)] * 4
bounds_model3 = [(0, 5)] + [(-1/3, 1)] * 6


models = [model1, model2, model3]
bounds = [bounds_model1, bounds_model2, bounds_model3]

optimizer = Optimizer()

for i, (model, bnds) in enumerate(zip(models, bounds), 1):
    print(f'--- Model {i} ---')

    def loss_fn(params):
        preds = model(X_train, params)
        return mse(Y_train, preds)

    best_params_de, best_loss_de, de_progress = optimizer.de(loss_fn, bnds)
    val_loss_de = mse(Y_val, model(X_val, best_params_de))
    print(f'DE: {best_params_de}')

    best_params_pso, best_loss_pso, pso_progress = optimizer.de(loss_fn, bnds)
    val_loss_pso = mse(Y_val, model(X_val, best_params_pso))
    print(f'PSO: {best_params_pso}\n')

    plt.plot(de_progress, label='DF')
    plt.plot(pso_progress, label='PSO')
    plt.title(f'Model {i}')
    plt.xlabel('Ітерація')
    plt.ylabel('Середньоквадратична похибка')
    plt.legend()
    plt.grid(True)
    plt.show()

    for dataset_name, X_set, Y_set, params, algo_name in [
        ('Train DE', X_train, Y_train, best_params_de, 'DE'),
        ('Validation DE', X_val, Y_val, best_params_de, 'DE'),
        ('Train PSO', X_train, Y_train, best_params_pso, 'PSO'),
        ('Validation PSO', X_val, Y_val, best_params_pso, 'PSO')
    ]:
        plt.scatter(X_set, Y_set, label='Дані', color='black')
        plt.plot(np.sort(X_set), model(np.sort(X_set), params), label=f'Прогноз', color='red')
        plt.title(f'{dataset_name} - Модель {i}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.show()
