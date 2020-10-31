import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from activation_functions import BinaryStep, SignFunction
from perceptron import Perceptron

dataset = pd.read_csv('database/or.csv')
X = dataset.iloc[:, 0:2].values
d = dataset.iloc[:, 2:].values

#dataset = pd.read_csv('database/iris.data', header=None)
#dataset[4].replace({"Iris-setosa": 1, "Iris-versicolor": -1, "Iris-virginica": -1}, inplace=True)
#X = dataset.iloc[:, 0:4].values
#d = dataset.iloc[:, 4:].values

p = Perceptron(X, d, 0.1, BinaryStep)

p.train()

print('')
print('')
print('###>>> Testes <<<###')
print(f'Input: [1, 1], output {p.evaluate([1,1])}')
print(f'Input: [1, 0], output {p.evaluate([1,0])}')
print(f'Input: [0, 1], output {p.evaluate([0,1])}')
print(f'Input: [0, 0], output {p.evaluate([0,0])}')


# Criando a figura com a solução do problema
for i in range(len(d)):
    if d[i] == 1:
        plt.plot(X[i,0], X[i,1], 'go')
    else:
        plt.plot(X[i,0], X[i,1], 'ro')

x_plot = np.arange(-2, 3)
y_plot = list(map(lambda x: (-1 * (p.W[1]/p.W[2]) * x) + (p.W[0]/p.W[2]), x_plot ))

plt.plot(x_plot, y_plot)