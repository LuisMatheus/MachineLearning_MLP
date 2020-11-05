import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mlp import MLP

dataset = pd.read_csv('database/Trabalho Pratico - MLP - Classificação de Padrões - treinamento.csv')
X = dataset.iloc[:, 0:4].values
d = dataset.iloc[:, 4:].values

mlp = MLP(X,d, [15, 3])

mlp.train()
