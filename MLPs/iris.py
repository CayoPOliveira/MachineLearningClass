from ucimlrepo import fetch_ucirepo
import random
import numpy as np
from classes import MLP

# porcentagem de dados para treinar
p = 0.8

# fetch dataset
iris = fetch_ucirepo(id=53)

# metadata
# print(iris.metadata)
# variable information
# print(iris.variables)

# data
X = iris.data.features.to_dict(orient="list")
y = iris.data.targets.to_dict(orient="list")

# Normalizando dados
infoX = {
    k : {'min':min(X[k]), 'max':max(X[k]), 'range':max(X[k])-min(X[k])}
    for k in X
}
X = [
    [ (i-infoX[k]['min'])/infoX[k]['range'] for i in X[k] ]
    for k in X
]
X = [[X[0][i], X[1][i], X[2][i], X[3][i]] for i in range(len(X[0]))]
# Representando Y
y = [[1,0,0] if val=='Iris-setosa' else [0,1,0] if val=='Iris-versicolor' else [0,0,1] for val in y['class']]

# Vizualizando os dados
# for i in X:
#     print(i)
# print(y)

# Separando data_train e data_test
if len(X) != len(y):
    print("Os dados em X não têm o mesmo tamanho que y")
    exit()

l = len(X)
train = int(p*l)
test = l - train
print(f"Número de dados total: {l}\nNúmero de dados de treinamento: {train}\nNúmero de dados de teste: {test}")

pos_test = random.sample(range(l), test)

X_train = np.array([X[i] for i in range(l) if i not in pos_test])
X_test = np.array([X[i] for i in range(l) if i in pos_test])

y_train = np.array([y[i] for i in range(l) if i not in pos_test])
y_test = np.array([y[i] for i in range(l) if i in pos_test])

# print(f"X_train=\n{X_train}\ny_train=\n{y_train}")
# print(f"X_train=\n{X_test}\ny_train=\n{y_test}")

## Restante do código permanece inalterado...

# Cria uma instância da rede neural
rede = MLP(ninputs=len(X[0]), nhidden=[16, 8, 4], noutputs=len(y_train[0]))

# # Treina a rede
rede.train(X_train, y_train, epochs=1000, learning_rate=0.01)

# # Faz previsões
previsoes = rede.forward(X_test)