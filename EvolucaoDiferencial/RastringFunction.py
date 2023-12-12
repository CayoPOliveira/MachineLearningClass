## O objetivo desse trabalho era aplicar a evolução diferêncial para cálcular o mínimo da função de Rastring
#   f(x) = 10n + sum(for i in range(1,n+1), xi^2 - 10cos(2*pi*xi) )
#   -5.12 <= xi <= 5.12
#   Obs: A resposta é min(f(x)) = f(0,..,0) = 0

# MODULOS ========================================================================
import math
import numpy as np
from numpy.random import random, choice

# PARÂMETROS DE SIMULAÇÃO =========================================================
n = 5 # n-dimensions
maxG = 30000 # Número máximo de Gerações
populationSize = 50 # Tamanho da população
crossoverRate = 0.5
mutationRate = 0.1

evolution = 1000 # A cada x gerações irá mostrar o resultado do melhor
stop = False # Determina se deve parar quando o melhor indivíduo tiver resultado menor que stopVal
stopVal = 1e-12 # Se stop==True, irá parar na geração que atingir valor menor que stopVal

# FUNÇÃO DE RASTRING ====================================================================
def f(X:list):
    global n
    return 10*n + sum(
            [(xi**2 - 10*math.cos(2*math.pi*xi)) for xi in X]
        )

# APTIDAO ===============================================================================
def aptidao(G:list):
    return [f(x) for x in G]

# TORNEIO SELECÃO =======================================================================
def Torneio(F: list):
    x,y,z = choice(len(F) , size=3, replace=False)
    menor = x if F[x] < F[y] and F[x] < F[z] \
        else y if F[y] < F[x] and F[y] < F[z] \
        else z
    return menor

# CROSSOVER =============================================================================
def crossover(pai1:list, pai2:list):
    P1, P2 = crossoverRate, 1-crossoverRate
    filho = [pai1[i] if pai1[i]==pai2[i] \
             else P1*pai1[i] + P2*pai2[i] \
             for i in range(n)]
    return filho

# MUTAÇÃO ===============================================================================
def mutate(genotipo:list):
    filho = genotipo[:]
    idx1, idx2 = choice(len(genotipo), size=2)
    filho[idx1], filho[idx2] = -5.12, 5.12
    return filho

# SIMULAÇÃO =============================================================================

# Primeira geração totalmente aleatória
G = [
        [5.12 * 2 * (random()-0.5) for x in range(n)]
        for p in range(populationSize)
    ]

# Rodar o número de gerações
i = 0
while(True):
    if not stop and i==maxG:
        break
    # Calculando a aptidão de todos da atual geração
    F = aptidao(G)

    # Gerando filhos
    Filhos = []
    for j in range(populationSize):
        pai1, pai2 = Torneio(F), Torneio(F)

        while(pai1==pai2): # Garantir que os pais são diferentes
            pai2 = Torneio(F)
        if F[pai2] < F[pai1]: # Gerantindo que o pai1 é melhor que pai2
            pai1, pai2 = pai2, pai1

        # Gerando filho
        filho = crossover(G[pai1], G[pai2])
        if random() < mutationRate: # Mutação
            filho = mutate(filho)
        Filhos.append(filho)

    # newG será a nova geração e estou garantindo que o melhor da geração anterior está na nela
    newG = [ G.pop(F.index(min(F))) ]
    if i%evolution==0:
        print(f"Geração {i}: Melhor f(x)={f(newG[0])}", flush=True)
    if(stop and f(newG[0]) < stopVal):
        print(f"STOP CONDITION - Geração {i} teve seu melhor f(x)={f(newG[0])}")
        break

    # Selecionando os membros da nova geração
    G_intermediaria = G[:] + Filhos[:]
    while(len(newG) != populationSize):
        idx = Torneio(aptidao(G_intermediaria))
        newG.append(G_intermediaria.pop(idx))

    # Nova geração passada para proxima iteração
    G = newG[:]
    i = i + 1


# Melhor indivíduo com menor valor da função
F = aptidao(G)
print(f"Resultado: {G[F.index(min(F))]}")
