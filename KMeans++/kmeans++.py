import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Função para inicializar os centróides aleatoriamente
def initialize_centroids(data, k):
    indices = np.random.choice(len(data), k, replace=False)
    return data[indices]

def initialize_centroids_pp(data, k):
    # Inicialização dos centróides usando o k-means++
    centroids = [data[np.random.randint(len(data))]]
    for _ in range(1, k):
        distances = [min([euclidean_distance(point, c) for c in centroids]) for point in data]
        probabilities = distances / sum(distances)
        cum_probabilities = np.cumsum(probabilities)
        random_value = np.random.rand()
        index = np.searchsorted(cum_probabilities, random_value)
        centroids.append(data[index])
    return np.array(centroids)

def assign_to_clusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    errors = []

    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append(point)
        error = distances[cluster_index] ** 2
        errors.append(error)

    return clusters, errors

def update_centroids(clusters):
    new_centroids = [np.mean(cluster, axis=0) for cluster in clusters]
    return new_centroids

def calculate_total_error(errors):
    return sum(errors)

def has_converged(old_centroids, new_centroids, tol=1e-4):
    return all(euclidean_distance(old, new) < tol for old, new in zip(old_centroids, new_centroids))

# Algoritmo K-means
def kmeans(data, k, max_iterations=100):
    centroids = initialize_centroids(data, k)
    eqt_values = []

    for i in range(max_iterations):
        clusters, errors = assign_to_clusters(data, centroids)
        eqt = calculate_total_error(errors)
        eqt_values.append(eqt)

        new_centroids = update_centroids(clusters)

        if has_converged(centroids, new_centroids):
            break

        centroids = new_centroids

    return clusters, centroids, eqt_values

def kmeanspp(data, k, max_iterations=100):
    centroids = initialize_centroids_pp(data, k)
    eqt_values = []

    for i in range(max_iterations):
        clusters, errors = assign_to_clusters(data, centroids)
        eqt = calculate_total_error(errors)
        eqt_values.append(eqt)

        new_centroids = update_centroids(clusters)

        if has_converged(centroids, new_centroids):
            break

        centroids = new_centroids

    return clusters, centroids, eqt_values


# Número de Clusters
k = 6

# Plotar os dados de cada cluster com cores diferentes e marcando as centroides
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
fig, ((axK1, axK2), (axKPP1, axKPP2)) = plt.subplots(2, 2, constrained_layout=True)

# Carregar os dados do arquivo
data = np.loadtxt("observacoescluster.txt")

# Executar o K-means com inicialização k-means
clusters, centroids, eqt_values = kmeans(data, k)

# Exibir os clusters, centróides e a curva do EQT
for i, cluster in enumerate(clusters):
    print(f"Cluster {i + 1} - Centróide: {centroids[i]}")
    for point in cluster:
        print(point)
    print("\n")

# Plotar os dados de cada cluster com cores diferentes e marcando as centroides - K-means
for i, cluster in enumerate(clusters):
    cluster_array = np.array(cluster)
    # Plotar os pontos do cluster com cor específica
    axK1.scatter(cluster_array[:, 0], cluster_array[:, 1], c=colors[i], label=f'Cluster {i + 1}', alpha=0.5)
    # Marcar a centróide com 'X'
    axK1.scatter(centroids[i][0], centroids[i][1], marker='o', s=200, c=colors[i], edgecolors='black')

axK1.set_xlabel('Feature 1')
axK1.set_ylabel('Feature 2')
axK1.set_title('Clusters e Centróides: K-Means')
axK1.legend()

# Plotar a curva do EQT - K-means
axK2.plot(range(len(eqt_values)), eqt_values)
axK2.set_xlabel('Iteração')
axK2.set_ylabel('Erro Quadrático Total (EQT) K-Means')
axK2.set_title('Curva do EQT x Iteração')

# Executar o K-means com inicialização k-means++
clusters, centroids, eqt_values = kmeanspp(data, k)

# Exibir os clusters, centróides e a curva do EQT
for i, cluster in enumerate(clusters):
    print(f"Cluster {i + 1} - Centróide: {centroids[i]}")
    for point in cluster:
        print(point)
    print("\n")

# Plotar os dados de cada cluster com cores diferentes e marcando as centroides - K-means
for i, cluster in enumerate(clusters):
    cluster_array = np.array(cluster)
    # Plotar os pontos do cluster com cor específica
    axKPP1.scatter(cluster_array[:, 0], cluster_array[:, 1], c=colors[i], label=f'Cluster {i + 1}', alpha=0.5)
    # Marcar a centróide com 'X'
    axKPP1.scatter(centroids[i][0], centroids[i][1], marker='o', s=200, c=colors[i], edgecolors='black')

axKPP1.set_xlabel('Feature 1')
axKPP1.set_ylabel('Feature 2')
axKPP1.set_title('Clusters e Centróides: K-Means++')
axKPP1.legend()

# Plotar a curva do EQT - K-means
axKPP2.plot(range(len(eqt_values)), eqt_values)
axKPP2.set_xlabel('Iteração')
axKPP2.set_ylabel('Erro Quadrático Total (EQT) K-Means++')
axKPP2.set_title('Curva do EQT x Iteração')

plt.show()
fig.savefig("Graphs.png")