import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering


def get_initial_cluster_centrals(X, claster_count):
    return X[np.random.choice(X.shape[0], claster_count, replace=False)]

def get_clusters(X, centroids):
    D = []
    for centroid in centroids:
        D.append(np.square(X - centroid).sum(axis=1))

    return np.argmin(np.array(D).T, axis=1)

def update_centroids(X, clusters, K):
    new_centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        if len(X[clusters == k]) == 0:
            continue

        new_centroids[k] = X[clusters == k].mean(axis=0)

    return new_centroids

def k_means_algorithm(X, K):
    centroids = get_initial_cluster_centrals(X, K)
    clusters = get_clusters(X, centroids)
    centroids_history = [centroids]

    while True:
        currrent_centroids = update_centroids(X, clusters, K)
        current_clusters = get_clusters(X, currrent_centroids)
        centroids_history.append(currrent_centroids)
        if (centroids_history[-1] == centroids_history[-2]).all():
            break

    return current_clusters, np.array(centroids_history)

def cost_func(X, c, centroids):
    M = X.shape[0]
    cost = 0
    for i in range(M):
        cost += np.square(X[i] - centroids[int(c[i])]).sum()

    return cost / M

def k_means(X, K, iteration_count=100):
    best_cost = np.inf
    best_result = None

    for i in range(iteration_count):
        clusters, centroids_history = k_means_algorithm(X, K)
        cost = cost_func(X, clusters, centroids_history[-1])

        if cost < best_cost:
            best_result = (clusters, centroids_history)
            best_cost = cost

    return best_result, best_cost

def compress(A, colors_count=16):
    X = np.reshape(A, [A.shape[0] * A.shape[1], A.shape[2]])
    (best_clusters, best_centroids_history), best_cost = k_means(X, colors_count, iteration_count=1)
    new_colors = np.round(best_centroids_history[-1]).astype(np.uint8)

    image = X.copy()
    for i in range(X.shape[0]):
        image[i, :] = new_colors[best_clusters[i]]

    return image.reshape(A.shape)

def compress_hierarchical_clusters(img, n_colors=16):
    X = np.reshape(img, [img.shape[0] * img.shape[1], img.shape[2]])

    cluster = AgglomerativeClustering(n_clusters=n_colors, affinity='euclidean', linkage='ward')
    cluster.fit(X)
    labels = cluster.labels_
    centroids = update_centroids(X, labels, n_colors).reshape((n_colors, 3))
    new_colors = np.round(centroids).astype(np.uint8)

    image = X.copy()
    for i in range(X.shape[0]):
        image[i, :] = new_colors[labels[i]]

    return image.reshape(img.shape)
