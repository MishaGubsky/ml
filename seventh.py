import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from common import load_data_from_mat_file, normalize_features
from sixth import compress

DATA_FILE_NAME_1 = 'Lab 7/ex7data1'
DATA_FILE_NAME_2 = 'Lab 7/ex7faces'
DATA_IMAGE_NAME = 'Lab 6/bird_small'

# 1

df_data = load_data_from_mat_file(DATA_FILE_NAME_1)
X = pd.DataFrame({'x1': df_data['X'][:, 0], 'x2': df_data['X'][:, 1]})
X_norm = normalize_features(X.values)

# 2

plt.plot(X_norm[:, 0], X_norm[:, 1], 'o')
plt.show()

# 3


def get_covariance_matrix(X):
    return np.dot(X.T, X) / X.shape[0]


# 4

def get_eigenvectors(X):
    Sigma = get_covariance_matrix(X)
    return np.linalg.svd(Sigma, full_matrices=False)


U, S, V = get_eigenvectors(X_norm)
print(U)

# 5

mu = X_norm.mean(axis=0)
projected_data = np.dot(X_norm, U)
variance = projected_data.std(axis=0).mean()
fig, ax = plt.subplots()
ax.plot(X_norm[:, 0], X_norm[:, 1], marker='o', linestyle="None", markersize=3)
for ind, axis in enumerate(U):
    start, end = mu, mu + variance * axis
    ax.annotate(
        '', xy=end, xycoords='data',
        xytext=start, textcoords='data',
        arrowprops=dict(facecolor='red', width=2.0))
ax.set_aspect('equal')
ax.set_title("Principle Component Analysis data with eigenvectors")
ax.set_xlabel('X1')
ax.set_ylabel('X2')
plt.show()

# 6


def pca_transform(X, n_dimensions, U=None):
    U = get_eigenvectors(X)[0] if U is None else U
    return np.dot(X, U[:, :n_dimensions]), U


n_dimensions = 1
Z, U = pca_transform(X_norm, n_dimensions, U=U)

# 7


def pca_reverse_transform(Z, U, n_dimensions):
    return np.dot(U[:, :n_dimensions], Z.T).T


X_approx = pca_reverse_transform(Z, U, n_dimensions)

# 8

fig, ax = plt.subplots()
ax.plot(X_norm[:, 0], X_norm[:, 1], marker='o', linestyle="None", markersize=3)

for i, x in enumerate(X_norm):
    x_approx = X_approx[i]
    plt.plot([x[0], x_approx[0]], [x[1], x_approx[1]], '--')

plt.plot(X_approx[:, 0], X_approx[:, 1], 'rx')
plt.show()

# 9

df_data = load_data_from_mat_file(DATA_FILE_NAME_2)
X = df_data['X']
print('X shape:', X.shape)

# 10

X_rand = X[np.random.choice(X.shape[0], 100, replace=False), :]
fig, axs = plt.subplots(10, 10)
axs = axs.flatten()

for i, x in enumerate(X_rand):
    image = np.reshape(x, (32, 32), order="F")
    axs[i].imshow(image, cmap='gray')
    axs[i].xaxis.set_visible(False)
    axs[i].yaxis.set_visible(False)

plt.show()

# 11

U, S, V = get_eigenvectors(X)

# 12


def plot_components(V, components_count):
    size = int(np.sqrt(components_count))
    fig, axs = plt.subplots(size, size, sharex=True, sharey=True, figsize=(10, 10))
    fig.suptitle(f"{components_count} PCA Eigenvectors of faces", fontsize=18)
    axs = axs.flatten()

    for i in range(components_count):
        image = np.reshape(V[i, :], (32, 32), order="F")
        axs[i].imshow(image, cmap='gray')
        axs[i].xaxis.set_visible(False)
        axs[i].yaxis.set_visible(False)


plot_components(V, 36)
plt.show()

# 14

plot_components(V, 100)
plt.show()

# 16

df_data = load_data_from_mat_file(DATA_IMAGE_NAME)
A = df_data['A']
compressed_A = compress(A)


# 17

A_norm = normalize_features(A.reshape((A.shape[0] * A.shape[1], A.shape[2])))
Ax_reduced, U = pca_transform(A_norm, 2)
Ax_approx = pca_reverse_transform(Ax_reduced, U, 2)

fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(xs=Ax_approx[:, 0], ys=Ax_approx[:, 1], zs=Ax_approx[:, 2], s=1)

ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(Ax_reduced[:, 0], Ax_reduced[:, 1], s=2)
plt.show()
