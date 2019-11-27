import numpy as np
from matplotlib import cm
import scipy.stats as stats
import matplotlib.pyplot as plt

from common import load_data_from_mat_file
np.seterr(divide='ignore', invalid='ignore')

DATA_FILE_NAME_1 = 'Lab 8/ex8data1'
DATA_FILE_NAME_2 = 'Lab 8/ex8data2'

# 1

df_data = load_data_from_mat_file(DATA_FILE_NAME_1)
X = df_data['X']
Xval = df_data['Xval']
Yval = df_data['yval']
print(f"{'x'.join([str(i) for i in X.shape])}-dimensional set:\n")

# 2

plt.scatter(X[:, 0], X[:, 1], marker='.')
plt.show()

# 3

fig, axs = plt.subplots(1, 2, figsize=(20, 5))
axs[0].hist(X[:, 0], 100, density=True)
axs[1].hist(X[:, 1], 100, density=True)
plt.show()

# 4


def get_dist_params(X):
    return X.mean(axis=0), X.std(axis=0)


Mu, Sigma = get_dist_params(X)
print('Mean:', Mu)
print('Sigma:', Sigma, '\n')

# 5


def p(X, Mu=None, Sigma=None):
    Mu, Sigma = get_dist_params(X) if Mu is None and Sigma is None else (Mu, Sigma)
    axis = int(len(X.shape) > 1)
    return stats.norm.pdf(X, Mu, Sigma).prod(axis=axis)


h = 2.
x = np.linspace(X[:, 0].min() - h, X[:, 0].max() + h, 50)
y = np.linspace(X[:, 1].min() - h, X[:, 1].max() + h, 50)
xx, yy = np.meshgrid(x, y)
Xnew = np.column_stack((xx.flatten(), yy.flatten()))
Z = p(Xnew).reshape((len(xx), len(yy)))
plt.contour(xx, yy, Z, cmap=cm.plasma)
plt.scatter(X[:, 0], X[:, 1], marker='o')

# 6


def f1_score(true_positive, false_positive, false_negative):
    try:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        return 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return 0


def get_check_params_function(X, Xval, Y, prop_function=p, dist_params_function=get_dist_params):
    Mu, Sigma = dist_params_function(X)
    props = prop_function(Xval, Mu=Mu, Sigma=Sigma)

    def func(eps):
        indexes = np.argwhere(props < eps)
        true_positive = Y[indexes].sum()
        false_positive = int(len(indexes) - true_positive)

        indexes = np.argwhere(props >= eps)
        false_negative = Y[indexes].sum()
        return f1_score(true_positive, false_positive, false_negative)

    return func


def get_optimal_eps(X, Xval, Yval, eps_list=None, prop_function=p, dist_params_function=get_dist_params):
    best_eps = 0
    max_score = -np.inf
    eps_list = np.arange(0, 0.4, 1e-2) if eps_list is None else eps_list
    check_params_function = get_check_params_function(
        X, Xval, Yval, prop_function=prop_function, dist_params_function=dist_params_function)

    for eps in eps_list:
        metric = check_params_function(eps)
        if metric >= max_score:
            max_score = metric
            best_eps = eps

    print('Best eps:', best_eps)
    print('F1 score:', max_score, '\n')
    return best_eps


best_eps = get_optimal_eps(X, Xval, Yval, eps_list=np.arange(0, 0.4, 1e-4))

# 7

X_props = p(X)
indexes = np.argwhere(X_props < best_eps)
anomalies = X[indexes.flatten()]
print(f'Number of anomalies: {anomalies.shape[0]}\n')
plt.scatter(anomalies[:, 0], anomalies[:, 1], c='r', marker='x', s=100)
plt.show()

# 8

df_data = load_data_from_mat_file(DATA_FILE_NAME_2)
X = df_data['X']
Xval = df_data['Xval']
Yval = df_data['yval']
print(f"\n{'x'.join([str(i) for i in X.shape])}-dimensional set:\n")

# 9

size = int(np.sqrt(X.shape[1]))

fig, axs = plt.subplots(size, size + 1, figsize=(20, 5))
axs = axs.flatten()
[fig.delaxes(ax) for ax in axs[X.shape[1]:]]

for i in range(X.shape[1]):
    axs[i].hist(X[:, i], 100, density=True)

plt.show()

# 10


def get_multivariate_dist_params(X):
    Mu, Sigma_v = get_dist_params(X)
    Sigma_m = np.zeros((len(Sigma_v), len(Sigma_v)))
    np.fill_diagonal(Sigma_m, Sigma_v)
    return Mu, Sigma_m


Mu, Sigma = get_multivariate_dist_params(X)
print('Mean:', Mu)
print('Sigma shape:', Sigma.shape, '\n')

# 11


def multivariate_p(X, Mu=None, Sigma=None):
    Mu, Sigma = get_multivariate_dist_params(X) if Mu is None and Sigma is None else (Mu, Sigma)
    return stats.multivariate_normal.pdf(X, Mu, Sigma)


X_props = multivariate_p(X)

min_rank = -int(np.log10(X_props.min()))
eps_list = [10 ** (-min_rank + i) for i in range(min_rank)]
best_eps = get_optimal_eps(X, Xval, Yval,
                           eps_list=eps_list,
                           prop_function=multivariate_p,
                           dist_params_function=get_multivariate_dist_params)

# 12

indexes = np.argwhere(X_props < best_eps)
anomalies = X[indexes.flatten()]
print(f'Number of anomalies: {anomalies.shape[0]}')
