import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from common import load_data, normalize_features, extend_x

DATA_FILE_NAME_1 = 'Lab 1/ex1data1'
DATA_FILE_NAME_2 = 'Lab 1/ex1data2'


# 1
data_array = load_data(DATA_FILE_NAME_1)
df_1 = pd.DataFrame({'population': data_array[:, 0], 'profit': data_array[:, 1]})

# 2

X = df_1['population'].values
Y = df_1['profit'].values
plt.plot(X, Y, 'o', label='dependency')

# 3


def get_hypotesis_function(theta0, theta1):
    def func(x):
        return theta1 * x + theta0

    return func


def get_cost_function(X, Y):
    m = Y.size

    def func(theta0, theta1):
        h = get_hypotesis_function(theta0, theta1)
        H = h(X)

        return np.sum((H - Y)**2) / 2 / m

    return func


# 4

def mse(H, Y):
    return sum([delta**2 for delta in H - Y]) / 2 / H.size


def get_gradient_function(X, Y):
    m = len(X)

    def func(theta0, theta1, H):
        theta0_gradient = (H - Y).sum() / m
        theta1_gradient = ((H - Y).dot(X)).sum() / m
        return theta0_gradient, theta1_gradient

    return func


def get_optimal_params(
        theta0, theta1, X, Y, learning_rate=0.01, eps=10**-3, iteration_count=1e3, save_error_history=False):
    gradient_function = get_gradient_function(X, Y)
    theta0_gradient = theta1_gradient = np.inf
    error_history = []
    iteration = 0

    while (abs(theta0_gradient) > eps or abs(theta1_gradient) > eps) and iteration < iteration_count:
        h = get_hypotesis_function(theta0, theta1)
        H = np.array([h(x) for x in X])
        theta0_gradient, theta1_gradient = gradient_function(theta0, theta1, H)
        theta0, theta1 = theta0 - (learning_rate * theta0_gradient), theta1 - (learning_rate * theta1_gradient)
        iteration += 1
        if save_error_history:
            error_history.append(mse(H, Y))

    return theta0, theta1, error_history


theta0, theta1, *_ = get_optimal_params(-4, 2, X, Y)
print('theta0, theta1:', theta0, theta1)
# -3.895780878865218 1.193033644245185

h = get_hypotesis_function(theta0, theta1)
plt.plot(X, h(X), label='hypotesis')

plt.show()

# 5

j = get_cost_function(X, Y)

fig = plt.figure()
ax = plt.axes(projection="3d")
T0 = np.arange(-5, -2.5, 0.01)
T1 = np.arange(1, 1.3, 0.01)
T0, T1 = np.meshgrid(T0, T1)
Z = np.array([j(np.ravel(T0)[i], np.ravel(T1)[i]) for i in range(T0.size)])
Z = Z.reshape(T0.shape)
ax.plot_surface(T0, T1, Z, linewidth=0, antialiased=False, cmap=cm.coolwarm)

ax.set_title('Cost function dependency from params (3D surface)')
ax.set_xlabel('T0')
ax.set_ylabel('T1')
ax.set_zlabel('J(T0, T1)')

plt.show()

plt.title('Cost function dependency from params (2D surface)')
plt.contour(T0, T1, Z, colors='black')
plt.xlabel('T0')
plt.ylabel('T1')

plt.show()

# 6

data_array = load_data(DATA_FILE_NAME_2)
df_2 = pd.DataFrame({'area': data_array[:, 0], 'rooms': data_array[:, 1], 'price': data_array[:, 2]})
A = df_2['area'].values
R = df_2['rooms'].values
P = df_2['price'].values
X = df_2[['area', 'rooms']].values
E_X = extend_x(X)

# 7


def not_vectorized_hypotesis(T, X):
    H = []
    for x in X:
        H.append(T[0] + sum([T[i + 1] * xi for i, xi in enumerate(x)]))

    return np.array(H)



def get_optimal_params_using_vectors(Thetas, X, Y, learning_rate=1e-1, eps=10**-3,
                                     iteration_count=1e5, save_error_history=False):
    theta_gradient = np.inf
    error_history = []
    iteration = 0
    T = Thetas.copy()

    while np.any(np.abs(theta_gradient) > eps) and iteration < iteration_count:
        H = vectorized_hypotesis(T, X)
        theta_gradient = vectorized_gradient_function(X, H, Y)
        T -= learning_rate * theta_gradient
        iteration += 1
        if save_error_history:
            error_history.append(mse(H, Y))

    return T, error_history

def get_multifeatures_regression_optimal_params(
        Thetas, X, Y, learning_rate=1e-1, eps=10**-3, iteration_count=1e5, save_error_history=False):
    theta_gradient = np.inf
    error_history = []
    iteration = 0
    T = Thetas.copy()

    while np.any(np.abs(theta_gradient) > eps) and iteration < iteration_count:
        H = not_vectorized_hypotesis(T, X)
        theta_gradient = not_vectorized_gradient_function(X, H, Y)
        T -= learning_rate * theta_gradient
        iteration += 1
        if save_error_history:
            error_history.append(mse(H, Y))

    return T, error_history


T0 = np.array([-1., 1., 1.])
norm_A = normalize_features(A)
norm_R = normalize_features(R)
norm_X = np.hstack((norm_A.T, norm_R.T))
E_X_norm = extend_x(norm_X)

thetas, error_history = get_multifeatures_regression_optimal_params(T0, X, P, save_error_history=True)
thetas, norm_error_history = get_multifeatures_regression_optimal_params(T0, norm_X, P, save_error_history=True)

max_iterations = np.min([len(error_history), len(norm_error_history)])
xx = np.arange(1, max_iterations + 1)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].set_title('Using not normolized params')
axs[0].plot(xx, error_history[:max_iterations])
axs[0].set_xlabel('Iteration count')
axs[0].set_ylabel('MSE')

axs[1].set_title('Using normolized params')
axs[1].plot(xx, norm_error_history[:max_iterations])
axs[1].set_xlabel('Iteration count')
plt.show()

# 8


def vectorized_cost_function(T, X, Y):
    H = np.dot(T, X.T)
    return np.square(H - Y).sum() / 2 / H.shape[0]


def vectorized_hypotesis(T, X):
    return np.dot(X, T)


def vectorized_gradient_function(X, H, Y):
    return np.dot((H - Y).T, X) / Y.shape[0]


def get_optimal_params_using_vectors(
        Thetas, X, Y, learning_rate=1e-1, eps=10**-3, iteration_count=1e5, save_error_history=False):
    theta_gradient = np.inf
    error_history = []
    iteration = 0
    T = Thetas.copy()

    while np.any(np.abs(theta_gradient) > eps) and iteration < iteration_count:
        H = vectorized_hypotesis(T, X)
        theta_gradient = vectorized_gradient_function(X, H, Y)
        T -= learning_rate * theta_gradient
        iteration += 1
        if save_error_history:
            error_history.append(mse(H, Y))

    return T, error_history


# 9

import datetime

now = datetime.datetime.now()
_ = get_multifeatures_regression_optimal_params(T0, norm_X, P, learning_rate=1e-2)
print('Simple regression performance:', datetime.datetime.now() - now)

now = datetime.datetime.now()
_ = get_optimal_params_using_vectors(T0, E_X_norm, P, learning_rate=1e-2)
print('Vectorizes regression performance:', datetime.datetime.now() - now)


# 10

learning_rates = [0.1, 0.01, 0.001, 0.0001]
size = int(np.sqrt(len(learning_rates)))
fig, axs = plt.subplots(size, size, figsize=(8, 8))
axs = axs.flatten()

for i, learning_rate in enumerate(learning_rates):
    _, errors = get_optimal_params_using_vectors(T0, E_X_norm, P, learning_rate=learning_rate, save_error_history=True)
    axs[i].plot(np.arange(1, len(errors) + 1), errors)
    axs[i].set_title(f'Linear rate = {learning_rate:.4f}')
    axs[i].set_ylabel('Error')
    if i >= len(axs) - size:
        axs[i].set_xlabel('Number of iteration')

plt.show()

# 11


def get_optimal_params_using_normal_equation(X, Y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)


_, history = get_optimal_params_using_vectors(T0, E_X_norm, P, learning_rate=1e-1, save_error_history=True)
print('Gradient descent error:', history[-1])

T_normal_equation = get_optimal_params_using_normal_equation(norm_X, P)
error = vectorized_cost_function(T_normal_equation, norm_X, P)
print('Normal equation error:', error)
