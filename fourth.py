import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

from common import load_data_from_mat_file, extend_x, flatten_array_of_objects


DATA_FILE_NAME_1 = 'Lab 4/ex4data1'
DATA_FILE_NAME_2 = 'Lab 4/ex4weights'

# 1

df_data = load_data_from_mat_file(DATA_FILE_NAME_1)
X = df_data['X']
E_X = extend_x(X)
Y = df_data['y']
# np.place(Y, Y == 10, 0)

# 2

df_data = load_data_from_mat_file(DATA_FILE_NAME_2)
Theta1 = df_data['Theta1']
Theta2 = df_data['Theta2']
Thetas = np.array([Theta1, Theta2])

# x = 400, a2 = 25, a3 = 10

# 3


def sigmoid_function(X):
    return 1. / (1 + np.exp(-X))


def forward_propagation_step(X, Thetas, activation_function=sigmoid_function):
    new_a_list = [X.copy().T]
    for ind, theta in enumerate(Thetas):
        Z = theta.dot(extend_x(new_a_list[-1].T).T)
        new_a_list.append(activation_function(Z))

    return np.array(new_a_list)

# 4


def check_prediction_accuracy(H, Y):
    true_predictions = 0
    for index, value in enumerate(Y):
        if H[index] + 1 == value:
            true_predictions += 1

    return true_predictions / len(Y)


def check_prediction_accuracy_with_fp(X, Thetas, Y):
    A = forward_propagation_step(X, Thetas)
    class_predictions = np.argmax(A[-1], axis=0)
    return check_prediction_accuracy(class_predictions, Y[:, 0])


print('prediction accuracy:', check_prediction_accuracy_with_fp(X, Thetas, Y))

# 5

one_hot_Y = np.zeros((5000, 10))
for ind, value in enumerate(Y[:, 0]):
    one_hot_Y[ind][value - 1] = 1

# 6


def cost_function(H, Y):
    return -(Y.dot(np.log(H).T) + (np.ones(Y.shape) - Y).dot(np.log(np.ones(H.shape) - H).T)).sum(axis=0) / len(Y)


def get_cost_function(X, Y, activation_function=sigmoid_function):
    def func(Thetas):
        A = forward_propagation_step(X, Thetas, activation_function)
        return cost_function(A[-1].T, Y)
    return func

# 7


def cost_function_with_reg(H, Y, Thetas, reg_param):
    R = [(np.square(theta) * reg_param).sum() for theta in Thetas]
    return cost_function(H, Y) + sum(R) / len(Y)


def get_cost_function_with_reg(X, Y, reg_param, activation_function=sigmoid_function):
    def func(Thetas):
        A = forward_propagation_step(X, Thetas, activation_function)
        return cost_function_with_reg(A[-1].T, Y, Thetas, reg_param)
    return func

# 8


def sigmoid_derivative_function(X):
    a = sigmoid_function(X)
    return np.multiply(a, (np.ones(a.shape) - a))

# 9


def get_random_Thetas():
    init_eps = 1e-3
    return np.array([
        np.random.rand(*Theta1.shape) * 2 * init_eps - init_eps,
        np.random.rand(*Theta2.shape) * init_eps - init_eps
    ])


random_Thetas = get_random_Thetas()

# 10


def back_propagation(Thetas, X, Y):
    A = forward_propagation_step(X, Thetas)
    M = Y.shape[0]
    deltas = [A[-1] - Y.T]
    for l in reversed(range(1, len(A) - 1)):
        derivative = sigmoid_derivative_function(A[l])
        delta = np.multiply(Thetas[l].T.dot(deltas[0]), extend_x(derivative.T).T)
        deltas.insert(0, delta[1:])

    return np.array([delta.dot(A[i].T) for i, delta in enumerate(deltas)]) / M


grads = back_propagation(Thetas, X, one_hot_Y)

# 11

GRAD_EPS = 1e-4


def check_gradient(X, Y, Thetas, Grads, activation_function=sigmoid_function, max_iterations=500):
    cost_function = get_cost_function(X, Y, activation_function)
    grad_vector = flatten_array_of_objects(Grads)
    iteration = 0
    grad_approx = []
    try:
        for l in range(Thetas.shape[0]):
            for i in range(Thetas[l].shape[0]):
                for j in range(Thetas[l].shape[1]):
                    Theta_l_plus = Thetas.copy()
                    Theta_l_plus[l][i, j] += GRAD_EPS
                    Theta_l_minus = Thetas.copy()
                    Theta_l_minus[l][i, j] -= GRAD_EPS
                    grad_approx_i = ((cost_function(Theta_l_plus) - cost_function(Theta_l_minus)) / 2 / GRAD_EPS).sum()
                    grad_approx.append(grad_approx_i)

                    iteration += 1
                    if max_iterations < iteration:
                        raise Exception
    except Exception:
        pass

    grad_approx = np.array(grad_approx)
    if grad_approx.size < grad_vector.size:
        temp = np.zeros(grad_vector.size)
        temp[:grad_approx.size] = grad_approx
        grad_approx = temp

    return np.allclose(grad_approx, grad_vector, atol=1)


# print('gradient check', check_gradient(X, one_hot_Y, Thetas, grads))

# 12


def back_propagation_with_reg(Thetas, X, Y, reg_param):
    D = back_propagation(Thetas, X, Y)
    M = Y.shape[0]
    return D + np.array([theta[:, 1:] for theta in Thetas]) * reg_param / M


grads_with_reg = back_propagation_with_reg(Thetas, X, one_hot_Y, 1)

# 13

print('gradient with reg check:', check_gradient(X, one_hot_Y, Thetas, grads_with_reg))

# 14

def gradient_descent_function_with_reg(Thetas, X, Y, reg_param=0, learning_rate=0.5, eps=1e-4, iteration_count=1e3):
    iteration = 0
    theta_list = Thetas.copy()
    flatten_theta_gradient = [eps + 1]

    while any([abs(t) > eps for t in flatten_theta_gradient]) and iteration < iteration_count:
        theta_gradient = back_propagation_with_reg(theta_list, X, Y, reg_param)
        # checking = check_gradient(X, Y, theta_list, theta_gradient)
        # print('checking', checking)
        for l, theta in enumerate(theta_list):
            theta[:, 1:] -= learning_rate * theta_gradient[l]

        flatten_theta_gradient = flatten_array_of_objects(theta_gradient)
        iteration += 1

    return theta_list, not any([abs(t) > eps for t in flatten_theta_gradient]), iteration


Thetas_with_reg, success, iteration = gradient_descent_function_with_reg(random_Thetas, X, one_hot_Y, reg_param=0.1)

print('Backpropagation algorithm with L2 regularization:')
print('success', success)
print('iteration', iteration, '\n')

# 15

print('prediction accuracy:', check_prediction_accuracy_with_fp(X, Thetas_with_reg, Y), '\n')

# 16


def plot_hidden_layer(X, Thetas):
    A = forward_propagation_step(X, Thetas)
    hidden_layer = A[1].T
    nums = list(range(150, 5000, 250))
    size = int(np.sqrt(hidden_layer.shape[1]))
    pictures = [hidden_layer[i].reshape((size, size)) for i in nums]
    fig, axs = plt.subplots(1, 20, figsize=(20, 0.85))
    for i, ax in enumerate(axs.flatten()):
        ax.pcolor(pictures[i], cmap=cm.gray)
        ax.axis('off')

    plt.show()


plot_hidden_layer(X, Thetas_with_reg)

# 17

reg_params_list = [100, 50, 10, 5, 1, 0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]

for i, reg_param in enumerate(reg_params_list):
    Thetas_with_reg, success, iteration = gradient_descent_function_with_reg(
        random_Thetas, X, one_hot_Y, reg_param=reg_param)
    print('prediction accuracy:', check_prediction_accuracy_with_fp(X, Thetas_with_reg, Y))
    print('reg_param:', reg_param, '\n')
    plot_hidden_layer(X, Thetas_with_reg)
