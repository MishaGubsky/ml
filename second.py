import math
import random
import numpy as np
import pandas as pd
from matplotlib import cm
from decimal import Decimal
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

from common import load_data, load_data_from_mat_file

DATA_FILE_NAME_1 = 'Lab 2/ex2data1'
DATA_FILE_NAME_2 = 'Lab 2/ex2data2'
DATA_FILE_NAME_3 = 'Lab 2/ex2data3'

# # 1
# data_array = load_data(DATA_FILE_NAME_1)
# df_1 = pd.DataFrame({'ex1': data_array[:, 0], 'ex2': data_array[:, 1], 'passed': data_array[:, 2]})
# sorted_df_1 = df_1.sort_values(by=['ex1', 'ex2'])

# # 2

# dataset = sorted_df_1.query('passed==1')
# plt.plot(dataset.ex1, dataset.ex2, 'g+')
# dataset = sorted_df_1.query('passed==0')
# plt.plot(dataset.ex1, dataset.ex2, 'ro')


# # 3

# X = sorted_df_1[['ex1', 'ex2']].values
# X = np.concatenate((np.ones((len(X), 1)), X), axis=1)
# Y = sorted_df_1['passed'].values


def logistic_regration_hypotesis(T, X):
    return 1 / (1 + np.exp(-X.dot(T)))


def cost_function(H, Y):
    return -(Y.dot(np.log(H)) + (np.ones(Y.shape) - Y).dot(np.log(np.ones(H.shape) - H))) / len(Y)


def get_cost_function(X, Y):
    def func(T):
        H = logistic_regration_hypotesis(T, X)
        return cost_function(H, Y)
    return func


def gradient_function(X, Y, H):
    return (H - Y).dot(X) / len(Y)


def get_gradient_function(X, Y):
    def func(T):
        H = logistic_regration_hypotesis(T, X)
        return gradient_function(X, Y, H)
    return func


def gradient_descent_function(T, X, Y, learning_rate, eps, iteration_count):
    iteration = 0
    theta_gradient = [eps + 1]

    while any([abs(t) > eps for t in theta_gradient]) and iteration < iteration_count:
        H = np.array(logistic_regration_hypotesis(T, X))
        # print(T, theta_gradient)
        theta_gradient = gradient_function(X, Y, H)
        T -= learning_rate * theta_gradient
        iteration += 1

    return T, not any([abs(t) > eps for t in theta_gradient]), iteration

# # 4
# ## Nedler-Mead


# T0 = np.array([-2, 0.1, 0.1])
# eps = 1.0e-6

# cost_func = get_cost_function(X, Y)
# res = opt.minimize(cost_func, T0, method='Nelder-Mead', options={'xtol': eps, 'disp': True})
# print(res)
# print('\n\n')

# ## BFGS


# T0 = np.array([-2, 0.1, 0.1])

# cost_func = get_cost_function(X, Y)
# grad_func = get_gradient_function(X, Y)
# res = opt.minimize(cost_func, T0, method='bfgs', jac=grad_func, options={'disp': True})
# print(res)

# # 5

# T = [-25.16134055,   0.20623176,   0.20147166]


def predict_prop(X):
    if len(X) == 2:
        X = np.concatenate((np.ones(1), X), axis=0)

    return round(logistic_regration_hypotesis(T, X), 3) * 100

# # 6


# T = [-25.16134055,   0.20623176,   0.20147166]


def decision_boundary_function(x):
    return (-T[0] - T[1] * x) / T[2]


# x = [*range(25, 100)]
# H = [decision_boundary_function(xi) for xi in x]

# plt.plot(x, H)
# plt.show()

# # 7

# data_array = load_data(DATA_FILE_NAME_2)
# df_2 = pd.DataFrame({'test1': data_array[:, 0], 'test2': data_array[:, 1], 'passed': data_array[:, 2]})
# sorted_df_2 = df_2.sort_values(by=['test1', 'test2'])
# X = sorted_df_2[['test1', 'test2']].values
# Y = sorted_df_2['passed'].values

# # 8

# dataset = sorted_df_2.query('passed==1')
# plt.plot(dataset.test1, dataset.test2, 'g+')
# dataset = sorted_df_2.query('passed==0')
# plt.plot(dataset.test1, dataset.test2, 'ro')
# # plt.show()

# # 9


def get_extended_x(X):
    extended_X = []

    for i in range(0, 7):
        for j in range(0, 7):
            if i + j > 6:
                continue

            extended_X.append([x[0] ** i * x[1] ** j for x in X])

    return np.array(extended_X).T


# E_X = get_extended_x(X)
# # print(E_X)

# # 10


def cost_function_with_reg(H, Y, R):
    return -(Y.dot(np.log(H)) + (np.ones(Y.shape) - Y).dot(np.log(np.ones(H.shape) - H)) + R.sum()) / len(Y)


def get_cost_function_with_reg(X, Y, reg_param):
    def func(T):
        H = logistic_regration_hypotesis(T, X)
        R = np.square(T[1:]) * reg_param
        return cost_function_with_reg(H, Y, R)
    return func


def gradient_function_with_reg(X, Y, H, R):
    return ((H - Y).dot(X) + R) / len(Y)


def get_gradient_function_with_reg(X, Y, reg_param):
    def func(T):
        H = logistic_regration_hypotesis(T, X)
        R = T * reg_param
        R[0] = 0
        return gradient_function_with_reg(X, Y, H, R)
    return func


def gradient_descent_function_with_reg(T, X, Y, learning_rate, reg_param, eps, iteration_count):
    iteration = 0
    theta_gradient = [eps + 1]

    while any([abs(t) > eps for t in theta_gradient]) and iteration < iteration_count:
        H = np.array(logistic_regration_hypotesis(T, X))
        R = T * reg_param
        R[0] = 0
        theta_gradient = gradient_function_with_reg(X, Y, H, R)
        # print(T, theta_gradient)
        T -= learning_rate * theta_gradient
        iteration += 1

    return T, not any([abs(t) > eps for t in theta_gradient]), iteration


# T0 = np.ones(28, dtype=float)

# learning_rate = 1.0e-1
# reg_param = 1e-2
# eps = 1.0e-3
# iteration_count = 1.0e6
# theta, success, iteration = gradient_descent_function_with_reg(
#     T0, E_X, Y, learning_rate, reg_param, eps, iteration_count)

# print('L2 regularization:')
# print('theta', list(theta))
# print('success', success)
# print('iteration', iteration, '\n')

# # 11
# ##  Nedler-Mead

# T0 = np.ones(28, dtype=float)
# reg_param = 1e-1
# eps = 1.0e-1

# cost_func = get_cost_function_with_reg(E_X, Y, reg_param)
# res = opt.minimize(cost_func, T0, method='Nelder-Mead', options={'xtol': eps, 'disp': True})
# success = res.success

# print(res)
# print(res.success)
# print('\n\n')
# T = res.x

# ## BFGS

# T0 = np.ones(28, dtype=float)
# reg_param = 1e-2
# success = False

# cost_func = get_cost_function_with_reg(E_X, Y, reg_param)
# grad_func = get_gradient_function_with_reg(E_X, Y, reg_param)
# res = opt.minimize(cost_func, T0, method='bfgs', jac=grad_func, options={'disp': True})

# print(res)

# T = res.x

# # 12

# T = np.array([
#     3.788222961806053, 4.64429107285844, -6.068099288695758, -2.5446834111645336, -6.271168370690532,
#     2.8498431482312303, 0.30319248340021604, 2.0362498501728266, -6.696840463286545, 2.4684385851531783,
#     -1.990087560557413, -4.316802919724697, -3.6315893960144052, -5.575263779981924, -0.10786011242570737,
#     -3.6991611846659773, -3.6080049613616496, -4.468500605393321, 2.1161709668858255, 2.9874912859995164,
#     5.021226982392175, 3.204473862829851, -3.7508783143023083, -0.8653574461540079, -0.845156509588659,
#     -1.7174540165008965, 0.45113767052163006, -5.427897415287516
# ])


def get_predict_func(T):
    def func(X):
        e_x = get_extended_x([X])
        return round(logistic_regration_hypotesis(T, e_x)[0], 3) * 100

    return func


# # predict_prop = get_predict_func(T)

# # 13

# T = np.array([
#     3.788222961806053, 4.64429107285844, -6.068099288695758, -2.5446834111645336, -6.271168370690532,
#     2.8498431482312303, 0.30319248340021604, 2.0362498501728266, -6.696840463286545, 2.4684385851531783,
#     -1.990087560557413, -4.316802919724697, -3.6315893960144052, -5.575263779981924, -0.10786011242570737,
#     -3.6991611846659773, -3.6080049613616496, -4.468500605393321, 2.1161709668858255, 2.9874912859995164,
#     5.021226982392175, 3.204473862829851, -3.7508783143023083, -0.8653574461540079, -0.845156509588659,
#     -1.7174540165008965, 0.45113767052163006, -5.427897415287516
# ])
# ex_x = []
# X1 = np.arange(-1., 1.25, 0.01)
# X2 = np.arange(-1., 1.25, 0.01)

# e_x = get_extended_x([[x1, x2] for x1 in X1 for x2 in X2])
# Z = np.array([logistic_regration_hypotesis(T, e_x)])


# X1, X2 = np.meshgrid(X1, X2)
# Z = Z.reshape(X1.shape)
# plt.contour(X1, X2, Z, colors='black', levels=1)

# plt.show()

# # 14

# T0 = np.ones(28, dtype=float)
# success = False
# reg_param0 = 1e-3
# Ts = []

# for i in range(7):
#     reg_param = reg_param0 * 10 ** i
#     cost_func = get_cost_function_with_reg(E_X, Y, reg_param)
#     grad_func = get_gradient_function_with_reg(E_X, Y, reg_param)
#     res = opt.minimize(cost_func, T0, method='bfgs', jac=grad_func, options={'disp': True})
#     Ts.append(res.x)

# contours = []
# colors = ['blue', 'gold', 'orange', 'black', 'brown', 'cyan', 'magenta']
# X1 = np.arange(-1., 1.25, 0.01)
# X2 = np.arange(-1., 1.25, 0.01)

# e_x = get_extended_x([[x1, x2] for x1 in X1 for x2 in X2])
# for i, t in enumerate(Ts):
#     Z = np.array([logistic_regration_hypotesis(t, e_x)])

#     x1, x2 = np.meshgrid(X1, X2)
#     Z = Z.reshape(x1.shape)
#     contours.append(plt.contour(x1, x2, Z, levels=1, colors=colors[i]))

# plt.legend([mpatches.Patch(color=colors[i]) for i in range(7)], [f'lambda=1e{i-3}' for i in range(7)])

# plt.show()


# 15

df_data = load_data_from_mat_file(DATA_FILE_NAME_3)
X = df_data['X']
Y = df_data['y'][:, 0]

df_3 = pd.DataFrame(
    np.concatenate((df_data['X'], df_data['y']), axis=1),
    columns=[*[f'x{i}' for i in range(400)], 'y'])

# 16

# images_hash = {}
# for i, v in enumerate(X):
#     k = Y[i]
#     if k not in images_hash:
#         images_hash[k] = [v]
#         continue

#     images_hash[k].append(v)

# shape = (20, 20)
# fig, axs = plt.subplots(2, 5)
# axs = axs.flatten()
# for k, v in images_hash.items():
#     ax = axs[k % 10]
#     image = random.choice(v)
#     data = image.reshape(shape)
#     im = ax.imshow(data, cmap='gray', origin='lower')

# plt.show()

# 17


def get_binary_classificator(y0, X, Y):
    T0 = np.ones(len(X[0]), dtype=float)
    y = [1 if y == y0 else 0 for y in Y]
    iteration_count = 1e4
    learning_rate = 1e-1
    eps = 1e-1

    theta, success, iteration = gradient_descent_function(T0, X, y, learning_rate, eps, iteration_count)

    if not success:
        print('Binary classification with L2 regularization:')
        # print('y0', y0)
        # print('theta', theta)
        print('success', success)
        print('iteration', iteration, '\n')

    def func(X):
        return round(logistic_regration_hypotesis(theta, X), 3) * 100

    return func


classificator = get_binary_classificator(3, X, Y)
print(classificator(X[1990]))
# print([classificator(x) for x in X])


# 18


def get_binary_classificator_with_reg(y0, X, Y):
    T0 = np.ones(len(X[0]), dtype=float)
    y = [1 if y == y0 else 0 for y in Y]
    iteration_count = 1e6
    learning_rate = 1e-1
    reg_param = 1e-1
    eps = 1e-1

    theta, success, iteration = gradient_descent_function_with_reg(
        T0, X, y, learning_rate, reg_param, eps, iteration_count)

    if not success:
        print('Binary classification with L2 regularization:')
        # print('y0', y0)
        # print('theta', theta)
        print('success', success)
        print('iteration', iteration, '\n')

    def func(X):
        return round(logistic_regration_hypotesis(theta, X), 3) * 100

    return func


classificator = get_binary_classificator_with_reg(3, X, Y)
x = X[1990]
print(classificator(x))

# 19

shuffled_df_3 = df_3.sample(frac=1)
unique_y = np.unique(df_3['y'])

tranning = shuffled_df_3[:int(len(shuffled_df_3) * 0.8)]
tranning_x = tranning[tranning.columns.difference(['y'])].values
tranning_y = tranning['y'].values

classificators = {v: get_binary_classificator(v, tranning_x, tranning_y) for v in unique_y}

# 20

testing = shuffled_df_3[int(len(shuffled_df_3) * 0.8):]
testing_x = testing[testing.columns.difference(['y'])].values
testing_y = testing['y'].values


def predict_class(X, classificators=classificators):
    probs = {y_class: classificator(X) for y_class, classificator in classificators.items()}
    return max(probs, key=probs.get)  # class number


# 21

true_predictions = 0
for index, row in tranning.iterrows():
    predicted_class = predict_class(row[tranning.columns.difference(['y'])])
    if predicted_class == row['y']:
        true_predictions += 1

print('prediction accuracy:', true_predictions / len(tranning))
