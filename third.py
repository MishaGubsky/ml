import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import load_data_from_mat_file, extend_x, normalize_features

DATA_FILE_NAME_1 = 'Lab 3/ex3data1'

# 1

df_data = load_data_from_mat_file(DATA_FILE_NAME_1)
XY = pd.DataFrame({'x': df_data['X'][:, 0], 'y': df_data['y'][:, 0]})
sorted_XY = XY.sort_values(by=['x'])
X = sorted_XY['x'].values
E_X = extend_x(X)
Y = sorted_XY['y'].values

XYval = pd.DataFrame({'x': df_data['Xval'][:, 0], 'y': df_data['yval'][:, 0]})
sorted_XYval = XYval.sort_values(by=['x'])
Xval = sorted_XYval['x'].values
E_Xval = extend_x(Xval)
Yval = sorted_XYval['y'].values

XYtest = pd.DataFrame({'x': df_data['Xtest'][:, 0], 'y': df_data['Xtest'][:, 0]})
sorted_XYtest = XYtest.sort_values(by=['x'])
Xtest = sorted_XYtest['x'].values
E_Xtest = extend_x(Xtest)
Ytest = sorted_XYtest['y'].values

# 2

# plt.plot(X, Y, 'b.')

# 3


def linear_regression_hypotesis(T, X):
    return X.dot(T)


def cost_function_with_reg(H, Y, R):
    return (np.square(H - Y).sum() + R.sum()) / 2 / Y.size


def get_cost_function_with_reg(X, Y, reg_param):
    def func(T):
        H = linear_regression_hypotesis(T, X)
        R = np.square(T[1:]) * reg_param
        return cost_function_with_reg(H, Y, R)
    return func

# 4


def gradient_function_with_reg(X, Y, H, R):
    return ((H - Y).dot(X) + R) / len(Y)


def get_gradient_function_with_reg(X, Y, reg_param):
    def func(T):
        H = linear_regression_hypotesis(T, X)
        R = T * reg_param
        R[0] = 0
        return gradient_function_with_reg(X, Y, H, R)
    return func


def gradient_descent_function_with_reg(T, X, Y, learning_rate, reg_param, eps, iteration_count):
    iteration = 0
    theta_gradient = [eps + 1]

    while any([abs(t) > eps for t in theta_gradient]) and iteration < iteration_count:
        H = linear_regression_hypotesis(T, X)
        R = T * reg_param
        R[0] = 0
        theta_gradient = gradient_function_with_reg(X, Y, H, R)
        # print(theta_gradient)
        T -= learning_rate * theta_gradient
        iteration += 1

    return T, not any([abs(t) > eps for t in theta_gradient]), iteration


# 5

# T0 = np.ones(E_X.shape[1])
# learning_rate = 1.0e-4
# reg_param = 0
# eps = 1e-2
# iteration_count = 1.0e5
# theta, success, iteration = gradient_descent_function_with_reg(T0, E_X, Y, learning_rate,
#                                                                reg_param, eps, iteration_count)

# print('L2 regularization:')
# print('theta', list(theta))
# print('success', success)
# print('iteration', iteration, '\n')

# x = np.arange(-50, 40, 0.1)
# e_x = extend_x(x)
# y = linear_regression_hypotesis(theta, e_x)
# plt.plot(x, y, 'g')
# plt.show()

# 6


def get_learning_curves(training_sets, validation_set, initials={}):
    training_errors = {}
    validation_errors = {}
    eps = initials.get('eps') or 1e-2
    reg_param = initials.get('reg_param') or 0
    learning_rate = initials.get('learning_rate') or 1.0e-4
    iteration_count = initials.get('iteration_count') or 1.0e5

    e_xval = extend_x(validation_set['x'])
    for training_set in training_sets:
        T0 = initials.get('t0') or np.ones(training_set['x'].shape[1])

        theta, success, iteration = gradient_descent_function_with_reg(
            T0, training_set['x'], training_set['y'], learning_rate, reg_param, eps, iteration_count)

        set_length = len(training_set['y'])
        cost_function = get_cost_function_with_reg(training_set['x'], training_set['y'], reg_param)
        training_errors[set_length] = cost_function(theta)

        cost_function = get_cost_function_with_reg(e_xval, validation_set['y'], reg_param)
        validation_errors[set_length] = cost_function(theta)

    return training_errors, validation_errors


# e_x = extend_x(XY['x'])
# training_sets = [{'x': e_x[:count], 'y': XY['y'][:count].values} for count in range(1, len(XY['y']) + 1)]
# training_errors, validation_errors = get_learning_curves(training_sets, XYval)

# plt.plot(list(training_errors.keys()), list(training_errors.values()), label='training')
# plt.plot(list(validation_errors.keys()), list(validation_errors.values()), label='validation')
# plt.legend()
# plt.show()

# 7


def extend_x_by_p(X, p, x_index=1):
    try:
        X.shape[1]
    except IndexError:
        X = np.array([X]).T
        x_index = 0

    x1 = X[:, x_index]
    new_features = np.array([np.power(x1, i) for i in range(2, p + 1)])
    return np.concatenate((X, new_features.T), axis=1)


# 8

norm_X = normalize_features(X)
norm_Xval = normalize_features(Xval)

# 9

p = 8
e_x = extend_x_by_p(extend_x(norm_X), p)
T0 = np.ones(e_x.shape[1])
learning_rate = 1.0e0
reg_param = 0
eps = 1e-4
iteration_count = 1.0e7
theta, success, iteration_count = gradient_descent_function_with_reg(
    T0, e_x, Y, learning_rate, reg_param, eps, iteration_count)

print('Linear regression with L2 regularization:')
print('p', p)
print('theta', list(theta))
print('success', success)
print('iteration', iteration_count, '\n')

10

plt.plot(norm_X, Y, 'b.', label='data')
x = np.arange(np.min(norm_X), np.max(norm_X), 0.01)
y = linear_regression_hypotesis(theta, extend_x_by_p(extend_x(x), p))
plt.plot(x, y, 'g', label='hypotesis')
plt.show()

training_sets = [{'x': e_x[:count], 'y': Y[:count]} for count in range(1, len(Y) + 1)]
validation_set = {'x': extend_x_by_p(norm_Xval, p), 'y': Yval}
training_errors, validation_errors = get_learning_curves(training_sets, validation_set)

plt.plot(list(training_errors.keys()), list(training_errors.values()), label='training')
plt.plot(list(validation_errors.keys()), list(validation_errors.values()), label='validation')
plt.legend()

plt.show()

# 11


def plot_graphs(x, y, xval, yval, initials={}):
    p = initials.get('p', 8)
    e_x = extend_x_by_p(extend_x(x), p)
    T0 = initials.get('T0', np.ones(e_x.shape[1]))
    learning_rate = initials.get('learning_rate', 1e0)
    reg_param = initials.get('reg_param', 0)
    eps = initials.get('eps', 1e-4)
    iteration_count = initials.get('iteration_count', 1.0e7)

    theta, success, iteration_count = gradient_descent_function_with_reg(
        T0, e_x, y, learning_rate, reg_param, eps, iteration_count)

    print('Linear regression with L2 regularization:')
    print('p', p)
    print('reg_param', reg_param)
    print('theta', list(theta))
    print('success', success)
    print('iteration', iteration_count, '\n')

    plt.plot(x, y, 'b.', label='data')
    _x = np.arange(np.min(x), np.max(x), 0.01)
    _y = linear_regression_hypotesis(theta, extend_x_by_p(extend_x(_x), p))
    plt.plot(_x, _y, 'g', label='hypotesis')
    plt.show()

    training_sets = [{'x': e_x[:count], 'y': y[:count]} for count in range(1, len(y) + 1)]
    validation_set = {'x': extend_x_by_p(xval, p), 'y': yval}
    training_errors, validation_errors = get_learning_curves(training_sets, validation_set)

    plt.plot(list(training_errors.keys()), list(training_errors.values()), label='training')
    plt.plot(list(validation_errors.keys()), list(validation_errors.values()), label='validation')
    plt.legend()

    plt.show()


plot_graphs(norm_X, Y, norm_Xval, Yval, initials={'reg_param': 1})
plot_graphs(norm_X, Y, norm_Xval, Yval, initials={'reg_param': 100, 'eps': 1e-6, 'learning_rate': 1e-2})

# 12

learning_rate = 1.0e0
eps = 1e-4
iteration_count = 1.0e7

reg_errors = {}
thetas_hash = {}
for reg_param in np.arange(0, 5, 0.01):
    theta, success, iteration_count = gradient_descent_function_with_reg(
        T0, e_x, Y, learning_rate, reg_param, eps, iteration_count)
    thetas_hash[reg_param] = theta

    cost_function = get_cost_function_with_reg(extend_x_by_p(extend_x(norm_Xval), p), Yval, reg_param)
    reg_errors[reg_param] = cost_function(theta)

min_reg_param = min(reg_errors, key=reg_errors.get)
print('min error with reg_param:', min_reg_param)
plt.plot(list(reg_errors.keys()), list(reg_errors.values()))
plt.xlabel('Regularization param')
plt.ylabel('Error')
plt.show()

# 13

norm_Xtest = normalize_features(Xtest)
Ytest = normalize_features(Ytest)
reg_param = min_reg_param

cost_function = get_cost_function_with_reg(extend_x_by_p(extend_x(norm_Xtest), p), Ytest, reg_param)
print('cost error:', cost_function(thetas_hash[reg_param]))
