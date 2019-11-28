import re
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston

np.seterr(divide='ignore', invalid='ignore', over='ignore')

# 1

df_data = load_boston()
X = df_data[:, :-1]
Y = df_data[:, -1]

# 2

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.75)

# 4 - 7

trees_count = 50
etha = 0.9  # composition element coef


class GradientBoostingRegressor:
    models = []

    def __init__(self, etha=0.05, trees_count=50, **tree_params):
        self.ethas = np.array([etha(i) if callable(etha) else etha for i in range(trees_count)])
        self.tree_params = tree_params
        self.trees_count = trees_count
        self.trees = []

    def _negative_gradient(self, predictions, y):
        return y - predictions

    def predict(self, X, trees_limit=None):
        trees = self.trees[:trees_limit] if trees_limit else self.trees
        predictions = np.array([tree.predict(X) for tree in trees]).T
        return np.dot(predictions, self.ethas[:len(trees)])

    def fit(self, X, Y):
        for i in range(self.trees_count):
            predictions = self.predict(X)
            gradients = self._negative_gradient(predictions, Y)

            decision_tree = DecisionTreeRegressor(**self.tree_params)
            decision_tree.fit(X, gradients)
            self.trees.append(decision_tree)

    def score(self, X, Y, error_function=mean_squared_error, **predict_params):
        return error_function(Y, self.predict(X, **predict_params))


regressor = GradientBoostingRegressor(etha, max_depth=5, random_state=42)
regressor.fit(X_train, Y_train)


print('Prediction params:\nEthas: 0.9\n')
print('Prediction error on train set: ', regressor.score(X_train, Y_train))
print('Prediction error on test set: ', regressor.score(X_test, Y_test))

# 8

print('\n\nPrediction params:\nEthas: 0.9 / (1.0 + i)\n')

etha = lambda i: 0.9 / (1.0 + i)
regressor = GradientBoostingRegressor(etha, max_depth=5, random_state=42)
regressor.fit(X_train, Y_train)

print('Prediction error on train set: ', regressor.score(X_train, Y_train))
print('Prediction error on test set: ', regressor.score(X_test, Y_test))

# 9

# error by trees_count

trees_count = np.arange(1, 50, 2)
max_count = trees_count.max()

regressor = GradientBoostingRegressor(etha, trees_count=max_count, max_depth=5, random_state=42)
regressor.fit(X_train, Y_train)

y_train = np.array([regressor.score(X_train, Y_train, trees_limit=count) for count in trees_count])
y_test = np.array([regressor.score(X_test, Y_test, trees_limit=count) for count in trees_count])

best_trees_count = trees_count[np.argmin(y_test)]

plt.plot(trees_count, y_train, marker='.', label="train")
plt.plot(trees_count, y_test, marker='.', label="test")
plt.xlabel('Iteration count (trees count)')
plt.ylabel('Error')
plt.legend()
plt.show()

# error by trees_depth

trees_depth = np.arange(1, 20)
y_train, y_test = [], []

for depth in trees_depth:
    regressor = GradientBoostingRegressor(etha, max_depth=depth, random_state=42)
    regressor.fit(X_train, Y_train)
    y_train.append(regressor.score(X_train, Y_train))
    y_test.append(regressor.score(X_test, Y_test))


best_trees_depth = trees_depth[np.argmin(np.array(y_test))]

plt.plot(trees_depth, y_train, marker='.', label="train")
plt.plot(trees_depth, y_test, marker='.', label="test")
plt.xlabel('Trees depth')
plt.ylabel('Error')
plt.legend()
plt.show()

# 10

regressor = GradientBoostingRegressor(etha,
                                      trees_count=best_trees_count, max_depth=best_trees_depth, random_state=42)
regressor.fit(X_train, Y_train)
rmse = np.sqrt(regressor.score(X_test, Y_test))
print('\n\nRMSE on Gradient Boosting regression: ', rmse)

y_train = [regressor.score(X_train, Y_train, trees_limit=count) for count in trees_count]
y_test = [regressor.score(X_test, Y_test, trees_limit=count) for count in trees_count]


regressor = LinearRegression().fit(X_train, Y_train)
predictions = regressor.predict(X_test)
rmse = np.sqrt(mean_squared_error(Y_test, predictions))
print('RMSE on Linear regression: ', rmse)
