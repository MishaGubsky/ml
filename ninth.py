import numpy as np
from scipy.sparse.linalg import svds

from common import load_data_from_mat_file, load_data

np.seterr(divide='ignore', invalid='ignore', over='ignore')

DATA_FILE_NAME_1 = 'Lab 9/ex9_movies'
DATA_FILE_NAME_2 = 'Lab 9/movie_ids'

# 1

df_data = load_data_from_mat_file(DATA_FILE_NAME_1)
Y = df_data['Y']
R = df_data['R']

# 2

features_count = 50

# 3 - 6


class RecomendationSystem:
    def __init__(self, features_count=10, reg_param=1e-1, learning_rate=1e-1, eps=1e-5, iteration_count=1e4):
        self.iteration_count = iteration_count
        self.features_count = features_count
        self.learning_rate = learning_rate
        self.reg_param = reg_param
        self.eps = eps

    def _cost_function(self):
        return np.square((np.dot(self.X, self.Theta) - self.Y) * self.R).sum() / 2

    def _cost_function_with_reg(self):
        error = self._cost_function()
        return error + self.reg_param * (np.square(self.X).sum() + np.square(self.Theta).sum()) / 2

    def _gradient_function(self):
        mean_error = (np.dot(self.X, self.Theta) - self.Y) * self.R
        delta_theta = np.dot(self.X.T, mean_error)
        return self.learning_rate * (delta_theta + self.reg_param * self.Theta)

    def _gradient_function_with_reg(self):
        return self._gradient_function() + self.learning_rate * self.reg_param * self.Theta

    def gradient_descent(self):
        iteration = 0
        accuracy_limit_achieved = False

        while not accuracy_limit_achieved and iteration < self.iteration_count:
            theta_gradient = self._gradient_function_with_reg()
            self.Theta -= theta_gradient

            accuracy_limit_achieved = not np.any(np.where(theta_gradient > self.eps))
            iteration += 1

        return self.Theta, accuracy_limit_achieved, iteration

    def _train(self):
        theta, accuracy_limit_achieved, iteration = self.gradient_descent()
        print('Accuracy limit achieved:', accuracy_limit_achieved)
        print('Iterations:', iteration)

    def fit(self, Y, R):
        self.Y, self.R = Y, R
        self.n_m, self.n_u = Y.shape
        self.X = np.random.rand(self.n_m, self.features_count)
        self.Theta = np.random.rand(self.features_count, self.n_u)
        self._train()

    def predict(self, user_id, top=10):
        predictions = np.dot(self.X, self.Theta)
        ratings = (self.R[:, user_id] != 1) * predictions[:, user_id]
        return ratings.argsort()[-top:][::-1]


# 7

# recomendation_system = RecomendationSystem(features_count=features_count)
# recomendation_system.fit(Y, R)

# 8

# 72 Mask, The (1984); 73 Maverick (1994); # 250 Fifth Element, The (1997); 257 Men in Black (1997);
# 204 Back to the Future (1985); 22 Braveheart (1995)
indexes = [72, 73, 250, 257, 204, 22]

my_ratings, presence = np.zeros(Y.shape[0], dtype=int), np.zeros(R.shape[0], dtype=int)
for i in indexes:
    my_ratings[i], presence[i] = 5, 1

my_Y = np.column_stack((Y, my_ratings))
my_R = np.column_stack((R, presence))
my_id = my_Y.shape[1] - 1

# 9

movies_list = load_data(DATA_FILE_NAME_2, separator='\n', convert_type=str, encoding='ISO-8859-1')

print('Recomendation system:')
recomendation_system = RecomendationSystem()
recomendation_system.fit(my_Y, my_R)
predictions = recomendation_system.predict(my_id)

print('\nPredicted movies:')
for movie in movies_list[predictions].flatten():
    print('- ', movie)

# 10


class SVDRecomendationSystem(RecomendationSystem):
    def fit(self, Y, R):
        self.Y, self.R = Y, R
        self.X, _, self.Theta = svds(Y.astype('float64'), k=features_count)
        self._train()


print('\nSVD recomendation system:')
svd_system = SVDRecomendationSystem()
svd_system.fit(my_Y, my_R)
svd_predictions = svd_system.predict(my_id)

print('\nPredicted movies using svd:')
for movie in movies_list[svd_predictions].flatten():
    print('- ', movie)

print('\nPrediction intersections: ', list(set(svd_predictions) & set(predictions)))
