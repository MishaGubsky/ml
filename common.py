import os
import numpy as np
from scipy.io import loadmat

COLORS = ['r', 'g', 'b', 'gold', 'orange', 'black', 'brown', 'cyan', 'magenta']
MARKERS = ['x', 'o', '.', '+']

DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), f'Data/')
CUSTOM_DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), f'CustomData/')


def load_data(filename, convert_type=float, separator=',', directory=DATA_DIRECTORY, encoding='utf-8',
              skip_extention=False, split_function=None):

    filepath = directory + filename
    if not skip_extention:
        filepath += '.txt'

    if not split_function:
        def split_function(line):
            return line.replace('\n', '').split(separator)

    data = []
    with open(filepath, 'r', encoding=encoding) as f:
        for line in f.readlines():
            try:
                if isinstance(convert_type, (list, tuple)):
                    data.append([
                        convert_type[j](el)
                        for row in line.replace('\n', '')
                        for j, el in enumerate(row.split(separator))
                    ])
                else:
                    data.append([convert_type(x) for x in split_function(line)])
            except TypeError:
                pass

    return np.array(data)


def load_data_from_mat_file(filename):
    filepath = os.path.join(os.path.dirname(__file__), f'Data/{filename}.mat')
    return loadmat(filepath)


def __convert_to_2d(X):
    try:
        X.shape[1]
        return X, False
    except IndexError:
        return np.array([X]).T, True


def extend_x(X):
    X, transposed = __convert_to_2d(X)
    return np.concatenate((np.array([np.ones(X.shape[0])]).T, X), axis=1)


def normalize_features(X):
    X, transposed = __convert_to_2d(X)
    delta = X.max(axis=0) - X.min(axis=0)
    average = X.sum(axis=0) / len(X)
    normalized = (X - average) / delta
    return normalized.T if transposed else normalized


def flatten_array_of_objects(X):
    return np.hstack([x.flatten() for x in X])
