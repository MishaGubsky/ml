import numpy as np
import pandas as pd
from matplotlib import cm
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from os.path import isfile, join
from os import listdir
import collections
import nltk
import html
import re

from common import load_data_from_mat_file, COLORS, MARKERS, load_data, DATA_DIRECTORY, CUSTOM_DATA_DIRECTORY

DATA_FILE_NAME_1 = 'Lab 5/ex5data1'
DATA_FILE_NAME_2 = 'Lab 5/ex5data2'
DATA_FILE_NAME_3 = 'Lab 5/ex5data3'
DATA_FILE_NAME_4 = 'Lab 5/spamTrain'
DATA_FILE_NAME_5 = 'Lab 5/spamTest'
DATA_FILE_NAME_6 = 'Lab 5/vocab'

# 1

df_data = load_data_from_mat_file(DATA_FILE_NAME_1)
df_1 = pd.DataFrame({'x1': df_data['X'][:, 0], 'x2': df_data['X'][:, 1], 'y': df_data['y'][:, 0]})
X = df_1[['x1', 'x2']]
Y = df_1['y']

# 2


def plot_data():
    for i, group in df_1.groupby(['y']):
        plt.plot(group.x1, group.x2, COLORS[i] + MARKERS[i])


plot_data()
plt.show()

# 3

classifier_c1 = SVC(kernel='linear', C=1.0)
classifier_c1.fit(X, Y)

# 4

def plot_decision_boundary(classifiers_map, X, Y, **kwargs):
    kwargs.setdefault('contour_params', {})
    kwargs.setdefault('scatter_params', {})
    kwargs.setdefault('bias', 0.5)
    bias = kwargs.get('bias')
    h = .02

    x_min, x_max = X.x1.min() - 1, X.x1.max() + 1
    y_min, y_max = X.x2.min() - 1, X.x2.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = classifiers_map[0][1].predict(np.c_[xx.ravel(), yy.ravel()])
    plt.figure(figsize=(5 * len(classifiers_map), 4))

    for i, (title, classifier) in enumerate(classifiers_map):
        plt.subplot(1, len(classifiers_map), i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.2, **kwargs['contour_params'])

        plt.scatter(X.x1, X.x2, c=Y, cmap=cm.coolwarm, **kwargs['scatter_params'])
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.xlim(X.x1.min() - bias, X.x2.max() + bias)
        plt.ylim(X.x2.min() - bias, X.x2.max() + bias)
        plt.title(title)

    plt.show()


classifiers_map = [(f'SVC with linear kernel C={c}', SVC(kernel='linear', C=c)) for c in [1., 100.]]
[c.fit(X, Y) for title, c in classifiers_map]

plot_decision_boundary(classifiers_map, X, Y)

# 5


def get_gaussian_kernel_function(sigma=1.):
    def func(X, L):
        F = [np.exp(-((X - l) ** 2).sum(axis=1) / (2 * sigma ** 2)) for l in L]
        return np.array(F).T

    return func


gaussian_kernel_function = get_gaussian_kernel_function()
kernel = gaussian_kernel_function(X.values, X.values)

# 6

df_data = load_data_from_mat_file(DATA_FILE_NAME_2)
df_2 = pd.DataFrame({'x1': df_data['X'][:, 0], 'x2': df_data['X'][:, 1], 'y': df_data['y'][:, 0]})
X = df_2[['x1', 'x2']]
Y = df_2['y']

# 7

gaussian_kernel_function = get_gaussian_kernel_function(0.1)
kernel = gaussian_kernel_function(X.values, X.values)

# 8

gaussian_classifier = SVC(kernel=gaussian_kernel_function, C=1., gamma='scale')
gaussian_classifier.fit(X.values, Y.values)

# 9

gaussian_classifier_map = ('SVC with gaussian kernel C=1', gaussian_classifier)
plot_decision_boundary([gaussian_classifier_map], X, Y, bias=0.0)

# 10

df_data = load_data_from_mat_file(DATA_FILE_NAME_3)
df_3 = pd.DataFrame({'x1': df_data['X'][:, 0], 'x2': df_data['X'][:, 1], 'y': df_data['y'][:, 0]})
X = df_3[['x1', 'x2']]
Y = df_3['y']
df_3val = pd.DataFrame({'x1': df_data['Xval'][:, 0], 'x2': df_data['Xval'][:, 1], 'y': df_data['yval'][:, 0]})
Xval = df_3val[['x1', 'x2']]
Yval = df_3val['y']

# 11


def search_optimal_params(X, y, Xval, Yval, C_list, gamma_list):
    best_score = -np.inf
    best_params = None
    for C in C_list:
        for gamma in gamma_list:
            classifier = SVC(kernel='rbf', C=C, gamma=gamma)
            classifier.fit(X, y)
            score = classifier.score(Xval, Yval)
            if score > best_score:
                best_score = score
                best_params = (C, gamma)

    return best_params


values_list = (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30)
best_params = search_optimal_params(X, Y, Xval, Yval, C_list=values_list, gamma_list=values_list)
C_train, gamma_train = best_params
sigma_train = 1 / (2 * gamma_train)

print(f'Best params for validation set: C = {C_train}, sigma squared = {sigma_train}')

# 12


gaussian_classifiers = [SVC(kernel='rbf', C=C_train, gamma=gamma_train)] * 2
[gaussian_classifiers[i].fit(x.values, y.values) for i, (x, y) in enumerate([(X, Y), (Xval, Yval)])]

gaussian_classifier_map = (f'SVC with gaussian C={C_train} Train set', gaussian_classifiers[0])
plot_decision_boundary([gaussian_classifier_map], X, Y, bias=0.1)

gaussian_classifier_map = (f'SVC with gaussian C={C_train} Valid set', gaussian_classifiers[1])
plot_decision_boundary([gaussian_classifier_map], Xval, Yval, bias=0.1)

# 13

df_data = load_data_from_mat_file(DATA_FILE_NAME_4)
X = np.array(df_data['X'])
Y = np.array(df_data['y'])[:, 0]

# 14

gaussian_classifier = SVC(kernel='rbf', C=1., gamma='scale')
gaussian_classifier.fit(X, Y)

# 15

df_data = load_data_from_mat_file(DATA_FILE_NAME_5)
Xtest = np.array(df_data['Xtest'])
Ytest = np.array(df_data['ytest'])[:, 0]

# 16

best_params = search_optimal_params(X, Y, Xtest, Ytest,
                                    C_list=np.arange(2, 4, 10), gamma_list=np.linspace(0.0005, 0.005, 10))
C_train, gamma_train = best_params
sigma_train = 1 / (2 * gamma_train)

print(f'Best params for validation set: C = {C_train}, sigma squared = {sigma_train}', '\n')
C_train, gamma_train = 2, 0.005

# 17


def prepare_body(body):
    # a
    body = body.lower()

    # b
    text = html.unescape(body)
    body = re.sub(r'<[^>]+?>', '', text)

    # c
    regx = re.compile(r"(http|https)://[^\s]*")
    body = regx.sub(repl=" httpaddr ", string=body)

    # d
    regx = re.compile(r"\b[^\s]+@[^\s]+[.][^\s]+\b")
    body = regx.sub(repl=" emailaddr ", string=body)

    # e
    regx = re.compile(r"\b(\d+|\d+\.\d+)\b")
    body = regx.sub(repl=" number ", string=body)

    # f
    regx = re.compile(r"[$]")
    body = regx.sub(repl=" dollar ", string=body)

    # h
    regx = re.compile(r"([^\w\s]+)|([_-]+)")
    body = regx.sub(repl=" ", string=body)
    regx = re.compile(r"\s+")
    body = regx.sub(repl=" ", string=body)

    # g
    body = body.strip(" ")
    bodywords = body.split(" ")
    keepwords = [word for word in bodywords if word not in stopwords.words('english')]
    stemmer = SnowballStemmer("english")
    stemwords = [stemmer.stem(wd) for wd in keepwords]
    body = " ".join(stemwords)

    return body


# 18

vocab_data = load_data(DATA_FILE_NAME_6, str, '\t')
vocab = dict(zip(vocab_data[:, 1].tolist(), vocab_data[:, 0].tolist()))

# 19


def replace_with_codes(body, vocab):
    return set([vocab[word] for word in body.split(' ') if word in vocab])

# 20


def transform(text_codes, vocab):
    values_vector = np.zeros(len(vocab), dtype=int)

    for i, code in enumerate(vocab.values()):
        values_vector[i] = int(code in text_codes)

    return values_vector


def build_test_set(emails, vocab, is_processed=False):
    test_set = []

    for email in emails:
        processed_text = email if is_processed else prepare_body(email)
        codes = replace_with_codes(processed_text, vocab)
        values_vector = transform(codes, vocab)
        test_set.append(values_vector)

    return np.array(test_set)


# 21

spam_classificator = SVC(kernel='rbf', C=C_train, gamma=gamma_train)
spam_classificator.fit(X, Y)


nltk.download("stopwords")
print('\n')

filenames = ['emailSample1', 'emailSample2', 'spamSample1', 'spamSample2']
emails = [open(DATA_DIRECTORY + f'Lab 5/{filename}.txt').read() for filename in filenames]
test_set = build_test_set(emails, vocab)

result = spam_classificator.predict(test_set)

print('Spam classifier prediction:', result)
print('Excpected result:', [0, 0, 1, 1], '\n')


# 22

filenames = ['emailExample', 'emailSpam', 'emailExample1', 'emailSpam1']
custom_emails = [open(CUSTOM_DATA_DIRECTORY + f'5/{filename}.txt').read() for filename in filenames]
test_set = build_test_set(custom_emails, vocab)

result = spam_classificator.predict(test_set)

print('Spam classifier prediction:', result)
print('Excpected result:', [0, 1, 0, 1], '\n')

# 23

spam_path = CUSTOM_DATA_DIRECTORY + f'5/spam'
ham_path = CUSTOM_DATA_DIRECTORY + f'5/easy_ham'

spamfiles = [join(spam_path, fname) for fname in listdir(spam_path)]
hamfiles = [join(ham_path, fname) for fname in listdir(ham_path)]

all_files = hamfiles + spamfiles
emails_processed = [''] * len(spamfiles)
Yreal = [0] * len(hamfiles) + [1] * len(spamfiles)

for i, filename in enumerate(spamfiles):
    try:
        body = prepare_body(open(filename, 'r', encoding='utf-8').read())
    except UnicodeDecodeError:
        continue

    if not body:
        continue

    emails_processed[i] = prepare_body(body)

# build vocab
all_words = [word for email in emails_processed for word in email.split(" ")]
words_counter = collections.Counter(all_words)

words_list = [word for word in words_counter if words_counter[word] > 100 and len(word) > 1]
custom_vocab = {word: i for i, word in enumerate(words_list)}

##

emails = [''] * len(hamfiles)
for i, filepath in enumerate(hamfiles):
    try:
        emails[i] = prepare_body(open(filename, 'r', encoding='utf-8').read())
    except UnicodeDecodeError:
        continue

emails += emails_processed
real_test_set = build_test_set(emails, custom_vocab, is_processed=True)

real_classifier = SVC(kernel='rbf', C=C_train, gamma=gamma_train)
real_classifier.fit(real_test_set, Yreal)

print(f'Real classicator score: {real_classifier.score(real_test_set, Yreal)}')
print(f'Test classicator score: {spam_classificator.score(Xtest, Ytest)}')
