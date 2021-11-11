import numpy as np
from sklearn.metrics import accuracy_score

from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
# print(type(iris.data))
X_train, X_test, y_train, y_test = train_test_split(iris.data[:100], iris.target[:100], test_size=0.2)

# Log classifier imple
def sigmoid(x):
    z = 1 / (1+ np.exp(-x))
    return z

def add_b(dataMat):
    dataMat = np.column_stack((np.mat(dataMat), np.ones(np.shape(dataMat)[0]))) # augmented matrix
    return dataMat

def LogRegre_(x_train, y_train, x_test, y_test, alpha=0.001, maxCycles = 500): # actually a bin-classifier
    x_train = add_b(x_train)
    x_test = add_b(x_test)
    y_train = np.mat(y_train).transpose() # np.ndarray -> np.mat
    y_test = np.mat(y_test).transpose()
    m, n = np.shape(x_train)
    weights = np.ones((n, 1)) # left mul compatible
    for i in range(0, maxCycles):
        h = sigmoid((x_train*weights))
        # print(x_train*weights) # test
        error = y_train - h
        weights += (alpha * x_train.transpose()*error)

    y_pre = sigmoid(np.dot(x_train, weights))
    for i in range(len(y_pre)):
        if y_pre[i] > 0.5:
            y_pre[i] = 1
        else:
            y_pre[i] = 0
    print("Train accuracy is %s" % (accuracy_score(y_train, y_pre)))

    y_pre = sigmoid(np.dot(x_test, weights))
    for i in range(len(y_pre)):
        if y_pre[i] > 0.5:
            y_pre[i] = 1
        else:
            y_pre[i] = 0
    print("Test accuracy is %s" % (accuracy_score(y_test, y_pre)))

    return weights

weights = LogRegre_(X_train, y_train, X_test, y_test)

## Attack on Log
import copy
import itertools

def Log_attack(weights, X_predict, y_predict):
    X_predict = add_b(X_predict)
    m = np.diag([0.5, 0.5, 0.5, 0.5])*4
    flag = True
    for i in range(1, 5):
        for ii in list(itertools.combinations([0, 1, 2, 3], i)):
            delta = np.zeros(4)
            for jj in ii:
                delta += m[jj]
            delta = np.append(delta, 0.)

            # pred after attack
            y_pre = sigmoid(np.dot(copy.deepcopy(X_predict)+delta, weights))

            if y_pre > 0.5:
                y_pre = 1
            else:
                y_pre = 0

            if y_predict != y_pre: # attack branch 1 effective
                X_predict += delta
                flag = False
                break

            y_pre = sigmoid(np.dot(copy.deepcopy(X_predict)-delta, weights))
            if y_pre > 0.5:
                y_pre = 1
            else:
                y_pre = 0
            if y_predict != y_pre: # attack branch 2 effective
                X_predict -= delta
                flag = False
                break
        if not flag: # attack overall effective
            break

    y_pre = sigmoid(np.dot(X_predict, weights)) # data->model->pre <-> np.dot(X_predict, model_weights)
    if y_pre > 0.5:
        y_pre = 1
    else:
        y_pre = 0
    print('attack data: ', X_predict[0, :-1])
    print('predict label: ', y_pre)

X_test_ = X_test[0:1]
y_test_ = y_test[0]
print('original data: ', X_test_)
print('original label: ', y_test_)

# print(type(weights))
Log_attack(weights, X_test_, y_test_)

# GNB imple
from collections import Counter

class GNB_():
    def __init__(self):
        self.prior = None
        self.avgs = None
        self.vars = None
        self.n_class = None

    def __get_prior(self, y):
        cnt = Counter(y) # a DS
        prior = np.array([cnt[i]/len(y) for i in range(len(cnt))])
        return prior

    def __get_avgs(self, X, y):
        return np.array([X[y==i].mean(axis=0) for i in range(self.n_class)])

    def __get_vars(self, X, y):
        return np.array([X[y==i].var(axis=0) for i in range(self.n_class)])

    def __get_likelihood(self, row):
        return (1 / np.sqrt(2 * np.pi * self.vars) * np.exp(-(row-self.avgs)**2 / (2 * self.vars))).prod(axis=1) # assume gaussian distribution to allow continuous regression

    def fit(self, X, y):
        self.prior = self.__get_prior(y)
        self.n_class = len(self.prior)
        self.avgs = self.__get_avgs(X, y)
        self.vars = self.__get_vars(X, y)

    def predict_prob(self, X):
        likelihood = np.apply_along_axis(self.__get_likelihood, axis=1, arr=X)
        probs = self.prior * likelihood # == posterior * observation
        probs_sum = probs.sum(axis=1)
        return probs / probs_sum[:, None] # normalize the array

    def predict(self, X):
        return self.predict_prob(X).argmax(axis=1) # multi-class

def get_acc(y, y_h):
    a = 0
    for i in range(len(y)):
        if y[i] == y_h[i]:
            a += 1
    return a/len(y)

clf = GNB_()
# train
clf.fit(X_train, y_train)
y_h = clf.predict(X_train)
acc = get_acc(y_train, y_h)
print("Train accuracy is %s"% acc)
# test
y_h = clf.predict(X_test)
acc = get_acc(y_test, y_h)
print("Test accuracy is %s"% acc)

## Attack on GNB
import copy
import itertools
def GNB_attack(clf, X_predict, y_predict):
    m = np.diag([0.5, 0.5, 0.5, 0.5])*4
    flag = True
    for i in range(1, 5):
        for ii in list(itertools.combinations([0, 1, 2, 3], i)):
            delta = np.zeros(4)
            for jj in ii:
                delta += m[jj]

            # pred after attack
            y_pre = clf.predict(copy.deepcopy(X_predict)+delta)

            if y_predict != y_pre: # attack branch 1 effective
                X_predict += delta
                flag = False
                break

            y_pre = clf.predict(copy.deepcopy(X_predict)-delta)
            if y_predict != y_pre: # attack branch 2 effective
                X_predict -= delta
                flag = False
                break
        if not flag: # attack overall effective
            break
    print('attack data: ', X_predict)
    print('predict label: ', clf.predict(copy.deepcopy(X_predict)))


X_test_ = X_test[0:1]
y_test_ = y_test[0]
print('original data: ', X_test_)
print('original label: ', y_test_)
GNB_attack(clf, X_test_, y_test_)