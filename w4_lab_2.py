from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

import numpy as np
from math import sqrt
from collections import Counter

# Implement KNN
def kNNClassify(K, X_train, y_train, X_predict): # LAZY classifier
    distances = [sqrt(np.sum((x-X_predict)**2)) for x in X_train]  # distance measure here
    sort = np.argsort(distances) # Index list
    topK = [y_train[i] for i in sort[:K]]
    votes = Counter(topK)
    y_predict = votes.most_common(1)[0][0]
    return y_predict

def kNN_predict(K, X_train, y_train, X_predict, y_predict):  # accuracy counter
    correct = 0
    for i in range(len(X_predict)):
        if y_predict[i] == kNNClassify(K, X_train, y_train, X_predict[i]):
            correct += 1

    print(correct/len(X_predict))


print("Training accuracy is ", end='')
kNN_predict(3, X_train, y_train, X_train, y_train)
print("Test accuracy is ", end='')
kNN_predict(3, X_train, y_train, X_test, y_test)

import itertools
import copy

# Attack on KNN
def kNN_attack(K, X_train, y_train, X_predict, y_predict):
    mat = np.diag([0.5, 0.5, 0.5, 0.5])*4
    flag = True
    for i in range(1, 5):
        for j in list(itertools.combinations([0,1,2,3],i)):
            delta = np.zeros(4)
            for k in j:
                delta += mat[k]

            # Core
            if y_predict != kNNClassify(K, X_train, y_train, copy.deepcopy(X_predict)+delta):
                X_predict += delta
                flag = False # Attack effective
                break

            if y_predict != kNNClassify(K, X_train, y_train, copy.deepcopy(X_predict) - delta):
                X_predict -= delta
                flag = False # Attack effective
                break

        if not flag:
            break

    print('data after attack: ', X_predict)
    print('predict label: ', kNNClassify(K, X_train, y_train, X_predict))

X_test_ = X_test[0]
y_test_ = y_test[0]

print('original data: ', X_test_)
print('original label: ', y_test_)
kNN_attack(3, X_train, y_train, X_test_, y_test_)


