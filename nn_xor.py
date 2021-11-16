import numpy as np

x_train = np.array([[0., 0.],
                    [0., 1.],
                    [1., 0.],
                    [1., 1.]])

y_train = np.array(np.logical_xor(x_train[:, 0] > 0.5, x_train[:, 1] > 0.5), dtype=int)

from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(max_iter=10000, hidden_layer_sizes=(4, 2), random_state=1)
# clf = Perceptron(tol=1e-3, random_state=0)
# clf.fit(x_train, y_train, coef_init=np.array(
#     [
#         [1.5],
#         [1.5]
#     ]
# ))
clf.fit(x_train, y_train)

# print(clf.coef_)

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(x_train, y_train, clf=clf, legend=2)

import matplotlib
import matplotlib.pyplot as plt
import random

n_sample = 100
x_test = np.array(
    [
        [random.random() for i in range(2)] for j in range(n_sample)
    ]
)
y_test = np.array(np.logical_and(x_test[:, 0] > 0.5, x_test[:, 1] > 0.5), dtype=int)
y_pred = clf.predict(x_test)
print(clf.score(x_test, y_test))
colors = matplotlib.cm.rainbow(np.linspace(0, 1, 5))
plt.scatter(x_test[:, 0], x_test[:,  1], color=[colors[i] for i in y_pred])

plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision regions on two-dimensional data')
plt.show()


