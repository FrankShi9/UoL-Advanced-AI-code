from sklearn import datasets
# dataset = datasets.load_digits()
#
# x = dataset.data
# y = dataset.target
#
# print("===== Get Basic Information ======")
# observations = len(x)
# features = len(dataset.feature_names)
# classes = len(dataset.target_names)
# print("Number of Observations: " + str(observations))
# print("Number of Features: " + str(features))
# print("Number of Classes: " + str(classes))
#
# print("===== Split Dataset ======")
from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
#
# print("===== Model Training ======")
# from sklearn.linear_model import Perceptron
# clf = Perceptron(tol=1e-3, random_state=0)
# clf.fit(x_train, y_train)
#
# print("===== Model Prediction ======")
# print("Labels of all instances:\n%s" % y_test)
# y_pred = clf.predict(x_test)
# print("Predictive outputs of all instances:\n%s" % y_pred)
#
# print("===== Confusion Matrix ======")
# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
#


from sklearn import svm
import numpy as np
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x = iris.data
y = iris.target

# Add noise
random_state = np.random.RandomState(0)
n_samples, n_features = x.shape
x = np.c_[x, random_state.randn(n_samples, 200*n_features)]

# Limit to the two first classes (i.e. label=0 or 1), and split into training and test
x_train, x_test, y_train, y_test = train_test_split(x[y < 2], y[y < 2], test_size=0.5, random_state=random_state)

# Create a simple classifier
clf = svm.LinearSVC(random_state=random_state)
clf.fit(x_train, y_train)
y_score = clf.decision_function(x_test)

precision, recall, _ = precision_recall_curve(y_test, y_score)
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()
plt.show(block=True)










