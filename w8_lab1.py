# confusion matrix

# from sklearn import datasets
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

# PR curve

# from sklearn import svm
# import numpy as np
# from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
# import matplotlib.pyplot as plt
#
# iris = datasets.load_iris()
# x = iris.data
# y = iris.target
#
# # Add noise
# random_state = np.random.RandomState(0)
# n_samples, n_features = x.shape
# x = np.c_[x, random_state.randn(n_samples, 200*n_features)]
#
# # Limit to the two first classes (i.e. label=0 or 1), and split into training and test
# x_train, x_test, y_train, y_test = train_test_split(x[y < 2], y[y < 2], test_size=0.5, random_state=random_state)
#
# # Create a simple classifier
# clf = svm.LinearSVC(random_state=random_state)
# clf.fit(x_train, y_train)
# y_score = clf.decision_function(x_test)
#
# precision, recall, _ = precision_recall_curve(y_test, y_score)
# disp = PrecisionRecallDisplay(precision=precision, recall=recall)
# disp.plot()
# plt.show(block=True)


# ROC curve
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interpolate
from sklearn.metrics import roc_auc_score

# Import some data to play with
dataset = datasets.load_iris()
x = dataset.data
y = dataset.target

# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# Add noisy features (much more columns for each row) to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = x.shape
# print(x)
x = np.c_[x, random_state.randn(n_samples, 200*n_features)]
# print(x)

print(x.shape)
print(n_samples, n_features, 'randn:', random_state.randn(n_samples, 200*n_features).shape)

# shuffle and split training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=random_state)

# Learn to predict each class against the other
clf = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state))
y_socre = clf.fit(x_train, y_train).decision_function(x_test)

print(y_test)


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_socre[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), y_socre.ravel())
roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([[0, 0], [1, 1]], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc='lower right')
plt.show()













