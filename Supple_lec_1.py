from sklearn import datasets
dataset = datasets.load_digits()
X = dataset.data
y = dataset.target

print("===== Get Basic Information ======")
observation = len(X)
features = len(dataset.feature_names)
classes = len(dataset.target_names)
print("Number of Observations: " + str( observation))
print("Number of Features: " + str(features))
print("Number of Classes: " + str(classes))

print("===== Split Dataset ======")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("===== Model Training ======")
from sklearn.linear_model import Perceptron
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X_train, y_train)

print("===== Model Prediction ======")
print("Labels of all instances:\n%s" % y_test)
y_pred = clf.predict(X_test)
print("Predictive outputs of all instances:\n%s" % y_pred)

print("===== Confusion Matrix ======")
from sklearn.metrics import classification_report, confusion_matrix
print("Consusion Matrix:\n%s" % confusion_matrix(y_test, y_pred))
print("Classification Report:\n%s" % classification_report(y_test, y_pred))
