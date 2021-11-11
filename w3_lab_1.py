from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

observations = len(X)
features = len(iris.feature_names)
classes = len(iris.target_names)

print("Number of Observations: " + str(observations))
print("Number of Features: " + str(features))
print("Number of Classes: "+ str(classes))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
print("Training accuracy: %s"% tree.score(X_train, y_train))
print("Test accuracy: %s"% tree.score(X_test, y_test))

print("Labels of all instances:\n %s"%y_test)
y_pred = tree.predict(X_test)
print("Predictive outputs of all instances:\n %s"%y_pred)

from sklearn.metrics import classification_report, confusion_matrix
print("Confusion Matrix:\n %s" %confusion_matrix(y_test, y_pred))
print("Classification Report:\n %s" %classification_report(y_test, y_pred))