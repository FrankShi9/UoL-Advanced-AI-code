from sklearn import datasets
dataset = datasets.load_digits()
x = dataset.data
y = dataset.target

observations = len(x)
features = len(dataset.feature_names)
classes = len(dataset.target_names)
print("Number of Observations: " + str( observations ))
print("Number of Features: " + str(features))
print("Number of Classes: " + str(classes))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.linear_model import Perceptron

clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(x_train, y_train)
print("Training accuracy is %s"% clf.score(x_train ,y_train))
print("Test accuracy is %s"% clf.score(x_test ,y_test))

print("Labels of all instances:\n%s" % y_test)
y_pred = clf.predict(x_test)
print("Predictive outputs of all instances:\n%s" % y_pred)

from sklearn.metrics import classification_report, confusion_matrix
print("Confusion Matrix:\n%s"% confusion_matrix (y_test , y_pred))
print("Classification Report:\n%s"% classification_report (y_test , y_pred))
