from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

print("Training accuracy is %s"% knn.score(X_train ,y_train))
print("Test accuracy is %s"% knn.score(X_test ,y_test))

print("Labels of all instances:\n%s" % y_test)
y_pred = knn.predict(X_test)
print("Predictive outputs of all instances:\n%s" % y_pred)

from sklearn.metrics import classification_report, confusion_matrix
print("Confusion Matrix:\n%s"% confusion_matrix(y_test, y_pred))
print("Classification Report:\n%s"%classification_report(y_test, y_pred))

