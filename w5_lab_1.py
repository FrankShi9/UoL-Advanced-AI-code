from sklearn import datasets
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data[:100], iris.target[:100], test_size=0.2)

# Logistic Reg
print("Logistic")
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression(solver='lbfgs', max_iter=10000)
reg.fit(X_train, y_train)
print("Training accuracy is %s" % reg.score(X_train, y_train))
print("Test accuracy is %s" % reg.score(X_test, y_test))

print("Labels of all instances:\n%s" % y_test)
y_pred = reg.predict(X_test)
print("Predictive outputs of all instances:\n%s" % y_pred)

from sklearn.metrics import classification_report, confusion_matrix
print("Confusion Matrix:\n%s" % confusion_matrix(y_test, y_pred))
print("Classification Report:\n%s" % classification_report(y_test, y_pred))






# Naive Bayes
print("Naive Bayes")
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print("Training accuracy is %s" % gnb.score(X_train, y_train))
print("Test accuracy is %s" % gnb.score(X_test, y_test))

print("Labels of all instances:\n%s" % y_test)
y_pred = gnb.predict(X_test)
print("Predictive outputs of all instances:\n%s" % y_pred)

from sklearn.metrics import classification_report, confusion_matrix
print("Confusion Matrix:\n%s" % confusion_matrix(y_test, y_pred))
print("Classification Report:\n%s" % classification_report(y_test, y_pred))


