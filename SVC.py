from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))