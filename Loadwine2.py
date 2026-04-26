# Step 1: Import libraries
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Step 2: Load dataset
wine = load_wine()
X = wine.data
y = wine.target

# Step 3: Show dataset (optional but useful for students)
df = pd.DataFrame(X, columns=wine.feature_names)
df['target'] = y

print("First 5 rows:\n", df.head())

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Initialize models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression(max_iter=2000)
}

# Step 6: Train and compare
print("\nModel Accuracies:\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"{name}: {acc:.2f}")