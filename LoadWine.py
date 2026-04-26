# Step 1: Import libraries
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Step 2: Load dataset
wine = load_wine()

X = wine.data
y = wine.target

# Step 3: Show dataset details
print("Feature Names:")
print(wine.feature_names)

print("\nTarget Names:")
print(wine.target_names)

# Convert to DataFrame for better visualization
df = pd.DataFrame(X, columns=wine.feature_names)
df['target'] = y

print("\nFirst 5 rows of dataset:")
print(df.head())

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Create model
model = DecisionTreeClassifier()

# Step 6: Train
model.fit(X_train, y_train)

# Step 7: Predict
y_pred = model.predict(X_test)

# Step 8: Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))