import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load dataset
df = pd.read_csv("mushrooms.csv")

# Encode categorical columns
encoder = LabelEncoder()

for column in df.columns:
    df[column] = encoder.fit_transform(df[column])

# X and y
X = df.drop("class", axis=1)
y = df["class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ID3-style Decision Tree
model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:")
print(accuracy_score(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Precision, Recall, F1
print("\nClassification Report:")
print(classification_report(y_test, y_pred))