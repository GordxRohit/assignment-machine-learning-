import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv("Advertising.csv")

# X and y
X = df[["TV"]]
y = df["Sales"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("MSE:",
      mean_squared_error(y_test, y_pred))

print("R²:",
      r2_score(y_test, y_pred))

# Plot
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)

plt.xlabel("TV")
plt.ylabel("Sales")
plt.title("Simple Linear Regression")

plt.show()