import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optional: Visual style
sns.set(style="whitegrid")

# 🔹 Load Data
df = pd.read_csv("/home/intra.cea.fr/ao280403/Bureau/ML Model/Data/Data B Poh _ Less than 4_ final.csv")  # Update with your actual file

# 🔹 Separate Features and Target
X = df.drop(columns="V0_B_ou_r0_B")  # Replace "target" with actual column name
y = df["V0_B_ou_r0_B"]

# 🔹 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Initialize and Fit Decision Tree
model = DecisionTreeRegressor(random_state=42, max_depth=5)
model.fit(X_train, y_train)

# 🔹 Predictions
y_pred = model.predict(X_test)

# 🔹 Evaluation Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("✅ Decision Tree Regressor Performance:")
print(f"  R² Score      : {r2:.4f}")
print(f"  MAE           : {mae:.4f}")
print(f"  RMSE          : {rmse:.4f}")

# 🔹 Feature Importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n🔹 Feature Importances:")
print(importances)

# 📊 Plot Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values, y=importances.index, palette="viridis")
plt.title("Feature Importances - Decision Tree")
plt.tight_layout()
plt.show()

# 📈 Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=residuals)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Actual Values")
plt.ylabel("Residuals")
plt.title("Residual Plot - Decision Tree")
plt.tight_layout()
plt.show()

# 📌 Optional: Target Distribution
plt.figure(figsize=(7, 5))
sns.histplot(y, bins=30, kde=True, color="steelblue")
plt.title("Target Variable Distribution")
plt.xlabel("Target")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
