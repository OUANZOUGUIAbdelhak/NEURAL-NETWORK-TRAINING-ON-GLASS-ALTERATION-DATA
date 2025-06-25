# Random Forest Regressor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load your dataset
data = pd.read_csv("/home/intra.cea.fr/ao280403/Bureau/ML Model/Data/Data Si ph LnV0.csv")  # Replace with your file

# Split features and target
X = data.drop("Ln(V0)", axis=1)  # Replace "target" with your actual target column name
y = data["Ln(V0)"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("✅ Random Forest Regressor Performance:")
print(f"  R² Score      : {r2:.4f}")
print(f"  MAE           : {mae:.4f}")
print(f"  RMSE          : {rmse:.4f}")

# Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\n🔹 Feature Importances:")
print(importances)

# Feature importance plot
plt.figure(figsize=(10, 8))
sns.barplot(x=importances.values, y=importances.index, palette="viridis")
plt.title("Feature Importances - Random Forest")
plt.tight_layout()
plt.savefig("rf_feature_importance.png")
plt.close()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Actual")
plt.ylabel("Residuals")
plt.title("Residual Plot - Random Forest")
plt.tight_layout()
plt.savefig("rf_residual_plot.png")
plt.close()
