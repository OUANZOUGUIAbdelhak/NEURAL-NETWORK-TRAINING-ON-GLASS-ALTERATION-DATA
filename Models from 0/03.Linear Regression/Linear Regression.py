import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 🔹 Load your data (replace with correct path)
df = pd.read_csv("/home/intra.cea.fr/ao280403/Bureau/ML Model/Data/Data B Poh _ Less than 4_ final.csv")  # e.g., "data.csv"

# 🔹 Features and Target
X = df.drop(columns=["V0_B_ou_r0_B"])
y = df["V0_B_ou_r0_B"]

# 🔹 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔹 Feature scaling (important for regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🔹 Train Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 🔹 Predict
y_pred = model.predict(X_test_scaled)

# 🔹 Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"✅ Linear Regression Performance:")
print(f"  R² Score      : {r2:.4f}")
print(f"  MAE           : {mae:.4f}")
print(f"  RMSE          : {rmse:.4f}")

# 🔹 Coefficients
coeffs = pd.Series(model.coef_, index=X.columns)
print("\n🔹 Feature Coefficients (Impact on Target):")
print(coeffs.sort_values(ascending=False))

# 🔹 Residual Plot
plt.figure(figsize=(8, 5))
sns.residplot(x=y_test, y=y_pred, lowess=True, line_kws={'color': 'red'})
plt.title("Residual Plot")
plt.xlabel("True Values")
plt.ylabel("Residuals")
plt.tight_layout()
plt.savefig("residuals.png")
print("📊 Residual plot saved as 'residuals.png'")

# 🔹 Predicted vs Actual
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # perfect prediction line
plt.title("Predicted vs Actual")
plt.xlabel("Actual V0_B_ou_r0_B")
plt.ylabel("Predicted V0_B_ou_r0_B")
plt.tight_layout()
plt.savefig("pred_vs_actual.png")
print("📊 Predicted vs Actual plot saved as 'pred_vs_actual.png'")
