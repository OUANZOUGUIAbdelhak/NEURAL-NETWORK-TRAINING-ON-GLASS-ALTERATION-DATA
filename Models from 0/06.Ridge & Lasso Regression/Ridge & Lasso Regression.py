import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
file_path = "/home/intra.cea.fr/ao280403/Bureau/ML Model/Data/Data B Ph _ Less than 4_ final.csv"  # <-- replace with actual path
df = pd.read_csv(file_path)

# Define features and target
X = df.drop(columns=["V0_B_ou_r0_B"])
y = df["V0_B_ou_r0_B"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

### ========== Ridge Regression ==========
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

print("✅ Ridge Regression Performance:")
print(f"  R² Score      : {r2_score(y_test, y_pred_ridge):.4f}")
print(f"  MAE           : {mean_absolute_error(y_test, y_pred_ridge):.4f}")
print(f"  RMSE          : {np.sqrt(mean_squared_error(y_test, y_pred_ridge)):.4f}\n")

# Coefficients
ridge_coefs = pd.Series(ridge.coef_, index=X.columns).sort_values(ascending=False)
print("🔹 Ridge Coefficients:\n", ridge_coefs)

# Plot residuals
plt.figure(figsize=(6, 4))
sns.residplot(x=y_test, y=y_pred_ridge, line_kws={"color": "red"})
plt.title("Ridge Residuals")
plt.xlabel("Actual")
plt.ylabel("Residuals")
plt.tight_layout()
plt.show()


### ========== Lasso Regression ==========
lasso = Lasso(alpha=0.1)  # You can tune alpha
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)

print("\n✅ Lasso Regression Performance:")
print(f"  R² Score      : {r2_score(y_test, y_pred_lasso):.4f}")
print(f"  MAE           : {mean_absolute_error(y_test, y_pred_lasso):.4f}")
print(f"  RMSE          : {np.sqrt(mean_squared_error(y_test, y_pred_lasso)):.4f}\n")

# Coefficients
lasso_coefs = pd.Series(lasso.coef_, index=X.columns).sort_values(ascending=False)
print("🔹 Lasso Coefficients:\n", lasso_coefs)

# Plot residuals
plt.figure(figsize=(6, 4))
sns.residplot(x=y_test, y=y_pred_lasso, line_kws={"color": "red"})
plt.title("Lasso Residuals")
plt.xlabel("Actual")
plt.ylabel("Residuals")
plt.tight_layout()
plt.show()
