import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data (replace with your actual file path)
df = pd.read_csv("Data for B - Glass Data(2).csv")

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Separate features and target
X = df.drop(columns=["V0_B_ou_r0_B"])
y = df["V0_B_ou_r0_B"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define GPR model with a composite kernel (RBF + WhiteKernel for noise)
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e1))
gpr = GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=10)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'kernel__k1__k1__constant_value': [0.1, 1.0, 10.0],
    'kernel__k1__k2__length_scale': [0.1, 1.0, 10.0],
    'kernel__k2__noise_level': [1e-10, 1e-5, 1.0]
}
grid_search = GridSearchCV(gpr, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_gpr = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)
print("Best cross-validated R²:", grid_search.best_score_)

# Predict on test set
y_pred = best_gpr.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test Mean Squared Error: {mse:.4f}")
print(f"Test R² Score: {r2:.4f}")

# Cross-validation scores
scores = cross_val_score(best_gpr, X_scaled, y, cv=5, scoring='r2')
print("Cross-validated R² scores:", scores)
print("Average R²:", scores.mean())

# Feature importance (approximated via permutation importance)
from sklearn.inspection import permutation_importance
perm_importance = permutation_importance(best_gpr, X_test, y_test, n_repeats=10, random_state=42)
features = df.drop(columns=["V0_B_ou_r0_B"]).columns
important = sorted(zip(perm_importance.importances_mean, features), reverse=True)
print("Top 10 features:", important[:10])

# Predicted vs Actual plot
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual V0_B_ou_r0_B")
plt.ylabel("Predicted V0_B_ou_r0_B")
plt.title("GPR: Actual vs Predicted Dissolution Rates")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.grid()
plt.savefig("gpr_actual_vs_predicted.png")
plt.show()