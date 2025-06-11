import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------
# CONFIGURATION
# ------------------------
sns.set(style="whitegrid")
os.makedirs("figures", exist_ok=True)
RANDOM_STATE = 42

# ------------------------
# LOAD DATA
# ------------------------
df = pd.read_csv("/home/intra.cea.fr/ao280403/Bureau/ML Model/Data_Visualization/Data_for_B.csv")
print("Missing values:\n", df.isnull().sum())

X = df.drop(columns=["V0_B_ou_r0_B"])
y = df["V0_B_ou_r0_B"]
feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# ------------------------
# GPR MODEL + GRID SEARCH
# ------------------------
kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

param_grid = {
    "kernel__k1__k1__constant_value": [0.1, 1.0],
    "kernel__k1__k2__length_scale": [1.0, 10.0],
    "kernel__k2__noise_level": [1.0],
}

grid_search = GridSearchCV(gpr, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)
print("Best cross-validated R²:", grid_search.best_score_)

# ------------------------
# TEST SET PERFORMANCE
# ------------------------
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test Mean Squared Error: {mse:.4f}")
print(f"Test R² Score: {r2:.4f}")

# ------------------------
# CROSS-VALIDATION SCORES
# ------------------------
cv_results = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
print("Cross-validated R² scores:", cv_results)
print("Average R²:", np.mean(cv_results))

# ------------------------
# FEATURE IMPORTANCE (based on variance)
# ------------------------
X_std = (X - X.mean()) / X.std()
feature_std = X_std.std().values
feature_weights = np.abs(best_model.alpha_ @ X_train.to_numpy())  # shape (n_features,)

# Normalize importance
feature_importance = np.abs(np.mean(X_train * best_model.alpha_.reshape(-1, 1), axis=0))
feature_importance = pd.Series(feature_importance, index=feature_names)
top10 = feature_importance.abs().sort_values(ascending=False).head(10)
print("Top 10 features:")
for feat, val in top10.items():
    print(f"{feat}: {val:.4f}")

# Save feature importance as CSV + image
top10.to_csv("figures/top10_features.csv")

plt.figure(figsize=(8, 5))
sns.barplot(x=top10.values, y=top10.index, palette="viridis")
plt.xlabel("Importance")
plt.title("Top 10 des variables explicatives")
plt.tight_layout()
plt.savefig("figures/top10_features.png", dpi=300)
plt.close()

# ------------------------
# SAVE predicted vs actual
# ------------------------
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred, color='teal')
lims = [min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))]
plt.plot(lims, lims, 'k--', alpha=0.7)
plt.xlabel("Vitesse réelle")
plt.ylabel("Vitesse prédite")
plt.title("Actual vs. Predicted sur le test")
plt.tight_layout()
plt.savefig("figures/actual_vs_predicted.png", dpi=300)
plt.close()

# ------------------------
# SAVE boxplot of CV scores
# ------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(data=cv_results)
plt.scatter(np.zeros_like(cv_results), cv_results, color='red')
plt.ylabel("R²")
plt.title("Distribution des R² en Validation Croisée")
plt.tight_layout()
plt.savefig("figures/cv_r2_boxplot.png", dpi=300)
plt.close()

# ------------------------
# Save CV scores as image
# ------------------------
cv_df = pd.DataFrame(cv_results, columns=["R2"])
cv_df.to_csv("figures/cv_r2_scores.csv", index=False)

plt.figure(figsize=(4, 3))
sns.barplot(x=np.arange(1, len(cv_results) + 1), y=cv_results, palette="mako")
plt.xlabel("Fold")
plt.ylabel("R²")
plt.title("Scores R² par pli")
plt.tight_layout()
plt.savefig("figures/cv_r2_folds.png", dpi=300)
plt.close()

print("\n✅ Tous les graphiques et tableaux ont été enregistrés dans le dossier 'figures/'")
