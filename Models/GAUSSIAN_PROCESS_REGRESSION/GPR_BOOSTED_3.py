import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# Charger les données
df = pd.read_csv("/home/intra.cea.fr/ao280403/Bureau/ML Model/Data B Ph.csv")

# Vérifier les valeurs manquantes
print("Missing values:\n", df.isnull().sum())

# Inspecter la variable cible
print("Target variable statistics:")
print(df["V0_B_ou_r0_B"].describe())
print("Negative values in target:", (df["V0_B_ou_r0_B"] < 0).sum())
print("Zero values in target:", (df["V0_B_ou_r0_B"] == 0).sum())

# Nettoyer la variable cible (supprimer les valeurs négatives ou non finies)
df = df[df["V0_B_ou_r0_B"] >= 0]  # Garder seulement les valeurs non négatives
df = df[np.isfinite(df["V0_B_ou_r0_B"])]  # Supprimer les NaN ou inf

# Sélectionner les features importantes
important_features = ['Température', 'Ph_Final', 'Li', 'B', 'O', 'Na', 'Mg', 'Al', 'Si']
X = df[important_features]
y = df["V0_B_ou_r0_B"]

# Log-transformer la cible avec un petit décalage pour éviter les problèmes
y_log = np.log1p(y + 1e-6)  # Ajouter une petite constante pour gérer les zéros

# Vérifier si y_log contient des NaN ou inf
print("NaN in y_log:", np.isnan(y_log).sum())
print("Inf in y_log:", np.isinf(y_log).sum())

# Supprimer les lignes où y_log est NaN ou inf
valid_indices = np.isfinite(y_log)
X = X[valid_indices]
y = y[valid_indices]
y_log = y_log[valid_indices]

# Mise à l'échelle des caractéristiques
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Division train/test
X_train, X_test, y_train_log, y_test_log = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)

# --- Modèle GPR ---
# Définir le noyau pour GPR
kernel = (
    ConstantKernel(1.0, (1e-3, 1e3)) *
    RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
    WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e1))
)

# Initialiser le modèle
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, random_state=42, n_restarts_optimizer=10)

# Recherche de grille
param_grid = {
    'kernel__k1__k1__constant_value': [0.01, 0.1, 1.0, 10.0, 100.0],
    'kernel__k1__k2__length_scale': [0.01, 0.1, 1.0, 10.0, 100.0],
    'kernel__k2__noise_level': [1e-10, 1e-5, 1e-3, 1e-1, 1.0]
}

grid_search = GridSearchCV(gpr, param_grid, cv=5, scoring='r2', n_jobs=-1, error_score='raise')
grid_search.fit(X_train, y_train_log)

# Meilleur modèle GPR
best_gpr = grid_search.best_estimator_
print("Best GPR parameters:", grid_search.best_params_)
print("Best cross-validated R² (GPR):", grid_search.best_score_)

# Prédictions sur le jeu de test
y_pred_log = best_gpr.predict(X_test)
y_pred = np.expm1(y_pred_log)  # Inverser la transformation
y_test = np.expm1(y_test_log)

# Évaluation GPR
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"GPR Test Mean Squared Error: {mse:.4f}")
print(f"GPR Test R² Score: {r2:.4f}")

# Cross-validation globale
scores = cross_val_score(best_gpr, X_scaled, y_log, cv=5, scoring='r2')
print("GPR Cross-validated R² scores:", scores)
print("GPR Average R²:", scores.mean())

# Importance des variables (GPR)
perm_importance = permutation_importance(best_gpr, X_test, y_test_log, n_repeats=10, random_state=42)
print("GPR Feature Importance:")
for imp, feat in sorted(zip(perm_importance.importances_mean, important_features), reverse=True):
    print(f"{feat}: {imp:.4f}")

# --- Modèle Random Forest ---
rf = RandomForestRegressor(random_state=42)
param_grid_rf = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='r2', n_jobs=-1)
grid_search_rf.fit(X_train, y_train_log)

# Meilleur modèle RF
best_rf = grid_search_rf.best_estimator_
print("Best RF parameters:", grid_search_rf.best_params_)
print("Best cross-validated R² (RF):", grid_search_rf.best_score_)

# Prédictions RF
y_pred_rf_log = best_rf.predict(X_test)
y_pred_rf = np.expm1(y_pred_rf_log)

# Évaluation RF
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"RF Test Mean Squared Error: {mse_rf:.4f}")
print(f"RF Test R² Score: {r2_rf:.4f}")

# Importance des variables (RF)
feature_importance_rf = pd.Series(best_rf.feature_importances_, index=important_features)
print("Random Forest Feature Importance:")
print(feature_importance_rf.sort_values(ascending=False))

# Graphique prédictions vs réels (GPR)
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual V0_B_ou_r0_B")
plt.ylabel("Predicted V0_B_ou_r0_B")
plt.title("GPR: Actual vs Predicted Dissolution Rates")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.grid()
plt.savefig("gpr_actual_vs_predicted.png")

# Graphique prédictions vs réels (RF)
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred_rf)
plt.xlabel("Actual V0_B_ou_r0_B")
plt.ylabel("Predicted V0_B_ou_r0_B")
plt.title("Random Forest: Actual vs Predicted Dissolution Rates")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.grid()
plt.savefig("rf_actual_vs_predicted_3.png")

# Ensemble prediction
y_pred_ensemble = (y_pred + y_pred_rf) / 2
print("Ensemble Test R²:", r2_score(y_test, y_pred_ensemble))