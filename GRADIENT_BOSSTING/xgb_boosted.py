import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Fonction log1p signée pour valeurs négatives
def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))

def signed_exp1m(x):
    return np.sign(x) * (np.expm1(np.abs(x)))

# Charger les données
df = pd.read_csv("Data for B - Glass Data-2.csv")

# Afficher les valeurs manquantes par colonne
print(df.isnull().sum())

# Supprimer les lignes avec valeurs manquantes
df = df.dropna()

# Séparer les features et la cible
X = df.drop(columns=["V0_B_ou_r0_B"])
y = df["V0_B_ou_r0_B"]

# Appliquer la transformation logarithmique signée
y_log = signed_log1p(y)

# Diviser en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Initialiser le modèle XGBoost
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# Prédictions et retour à l'échelle d'origine
y_pred_log = model.predict(X_test)
y_pred = signed_exp1m(y_pred_log)
y_test_original = signed_exp1m(y_test)

# Évaluer le modèle
mse = mean_squared_error(y_test_original, y_pred)
r2 = r2_score(y_test_original, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Validation croisée
cv_scores = cross_val_score(model, X, y_log, cv=5, scoring='r2')
print(f"Cross-validated R² scores: {cv_scores}")
print(f"Average R²: {np.mean(cv_scores)}")

# Importance des features
importances = model.feature_importances_
feature_names = X.columns
top_features = sorted(zip(importances, feature_names), reverse=True)[:10]
print("Top 10 features:", top_features)

# Affichage des importances
plt.figure(figsize=(10, 6))
plt.barh([name for _, name in reversed(top_features)], [score for score, _ in reversed(top_features)])
plt.xlabel("Feature Importance")
plt.title("Top 10 Most Important Features")
plt.tight_layout()
plt.savefig("/home/intra.cea.fr/ao280403/Bureau/ML Model/GRADIENT_BOSSTING/feature_importance.png")

# Distribution de la cible
plt.figure(figsize=(8, 4))
sns.histplot(y, kde=True)
plt.title("Distribution de la cible V0_B_ou_r0_B")
plt.savefig("/home/intra.cea.fr/ao280403/Bureau/ML Model/GRADIENT_BOSSTING/target_distribution.png")

# Scatter plot des valeurs prédites vs réelles
plt.figure(figsize=(8, 8))
plt.scatter(y_test_original, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Valeurs Réelles")
plt.ylabel("Valeurs Prédites")
plt.title("Valeurs Prédites vs Réelles")
plt.grid(True)
plt.savefig("/home/intra.cea.fr/ao280403/Bureau/ML Model/GRADIENT_BOSSTING/predicted_vs_actual_values.png")
