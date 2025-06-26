import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs("figures", exist_ok=True)

# === Chargement des données ===
file_path = "/home/intra.cea.fr/ao280403/Bureau/ML Model/Data/Data B Ph LnV0.csv"
df = pd.read_csv(file_path)

# === Préparation X / y ===
X = df.drop(columns=["Ln(V0)"])
y = df["Ln(V0)"]

# === Split train/test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# === Construction du réseau ===
model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1),
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train_scaled, y_train, validation_split=0.1, epochs=100, batch_size=16, verbose=0)

# === Prédiction & évaluation ===
y_pred_nn = model.predict(X_test_scaled).flatten()

r2 = r2_score(y_test, y_pred_nn)
mae = mean_absolute_error(y_test, y_pred_nn)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_nn))

print("\n✅ Neural Network Regression Performance:")
print(f"  R² Score      : {r2:.4f}")
print(f"  MAE           : {mae:.4f}")
print(f"  RMSE          : {rmse:.4f}")

# === Calcul des résidus ===
residuals = y_test - y_pred_nn

# Reconstruire X_test DataFrame avec index réinitialisé pour concaténer facilement
X_test_df = X_test.reset_index(drop=True)
y_test_df = y_test.reset_index(drop=True)

# Création DataFrame résidus + variables explicatives
residuals_df = pd.DataFrame({
    "residuals": residuals,
    "y_test": y_test_df
})

# Ajout des colonnes explicatives à residuals_df
for col in X_test_df.columns:
    residuals_df[col] = X_test_df[col]

# === Tracés résidus vs variables ===

# Liste des colonnes chimiques (à adapter selon tes colonnes)
composition_cols = [
    "Li", "B", "O", "Na", "Mg", "Al", "Si", "P", "K", "Ca", "Ti", "Cr",
    "Mn", "Fe", "Ni", "Zn", "Sr", "Y", "Zr", "Mo", "Ru", "Ag", "Cd",
    "Sn", "Sb", "Te", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "S_autres_TR",
    "Th", "U"
]

# Colonnes extra (avec bonne casse et accents)
extra_cols = ["Température", "ph"]

variables_a_tracer = composition_cols + extra_cols

# Vérifie la présence de chaque variable avant de tracer
variables_existantes = [col for col in variables_a_tracer if col in residuals_df.columns]

for col in variables_existantes:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=col, y="residuals", data=residuals_df)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f"Résidus vs {col}")
    plt.xlabel(col)
    plt.ylabel("Résidu (réel - prédit)")
    plt.tight_layout()
    plt.savefig(f"figures/plot_{col}.png", dpi=300)
    plt.close()

# === Corrélations entre résidus et variables explicatives ===
corrs = residuals_df[variables_existantes + ["residuals"]].corr()
corr_residuals = corrs["residuals"].drop("residuals")
print("\nCorrélations entre résidus et variables explicatives :")
print(corr_residuals.sort_values(ascending=False))

# Optionnel : heatmap des corrélations
plt.figure(figsize=(10, 8))
sns.heatmap(corrs, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matrice de corrélation (incluant les résidus)")
plt.tight_layout()
plt.savefig(f"figures/plot_{col}.png", dpi=300)
plt.close()
# === Heatmap des corrélations avec les résidus ===
plt.figure(figsize=(10, 12))
sns.heatmap(
    corrs.to_frame(),  # transforme en DataFrame pour compatibilité heatmap
    annot=True,
    cmap="coolwarm",
    center=0,
    cbar_kws={'label': 'Corrélation'},
    fmt=".2f"
)
plt.title("Corrélations entre les résidus et les variables explicatives")
plt.tight_layout()
plt.show()
