import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

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

# === Incertitude relative ±30 % ===
# En log, ±30 % de la valeur réelle y_test → ±(0.3 * y_test)
# Incertitude en log (toujours positive)
# Calculer l'incertitude (±30% en échelle linéaire transformée en log)
# Incertitude de 30% en échelle log
uncertainty = np.log(1.3)  # ≈ 0.2624

plt.figure(figsize=(6, 6))
plt.errorbar(
    y_test,
    y_pred_nn,
    xerr=uncertainty,  # Barre sur les valeurs réelles
    fmt='o',
    ecolor='gray',
    alpha=0.7,
    capsize=3,
    label="Prédictions avec incertitude (30%)"
)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Prédiction parfaite')
plt.xlabel("Valeurs réelles Ln(V₀)")
plt.ylabel("Valeurs prédites Ln(V₀)")
plt.title("Prédictions vs Réel avec barres d’incertitude (30%)")
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.savefig("predicted_vs_actual_with_uncertainty.png", dpi=300)
plt.show()


