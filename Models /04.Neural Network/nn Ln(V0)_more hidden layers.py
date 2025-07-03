import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# === Chargement des données ===
file_path = "/home/intra.cea.fr/ao280403/Bureau/ML Model/Data/Data Si ph Ln(V0) 20.csv"
df = pd.read_csv(file_path)

# === X / y ===
X = df.drop(columns=["Ln(V0)"])
y = df["Ln(V0)"]

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# === Nouveau modèle plus profond ===
model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),   # 38 features
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1),
])

# === Compilation ===
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# === Early stopping pour éviter l’overfitting ===
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# === Entraînement ===
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.1,
    epochs=200,
    batch_size=16,
    callbacks=[early_stop],
    verbose=0
)

# === Prédictions ===
y_pred = model.predict(X_test_scaled).flatten()

# === Évaluation ===
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n✅ Performance du modèle (réseau profond) :")
print(f"  R² Score : {r2:.4f}")
print(f"  MAE      : {mae:.4f}")
print(f"  RMSE     : {rmse:.4f}")

# === Visualisation : prédiction vs réel ===
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Valeurs réelles Ln(V₀)")
plt.ylabel("Valeurs prédites Ln(V₀)")
plt.title("Prédiction vs Réel (modèle profond)")
plt.axis('equal')
plt.tight_layout()
plt.show()
