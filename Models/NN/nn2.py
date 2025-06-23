import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------
# CONFIGURATION
# ------------------------
RANDOM_STATE = 42
os.makedirs("figures", exist_ok=True)

# ------------------------
# 1. CHARGEMENT ET PRÉTRAITEMENT
# ------------------------
# Charge le CSV
df = pd.read_csv("/home/intra.cea.fr/ao280403/Bureau/ML Model/Data B Ph _ Less than 4.csv")

# Gère les valeurs manquantes
df.fillna(df.mean(), inplace=True)

# Sépare features et cible
X = df.drop(columns=["V0_B_ou_r0_B"])  # si vous voulez exclure pH, sinon gardez-le
y = df["V0_B_ou_r0_B"]

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=RANDOM_STATE
)

# ------------------------
# 2. DÉFINITION DU MODÈLE
# ------------------------
model = Sequential([
    Dense(64, 
          input_dim=X_train.shape[1], 
          activation='relu', 
          kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(32, 
          activation='relu', 
          kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(1, activation='linear')
])

model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=[]
)

# Early stopping : arrête si la val_loss n'améliore pas pendant 15 epochs
es = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# ------------------------
# 3. ENTRAÎNEMENT
# ------------------------
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=16,
    callbacks=[es],
    verbose=1
)

# ------------------------
# 4. ÉVALUATION SUR TEST
# ------------------------
y_pred = model.predict(X_test).ravel()
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nTest Mean Squared Error: {mse:.4f}")
print(f"Test R² Score: {r2:.4f}")

# ------------------------
# 5. VISUALISATION & SAUVEGARDE
# ------------------------
# Courbes de perte
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.savefig("figures/mlp_loss_curves.png", dpi=300)
plt.close()

# Scatter actual vs predicted
plt.figure(figsize=(6,6))
lims = [min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))]
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot(lims, lims, 'k--', linewidth=1)
plt.xlabel("Vitesse réelle")
plt.ylabel("Vitesse prédite")
plt.title("Actual vs Predicted")
plt.tight_layout()
plt.savefig("figures/mlp_actual_vs_pred.png", dpi=300)
plt.close()

print("\n✅ Les graphes ont été enregistrés dans le dossier 'figures/'")
