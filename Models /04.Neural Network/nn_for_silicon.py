import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# =======================
# 1. Load & preprocess data
# =======================
file_path = "/home/intra.cea.fr/ao280403/Bureau/ML Model/Data/Data Si ph LnV0.csv" 
df = pd.read_csv(file_path)

# Split features and target
X = df.drop(columns=["Ln(V0)"])
y = df["Ln(V0)"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =======================
# 2. Build & train the model
# =======================
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)  # Output layer for regression
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=16,
    verbose=0
)

# =======================
# 3. Predictions & metrics
# =======================
y_pred_nn = model.predict(X_test_scaled).flatten()

r2 = r2_score(y_test, y_pred_nn)
mae = mean_absolute_error(y_test, y_pred_nn)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_nn))

print("\n✅ Neural Network Regression Performance:")
print(f"  R² Score      : {r2:.4f}")
print(f"  MAE           : {mae:.4f}")
print(f"  RMSE          : {rmse:.4f}")

# =======================
# 4. Advanced visualization (Ln(V₀) of Silicon)
# =======================
sns.set_theme(style="whitegrid", font_scale=1.2)

fig = plt.figure(constrained_layout=True, figsize=(14, 6))
gs = GridSpec(2, 2, figure=fig)

# --- 1. Predicted vs Actual ---
ax1 = fig.add_subplot(gs[:, 0])
scatter = sns.scatterplot(
    x=y_test, y=y_pred_nn, ax=ax1,
    hue=np.abs(y_test - y_pred_nn),  # color by residual magnitude
    palette="viridis", s=70, edgecolor='k', alpha=0.85
)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', lw=2, label="Ideal Fit")
ax1.set_xlabel("Actual Ln(V₀) of Silicon", fontsize=14)
ax1.set_ylabel("Predicted Ln(V₀) of Silicon", fontsize=14)
ax1.set_title("Predicted vs Actual Values\n(Ln(V₀) of Silicon)", fontsize=16, weight='bold')

# Annotate metrics
metrics_text = f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}"
ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes,
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", lw=1.2))
ax1.legend()
cbar = fig.colorbar(scatter.collections[0], ax=ax1)
cbar.set_label("Residual Magnitude", rotation=270, labelpad=15)

# --- 2. Residual distribution ---
ax2 = fig.add_subplot(gs[0, 1])
residuals = y_test - y_pred_nn
sns.histplot(residuals, kde=True, ax=ax2, color="skyblue", edgecolor="black")
ax2.set_title("Residual Distribution\n(Ln(V₀) of Silicon)", fontsize=14, weight='bold')
ax2.set_xlabel("Residuals (Ln(V₀) of Silicon)")
ax2.axvline(0, color='red', linestyle='--', lw=1.5)

# --- 3. Training history ---
ax3 = fig.add_subplot(gs[1, 1])
epochs_range = range(1, len(history.history['loss']) + 1)
ax3.plot(epochs_range, history.history['loss'], label='Training Loss', lw=2)
ax3.plot(epochs_range, history.history['val_loss'], label='Validation Loss', lw=2)
ax3.set_title("Training & Validation Loss\n(Ln(V₀) of Silicon)", fontsize=14, weight='bold')
ax3.set_xlabel("Epochs")
ax3.set_ylabel("Loss (MSE)")
ax3.legend()

# Save high-resolution combined plot
plt.savefig("model_results_LnV0_Silicon.png", dpi=600, bbox_inches="tight")
plt.show()
