import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
file_path = "/home/intra.cea.fr/ao280403/Bureau/ML Model/Data/Data B Ph LnV0.csv"  # Replace with actual path
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

# Build the neural network
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train_scaled, y_train, validation_split=0.1, epochs=100, batch_size=16, verbose=0)

# Predict
y_pred_nn = model.predict(X_test_scaled).flatten()

# Evaluation
r2 = r2_score(y_test, y_pred_nn)
mae = mean_absolute_error(y_test, y_pred_nn)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_nn))

print("\n✅ Neural Network Regression Performance:")
print(f"  R² Score      : {r2:.4f}")
print(f"  MAE           : {mae:.4f}")
print(f"  RMSE          : {rmse:.4f}")

# Plot residuals
plt.figure(figsize=(6, 4))
sns.residplot(x=y_test, y=y_pred_nn, line_kws={"color": "red"})
plt.title("Neural Network Residuals")
plt.xlabel("Actual")
plt.ylabel("Residuals")
plt.tight_layout()
plt.show()
