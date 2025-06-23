import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score

# Load data from CSV
df = pd.read_csv("/home/intra.cea.fr/ao280403/Bureau/ML Model/Data B Ph _ Less than 4 .csv")  # Replace with your path

# Check for missing values
print(df.isnull().sum())

# Separate features and target
X = df.drop(columns=["V0_B_ou_r0_B"])
y = df["V0_B_ou_r0_B"]

# Optional: scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=42)


# Initialize and train the model
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")


scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
print("Cross-validated R² scores:", scores)
print("Average R²:", scores.mean())

importances = model.feature_importances_
features = df.drop(columns=["V0_B_ou_r0_B"]).columns
important = sorted(zip(importances, features), reverse=True)
print("Top 10 features:", important[:10])


# Predicted vs Actual plot
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual V0_B_ou_r0_B")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Values")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.grid()
plt.show()
plt.savefig("actual_vs_predicted2.png")

