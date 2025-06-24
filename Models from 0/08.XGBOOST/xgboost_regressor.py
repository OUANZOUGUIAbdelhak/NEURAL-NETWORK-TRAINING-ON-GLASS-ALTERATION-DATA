import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset here
# Replace this with your actual dataset loading code
# Example: data = pd.read_csv('your_data.csv')
data = pd.read_csv('/home/intra.cea.fr/ao280403/Bureau/ML Model/Data/Data B Ph _ Less than 4_ final.csv')

# Define features and target
X = data.drop(columns=['V0_B_ou_r0_B'])  # Replace 'target_column' with your target column name
y = data['V0_B_ou_r0_B']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost regressor
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Calculate RMSE manually

print("✅ XGBoost Regressor Performance:")
print(f"  R² Score      : {r2:.4f}")
print(f"  MAE           : {mae:.4f}")
print(f"  RMSE          : {rmse:.4f}")

# Feature importance plotting
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 8))
sns.barplot(x=importances, y=features, palette="viridis")
plt.title("XGBoost Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
