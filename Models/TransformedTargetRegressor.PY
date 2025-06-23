import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import TransformedTargetRegressor

# Load the data (adjust the path as per your environment)
df = pd.read_csv('Data for B - Glass Data-2.csv', header=1)

# Separate features (X) and target (y)
X = df.drop('V0_B_ou_r0_B', axis=1)
y = df['V0_B_ou_r0_B']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Random Forest model
rf = RandomForestRegressor(random_state=42)

# Wrap it in TransformedTargetRegressor for target transformation
tt = TransformedTargetRegressor(
    regressor=rf,
    func=np.log1p,       # Apply log(1 + y) to handle positive values
    inverse_func=np.expm1 # Inverse transform predictions back to original scale
)

# Define the hyperparameter grid (prefix with 'regressor__' for TransformedTargetRegressor)
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(
    tt,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# Evaluate on the test set
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test Mean Squared Error (MSE): {mse:.4f}")
print(f"Test R-squared (R²): {r2:.4f}")

# Feature importance (access the regressor inside TransformedTargetRegressor)
feature_importances = best_model.regressor_.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
print("\nFeature Importances:\n", importance_df.sort_values(by='Importance', ascending=False))