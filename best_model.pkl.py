# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline

# 1. Load and inspect data
df = pd.read_csv('Data for B - Glass Data.csv')  # Replace with your file path

# 2. Handle missing values (if any)
print("Missing values:\n", df.isnull().sum())
# If missing values exist, consider using:
# df = df.dropna() or imputation

# 3. Remove constant features
selector = VarianceThreshold()
X = df.drop('V0_B_ou_r0_B', axis=1)
selector.fit(X)
constant_columns = [column for column in X.columns 
                    if column not in X.columns[selector.get_support()]]
df = df.drop(columns=constant_columns)

# 4. Prepare features and target
X = df.drop('V0_B_ou_r0_B', axis=1)
y = df['V0_B_ou_r0_B']

# 5. Check target distribution
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(y, bins=30)
plt.title('Target Distribution')

# Apply log transformation if skewed
if abs(y.skew()) > 1:
    y = np.log1p(y)
    plt.subplot(1, 2, 2)
    plt.hist(y, bins=30)
    plt.title('Transformed Target Distribution')
plt.tight_layout()
plt.show()

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Define models and preprocessing
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'SVR': Pipeline([('scaler', StandardScaler()), 
                    ('model', SVR())])
}

# 8. Model evaluation
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if abs(df['V0_B_ou_r0_B'].skew()) > 1:  # If we transformed y
        y_pred = np.expm1(y_pred)
        test_y = np.expm1(y_test)
    else:
        test_y = y_test
        
    rmse = np.sqrt(mean_squared_error(test_y, y_pred))
    mae = mean_absolute_error(test_y, y_pred)
    r2 = r2_score(test_y, y_pred)
    
    results[name] = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

# 9. Display results
results_df = pd.DataFrame(results).T
print("Model Performance Comparison:")
print(results_df.sort_values('RMSE'))

# 10. Hyperparameter tuning for best model
best_model_name = results_df['RMSE'].idxmin()
print(f"\nBest model: {best_model_name}")

# Example for XGBoost tuning
if best_model_name == 'XGBoost':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    grid_search = GridSearchCV(XGBRegressor(random_state=42), param_grid, cv=5,scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print("Best parameters:", grid_search.best_params_)

# 11. Final evaluation
y_pred = best_model.predict(X_test)

if abs(df['V0_B_ou_r0_B'].skew()) > 1:
    y_pred = np.expm1(y_pred)
    test_y = np.expm1(y_test)
else:
    test_y = y_test

print("\nFinal Model Performance:")
print(f"RMSE: {np.sqrt(mean_squared_error(test_y, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(test_y, y_pred):.4f}")
print(f"R²: {r2_score(test_y, y_pred):.4f}")

# 12. Save the model
import joblib
joblib.dump(best_model, 'best_model.pkl')