import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb

# =======================
# 1. Load data
# =======================
file_path = "/home/intra.cea.fr/ao280403/Bureau/ML Model/Data/Data B Ph LnV0 with less param.csv"
df = pd.read_csv(file_path)

X = df.drop(columns=["Ln(V0)"])
y = df["Ln(V0)"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (for models that need it)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =======================
# 2. Define models
# =======================
models = {
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "SVR": SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
    "XGBoost": xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
}

# =======================
# 3. Train & evaluate models
# =======================
results = []

for name, model in models.items():
    if name == "SVR":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results.append({"Model": name, "R²": r2, "RMSE": rmse, "MAE": mae})

results_df = pd.DataFrame(results)
print(results_df)

# =======================
# 4. Modern Visualization with better colors & value labels
# =======================
sns.set_theme(style="whitegrid", font_scale=1.3)

# Rich and clear colors for publications
palette = ["#D62828", "#F77F00", "#003049"]  # Red, Orange, Navy

# Melt DataFrame for bar plot
results_melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Value")

plt.figure(figsize=(10, 6))
barplot = sns.barplot(
    data=results_melted,
    x="Model", y="Value", hue="Metric",
    palette=palette, edgecolor="black"
)

# Add value labels on bars
for p in barplot.patches:
    value = f"{p.get_height():.3f}"
    barplot.annotate(
        value,
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='bottom',
        fontsize=11, color='black', weight='bold',
        xytext=(0, 5), textcoords='offset points'
    )

plt.title("Machine Learning Model Comparison (Ln(V₀) of Boron)", fontsize=16, weight='bold')
plt.ylabel("Score / Error", fontsize=13)
plt.xlabel("")
plt.xticks(rotation=15)
plt.legend(title="Metric", fontsize=12, title_fontsize=13)
plt.tight_layout()
plt.savefig("model_comparison_LnV0_Boron.png", dpi=600, bbox_inches="tight")
plt.show()
