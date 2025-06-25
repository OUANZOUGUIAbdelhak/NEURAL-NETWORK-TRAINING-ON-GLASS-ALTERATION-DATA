import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (adjust path if needed)
df = pd.read_csv("/home/intra.cea.fr/ao280403/Bureau/ML Model/Data/Data Si ph V0.csv")  # Replace with your actual path if different

# Display basic info
print("🔹 Dataset Info:\n")
print(df.info())
print("\n🔹 First 5 rows:\n")
print(df.head())
print("\n🔹 Statistical Description:\n")
print(df.describe())
print("\n🔹 Missing Values:\n")
print(df.isnull().sum())

# 📊 Visualize the target distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["V0_Si_ou_r0_Si"], kde=True, bins=30, color='skyblue')
plt.title("Distribution of Target: V0_Si_ou_r0_Si")
plt.xlabel("V0_Si_ou_r0_Si")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("target_distribution.png")  # Saves the plot
print("\n✅ Target distribution plot saved as 'target_distribution.png'")

# 🔍 Correlation with target
correlations = df.corr(numeric_only=True)["V0_Si_ou_r0_Si"].sort_values(ascending=False)
print("\n🔹 Top features correlated with target:\n")
print(correlations.head(10))
print("\n🔹 Least correlated features with target:\n")
print(correlations.tail(10))
