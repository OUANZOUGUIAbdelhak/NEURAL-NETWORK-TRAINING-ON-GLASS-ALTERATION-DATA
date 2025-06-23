import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from PIL import Image, ImageDraw, ImageFont

# --- Utility to save DataFrame as image ---
def df_to_image(df, path, font_size=16, padding=10):
    text = df.to_string()
    font = ImageFont.load_default()

    # Estimate size
    lines = text.split('\n')
    width = max([len(line) for line in lines]) * (font_size // 2)
    height = len(lines) * (font_size + 2)

    img = Image.new('RGB', (width + 2 * padding, height + 2 * padding), color='white')
    draw = ImageDraw.Draw(img)

    for i, line in enumerate(lines):
        draw.text((padding, padding + i * (font_size + 2)), line, fill='black', font=font)

    img.save(path)

# Configurations générales
sns.set(style="whitegrid")
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 100
os.makedirs("figures", exist_ok=True)  # dossier pour enregistrer les images

# Chargement de la base
df = pd.read_csv("/home/intra.cea.fr/ao280403/Bureau/ML Model/Data_for_B_positive.csv")
print("Shape :", df.shape)
print(df.head())

# ----------------------
# Aperçu statistique
# ----------------------
summary = df.describe().transpose()
summary.to_csv("figures/statistics_summary.csv")
df_to_image(summary.round(3), "figures/statistics_summary.png")
print(summary)

# ----------------------
# Distribution de la variable cible
# ----------------------
plt.figure(figsize=(6, 4))
sns.histplot(df['V0_B_ou_r0_B'], kde=True, bins=30, color="green")
plt.title("Distribution de la vitesse initiale d'altération (V0)")
plt.xlabel("Vitesse (V0)")
plt.ylabel("Densité")
plt.tight_layout()
plt.savefig("figures/hist_v0.png")
plt.close()

# ----------------------
# Corrélation (matrice)
# ----------------------
corr = df.corr(numeric_only=True)
plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap='coolwarm', center=0, annot=False, cbar=True)
plt.title("Matrice de corrélation")
plt.tight_layout()
plt.savefig("figures/correlation_matrix.png")
plt.close()

# ----------------------
# Corrélations avec la cible
# ----------------------
cor_target = corr["V0_B_ou_r0_B"].sort_values(ascending=False)
cor_target.to_csv("figures/correlations_with_target.csv")
df_to_image(cor_target.round(3), "figures/correlations_with_target.png")
print("\nTop corrélations avec V0 :")
print(cor_target.head(10))

# ----------------------
# Température et pH vs Vitesse
# ----------------------
fig1 = px.scatter(df, x="Température", y="V0_B_ou_r0_B", color="Ph_Final",
                  title="Influence de la température et pH sur V0")
fig1.write_image("figures/temp_vs_v0.png")

# ----------------------
# Éléments les plus présents (somme par colonne)
# ----------------------
compo_cols = df.columns.difference(['Température', 'Ph_Final', 'V0_B_ou_r0_B'])
total_elements = df[compo_cols].sum().sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 5))
sns.barplot(x=total_elements.index, y=total_elements.values, palette="Blues_d")
plt.title("Éléments les plus présents (en moyenne)")
plt.xticks(rotation=45)
plt.ylabel("Concentration moyenne")
plt.tight_layout()
plt.savefig("figures/top_elements.png")
plt.close()

# ----------------------
# Analyse pairplot (attention aux ressources si beaucoup de colonnes)
# ----------------------
cols_to_plot = ['V0_B_ou_r0_B', 'Température', 'Ph_Final'] + list(total_elements.head(5).index)
sns.pairplot(df[cols_to_plot], diag_kind="kde", corner=True)
plt.savefig("figures/pairplot_selected.png")
plt.close()

# ----------------------
# Détection multicolinéarité (corrélation > 0.95)
# ----------------------
high_corr_pairs = []
threshold = 0.95
for i in range(len(corr.columns)):
    for j in range(i):
        if abs(corr.iloc[i, j]) > threshold:
            col1 = corr.columns[i]
            col2 = corr.columns[j]
            high_corr_pairs.append((col1, col2, corr.iloc[i, j]))

high_corr_df = pd.DataFrame(high_corr_pairs, columns=["Var1", "Var2", "Correlation"])
high_corr_df.to_csv("figures/highly_correlated_pairs.csv", index=False)
df_to_image(high_corr_df.round(3), "figures/highly_correlated_pairs.png")

print("\nVariables très corrélées entre elles (r > 0.95) :")
print(high_corr_df)
