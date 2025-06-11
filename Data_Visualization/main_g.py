import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données (change 'glass_data.csv' par le nom de ton fichier CSV)
df = pd.read_csv('/home/intra.cea.fr/ao280403/Bureau/ML Model/Data_Visualization/Data_for_B.csv')

# Liste des éléments chimiques (suppose que les trois dernières colonnes sont 'Température', 'Ph_Final', 'V0_B_ou_r0_B')
element_columns = df.columns[:-3]

# Fonction pour tracer le pourcentage d'échantillons avec des valeurs non nulles pour chaque élément
def plot_nonzero_percentages(df, element_columns):
    nonzero_percentages = (df[element_columns] != 0).mean() * 100
    plt.figure(figsize=(12, 6))
    nonzero_percentages.plot(kind='bar')
    plt.title('Pourcentage d\'échantillons avec des valeurs non nulles pour chaque élément')
    plt.xlabel('Élément chimique')
    plt.ylabel('Pourcentage (%)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('element_nonzero_percentages.png')
    # plt.show()  # Décommenter pour afficher le graphique

# Fonction pour tracer les distributions des éléments présents dans au moins 20% des échantillons
def plot_element_distributions(df, element_columns, threshold=20):
    nonzero_percentages = (df[element_columns] != 0).mean() * 100
    frequent_elements = nonzero_percentages[nonzero_percentages >= threshold].index
    num_elements = len(frequent_elements)
    if num_elements > 0:
        rows = int(np.ceil(num_elements / 5))
        plt.figure(figsize=(15, 3 * rows))
        for i, element in enumerate(frequent_elements, 1):
            plt.subplot(rows, 5, i)
            sns.boxplot(y=df[element])
            plt.title(element)
        plt.tight_layout()
        plt.savefig('element_distributions.png')
        # plt.show()  # Décommenter pour afficher le graphique

# Fonction pour tracer le nuage de points de V0_B_ou_r0_B vs Température, coloré par Ph_Final
def plot_scatter_temperature_ph(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Température', y='V0_B_ou_r0_B', hue='Ph_Final', palette='viridis')
    plt.title('Vitesse d\'altération vs Température, coloré par pH')
    plt.xlabel('Température')
    plt.ylabel('V0_B_ou_r0_B')
    plt.legend(title='pH Final')
    plt.tight_layout()
    plt.savefig('scatter_temperature_ph.png')
    # plt.show()  # Décommenter pour afficher le graphique

# Fonction pour tracer la vitesse d'altération moyenne par tranches de pH
def plot_avg_alteration_by_ph(df):
    # Définir les tranches de pH
    df['Ph_Category'] = pd.cut(df['Ph_Final'], bins=[0, 8, 10, 14], labels=['pH < 8', '8 ≤ pH < 10', 'pH ≥ 10'])
    avg_alteration = df.groupby('Ph_Category')['V0_B_ou_r0_B'].mean().reset_index()
    plt.figure(figsize=(8, 6))
    sns.barplot(data=avg_alteration, x='Ph_Category', y='V0_B_ou_r0_B', palette='Blues_d')
    plt.title('Vitesse d\'altération moyenne par tranches de pH')
    plt.xlabel('Tranche de pH')
    plt.ylabel('V0_B_ou_r0_B moyen')
    plt.tight_layout()
    plt.savefig('avg_alteration_by_ph.png')
    # plt.show()  # Décommenter pour afficher le graphique

# Fonction pour tracer la matrice de corrélation
def plot_correlation_heatmap(df):
    # Sélectionner uniquement les colonnes numériques
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, linewidths=0.5)
    plt.title('Matrice de corrélation')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    # plt.show()  # Décommenter pour afficher le graphique

# Appeler les fonctions pour générer les graphiques
plot_nonzero_percentages(df, element_columns)
plot_element_distributions(df, element_columns, threshold=20)
plot_scatter_temperature_ph(df)
plot_avg_alteration_by_ph(df)
plot_correlation_heatmap(df)

print("Les graphiques ont été sauvegardés sous forme de fichiers PNG.")