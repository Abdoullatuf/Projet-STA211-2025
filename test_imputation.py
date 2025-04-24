import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from exploratory_analysis import handle_missing_values, load_data

# Charger les données
print("Chargement des données...")
df = load_data('data_train.csv')

# Vérification des données
print("\nDimensions du DataFrame:", df.shape)
print("\nColonnes du DataFrame:", df.columns.tolist())
print("\nValeurs manquantes par colonne:")
print(df.isnull().sum())

# Sauvegarder une copie des données originales
df_original = df.copy()

# Appliquer la nouvelle stratégie d'imputation
print("\nApplication de la nouvelle stratégie d'imputation...")
df_imputed = handle_missing_values(df, strategy='advanced', display_info=True)

# Identifier les colonnes avec des valeurs manquantes
missing_cols = df.columns[df.isnull().sum() > 0].tolist()
print("\nColonnes avec valeurs manquantes:", missing_cols)

if len(missing_cols) == 0:
    print("\nAucune valeur manquante détectée dans le jeu de données!")
    exit()

# Comparaison des distributions avant/après imputation
print("\nCréation des visualisations...")
fig, axes = plt.subplots(len(missing_cols), 2, figsize=(15, 5*len(missing_cols)))
fig.suptitle('Comparaison des distributions avant/après imputation', fontsize=16, y=1.02)

for i, col in enumerate(missing_cols):
    # Distribution originale (avec valeurs manquantes)
    sns.histplot(df_original[col].dropna(), ax=axes[i,0], kde=True)
    axes[i,0].set_title(f'{col} - Original (sans NA)')
    axes[i,0].set_xlabel('Valeur')
    axes[i,0].set_ylabel('Fréquence')
    
    # Distribution après imputation
    sns.histplot(df_imputed[col], ax=axes[i,1], kde=True)
    axes[i,1].set_title(f'{col} - Après imputation')
    axes[i,1].set_xlabel('Valeur')
    axes[i,1].set_ylabel('Fréquence')

plt.tight_layout()
plt.savefig('distributions_imputation.png')
plt.close()

# Affichage des statistiques descriptives
print("\nStatistiques descriptives avant imputation:")
print(df_original[missing_cols].describe())
print("\nStatistiques descriptives après imputation:")
print(df_imputed[missing_cols].describe())

# Analyse des corrélations après imputation
corr_after = df_imputed[missing_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_after, annot=True, cmap='coolwarm', center=0)
plt.title('Matrice de corrélation après imputation')
plt.savefig('correlations_imputation.png')
plt.close()

# Sauvegarder les données imputées
df_imputed.to_csv('data_train_imputed.csv', index=False)
print("\nLes données imputées ont été sauvegardées dans 'data_train_imputed.csv'") 