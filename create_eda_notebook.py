import nbformat as nbf
nb = nbf.v4.new_notebook()

# Title and Introduction
nb.cells.append(nbf.v4.new_markdown_cell("""# Analyse Exploratoire et Prétraitement des Données

Ce notebook se concentre sur l'analyse exploratoire et le prétraitement des données pour le projet STA211 de prédiction de publicités.

## Table des Matières
1. [Chargement et Aperçu des Données](#chargement)
2. [Gestion des Valeurs Manquantes](#valeurs-manquantes)
3. [Analyse Univariée](#analyse-univariee)
4. [Analyse Bivariée](#analyse-bivariee)
5. [Analyse Multivariée](#analyse-multivariee)
6. [Réduction de Dimensionnalité](#reduction-dim)
7. [Prétraitement Final](#pretraitement)
8. [Export des Données](#export)
"""))

# Import Libraries
nb.cells.append(nbf.v4.new_code_cell("""# Import des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
import prince  # Pour l'analyse factorielle multiple
import umap
import warnings

# Import des modules personnalisés
from data_preprocessing import load_data, analyze_missing_values, handle_missing_values
from exploratory_analysis import (univariate_analysis, bivariate_analysis, 
                                multivariate_analysis, dimension_reduction,
                                umap_visualization, analyze_feature_importance,
                                compare_visualization_methods, enhance_features)

# Configuration des visualisations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')
%matplotlib inline
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
"""))

# Data Loading
nb.cells.append(nbf.v4.new_markdown_cell("""## 1. Chargement et Aperçu des Données <a id='chargement'></a>"""))
nb.cells.append(nbf.v4.new_code_cell("""# Chargement des données avec la fonction dédiée
df = load_data('data_train.csv')

# Affichage des informations de base
print("Dimensions du dataset:", df.shape)
print("\\nAperçu des premières lignes:")
display(df.head())

print("\\nInformations sur les colonnes:")
display(df.info())

print("\\nStatistiques descriptives:")
display(df.describe())

# Analyse initiale des valeurs manquantes
missing_analysis = analyze_missing_values(df)
print("\\nAnalyse des valeurs manquantes effectuée.")
"""))

# Missing Values Analysis
nb.cells.append(nbf.v4.new_markdown_cell("""## 2. Gestion des Valeurs Manquantes <a id='valeurs-manquantes'></a>"""))
nb.cells.append(nbf.v4.new_code_cell("""# Analyse des valeurs manquantes
missing_values = df.isnull().sum()
missing_percentages = (missing_values / len(df)) * 100

# Affichage des colonnes avec des valeurs manquantes
missing_info = pd.DataFrame({
    'Valeurs manquantes': missing_values,
    'Pourcentage (%)': missing_percentages
}).query('`Valeurs manquantes` > 0')

print("Analyse des valeurs manquantes:")
display(missing_info)

# Visualisation des valeurs manquantes
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
plt.title('Distribution des valeurs manquantes')
plt.show()

# Application de la fonction handle_missing_values
df_imputed = handle_missing_values(df.copy(), strategy='advanced', display_info=True)
"""))

# Univariate Analysis
nb.cells.append(nbf.v4.new_markdown_cell("""## 3. Analyse Univariée <a id='analyse-univariee'></a>"""))
nb.cells.append(nbf.v4.new_code_cell("""# Analyse de la distribution de la variable cible
plt.figure(figsize=(10, 5))
sns.countplot(data=df_imputed, x='outcome')
plt.title('Distribution de la variable cible')
plt.show()

# Analyse univariée des variables numériques
univariate_analysis(df_imputed)

# Distribution des variables numériques
numeric_cols = df_imputed.select_dtypes(include=['float64', 'int64']).columns
n_cols = len(numeric_cols)
n_rows = (n_cols + 2) // 3

plt.figure(figsize=(15, 5*n_rows))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(n_rows, 3, i)
    sns.histplot(data=df_imputed, x=col, kde=True)
    plt.title(f'Distribution de {col}')
plt.tight_layout()
plt.show()
"""))

# Bivariate Analysis
nb.cells.append(nbf.v4.new_markdown_cell("""## 4. Analyse Bivariée <a id='analyse-bivariee'></a>"""))
nb.cells.append(nbf.v4.new_code_cell("""# Analyse des corrélations avec la variable cible
correlations, high_corr_pairs = bivariate_analysis(df_imputed)

# Matrice de corrélation
plt.figure(figsize=(12, 8))
numeric_data = df_imputed.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Matrice de corrélation')
plt.tight_layout()
plt.show()
"""))

# Multivariate Analysis
nb.cells.append(nbf.v4.new_markdown_cell("""## 5. Analyse Multivariée <a id='analyse-multivariee'></a>"""))
nb.cells.append(nbf.v4.new_code_cell("""# Analyse multivariée
high_corr_pairs = multivariate_analysis(df_imputed)

# Visualisation UMAP
umap_visualization(df_imputed)

# Analyse factorielle multiple (AFM)
numeric_cols = df_imputed.select_dtypes(include=['float64', 'int64']).columns
groups = [numeric_cols.tolist()]
mfa = prince.MFA(n_components=2, groups=groups, random_state=42)
mfa_coords = mfa.fit_transform(df_imputed[numeric_cols])

plt.figure(figsize=(10, 6))
plt.scatter(mfa_coords[0], mfa_coords[1], alpha=0.5)
plt.title('Analyse Factorielle Multiple')
plt.xlabel('Première composante')
plt.ylabel('Deuxième composante')
plt.show()
"""))

# Dimension Reduction
nb.cells.append(nbf.v4.new_markdown_cell("""## 6. Réduction de Dimensionnalité <a id='reduction-dim'></a>"""))
nb.cells.append(nbf.v4.new_code_cell("""# Réduction de dimensionnalité avec PCA et analyse des features importantes
X_final, pca_model, selected_features = dimension_reduction(df_imputed, display_info=True)

# Analyse de l'importance des features
importance_results = analyze_feature_importance(df_imputed)

# Comparaison des méthodes de visualisation
compare_visualization_methods(df_imputed)
"""))

# Final Preprocessing
nb.cells.append(nbf.v4.new_markdown_cell("""## 7. Prétraitement Final <a id='pretraitement'></a>"""))
nb.cells.append(nbf.v4.new_code_cell("""# Sélection des features importantes
top_features = importance_results.nlargest(20, 'Combined_Score')['feature'].tolist()
df_final = df_imputed[top_features + ['outcome']].copy()

# Création de features polynomiales et d'interaction
df_enhanced = enhance_features(df_final, top_features)

print("Dimensions finales du dataset:", df_enhanced.shape)
print("\\nAperçu des nouvelles features:")
display(df_enhanced.head())
"""))

# Export Data
nb.cells.append(nbf.v4.new_markdown_cell("""## 8. Export des Données <a id='export'></a>"""))
nb.cells.append(nbf.v4.new_code_cell("""# Sauvegarde des données prétraitées
df_enhanced.to_csv('data_train_processed.csv', index=False)
print("Données prétraitées sauvegardées dans 'data_train_processed.csv'")
"""))

# Save the notebook
with open('01_EDA_Preprocessing.ipynb', 'w') as f:
    nbf.write(nb, f) 