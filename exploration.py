import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Configuration de l'affichage
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
plt.style.use('seaborn')
sns.set_palette("husl")

def load_data(file_path):
    """Charge les données depuis un fichier CSV"""
    try:
        df = pd.read_csv(file_path)
        print(f"Données chargées avec succès. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        return None

def basic_info(df):
    """Affiche les informations de base sur le dataframe"""
    print("\n=== INFORMATIONS DE BASE ===")
    print("\nPremières lignes:")
    print(df.head())
    
    print("\nTypes de données:")
    print(df.dtypes)
    
    print("\nStatistiques descriptives:")
    print(df.describe())
    
    print("\nValeurs manquantes par colonne:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

def analyze_categorical_variables(df):
    """Analyse les variables catégorielles"""
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print("\n=== VARIABLES CATÉGORIELLES ===")
        for col in categorical_cols:
            print(f"\nDistribution de {col}:")
            print(df[col].value_counts(normalize=True).head())
            print(f"Nombre de catégories uniques: {df[col].nunique()}")

def analyze_numerical_variables(df):
    """Analyse les variables numériques"""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_cols) > 0:
        print("\n=== VARIABLES NUMÉRIQUES ===")
        for col in numerical_cols:
            print(f"\nStatistiques pour {col}:")
            print(df[col].describe())
            
            # Création d'un histogramme
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x=col, kde=True)
            plt.title(f'Distribution de {col}')
            plt.savefig(f'hist_{col}.png')
            plt.close()

def correlation_analysis(df):
    """Analyse les corrélations entre variables numériques"""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_cols) > 1:
        print("\n=== ANALYSE DES CORRÉLATIONS ===")
        corr_matrix = df[numerical_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Matrice de corrélation')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
        plt.close()

def main():
    # Chargement des données
    train_df = load_data('data_train.csv')
    
    if train_df is not None:
        # Analyse de base
        basic_info(train_df)
        
        # Analyse des variables catégorielles
        analyze_categorical_variables(train_df)
        
        # Analyse des variables numériques
        analyze_numerical_variables(train_df)
        
        # Analyse des corrélations
        correlation_analysis(train_df)

if __name__ == "__main__":
    main() 