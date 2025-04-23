import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from scipy import stats
from sklearn.impute import SimpleImputer, KNNImputer

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_missing_values(data):
    """
    Analyse les valeurs manquantes dans le jeu de données.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Le jeu de données à analyser
        
    Returns:
    --------
    dict
        Un dictionnaire contenant les résultats de l'analyse
    """
    logger.info("Début de l'analyse des valeurs manquantes")
    
    # Calcul des valeurs manquantes
    missing_values = data.isnull().sum()
    missing_percentage = (missing_values / len(data)) * 100
    
    # Création d'un DataFrame pour les résultats
    missing_analysis = pd.DataFrame({
        'Missing_Count': missing_values,
        'Missing_Percentage': missing_percentage
    })
    
    # Filtrage pour ne garder que les colonnes avec des valeurs manquantes
    missing_analysis = missing_analysis[missing_analysis['Missing_Count'] > 0]
    
    # Analyse du mécanisme des valeurs manquantes
    # Vérification de la corrélation entre les valeurs manquantes
    missing_corr = data.isnull().corr()
    
    # Visualisation
    plt.figure(figsize=(12, 6))
    
    # Graphique des valeurs manquantes
    plt.subplot(1, 2, 1)
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    plt.title('Carte des valeurs manquantes')
    
    # Graphique des pourcentages
    plt.subplot(1, 2, 2)
    missing_analysis['Missing_Percentage'].sort_values(ascending=False).plot(
        kind='bar', color='red'
    )
    plt.title('Pourcentage de valeurs manquantes par colonne')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Résultats de l'analyse
    results = {
        'missing_summary': missing_analysis,
        'missing_correlation': missing_corr,
        'total_missing': missing_values.sum(),
        'missing_percentage_total': (missing_values.sum() / (data.shape[0] * data.shape[1])) * 100
    }
    
    # Affichage des résultats
    print("\n=== ANALYSE DES VALEURS MANQUANTES ===\n")
    print(f"Nombre total de valeurs manquantes : {results['total_missing']}")
    print(f"Pourcentage total de valeurs manquantes : {results['missing_percentage_total']:.2f}%")
    print("\nDétail par colonne :")
    print(missing_analysis)
    
    return results

def analyze_class_distribution(data, target_column='outcome'):
    """
    Analyse la distribution des classes dans la variable cible.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Le jeu de données
    target_column : str
        Le nom de la colonne cible
        
    Returns:
    --------
    dict
        Un dictionnaire contenant les résultats de l'analyse
    """
    logger.info("Début de l'analyse de la distribution des classes")
    
    # Distribution des classes
    class_distribution = data[target_column].value_counts()
    class_percentage = (class_distribution / len(data)) * 100
    
    # Création d'un DataFrame pour les résultats
    distribution_analysis = pd.DataFrame({
        'Count': class_distribution,
        'Percentage': class_percentage
    })
    
    # Visualisation
    plt.figure(figsize=(12, 5))
    
    # Graphique des effectifs
    plt.subplot(1, 2, 1)
    sns.countplot(x=target_column, data=data)
    plt.title('Distribution des classes')
    
    # Graphique des pourcentages
    plt.subplot(1, 2, 2)
    distribution_analysis['Percentage'].plot(kind='pie', autopct='%1.1f%%')
    plt.title('Pourcentage des classes')
    plt.tight_layout()
    plt.show()
    
    # Résultats de l'analyse
    results = {
        'distribution': distribution_analysis,
        'imbalance_ratio': class_distribution.max() / class_distribution.min(),
        'total_samples': len(data)
    }
    
    # Affichage des résultats
    print("\n=== ANALYSE DE LA DISTRIBUTION DES CLASSES ===\n")
    print(f"Nombre total d'échantillons : {results['total_samples']}")
    print(f"Ratio de déséquilibre : {results['imbalance_ratio']:.2f}")
    print("\nDistribution des classes :")
    print(distribution_analysis)
    
    return results

def analyze_data(data, target_column='outcome'):
    """
    Effectue une analyse complète des données.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Le jeu de données
    target_column : str
        Le nom de la colonne cible
        
    Returns:
    --------
    dict
        Un dictionnaire contenant tous les résultats d'analyse
    """
    logger.info("Début de l'analyse complète des données")
    
    # Analyse des valeurs manquantes
    missing_results = analyze_missing_values(data)
    
    # Analyse de la distribution des classes
    distribution_results = analyze_class_distribution(data, target_column)
    
    # Résumé statistique des variables numériques
    numeric_summary = data.select_dtypes(include=['float64', 'int64']).describe()
    
    # Résumé des variables catégorielles
    categorical_summary = data.select_dtypes(include=['object', 'category']).describe()
    
    # Résultats complets
    results = {
        'missing_analysis': missing_results,
        'class_distribution': distribution_results,
        'numeric_summary': numeric_summary,
        'categorical_summary': categorical_summary
    }
    
    return results 