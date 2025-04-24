# data_analysis_utils.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Optional

class DataAnalyzer:
    """Classe pour l'analyse exploratoire des données."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialise l'analyseur de données.
        
        Args:
            df (pd.DataFrame): DataFrame à analyser
        """
        self.df = df
        self.top_20_vars = ['X2', 'X1244', 'X352', 'X1400', 'X184', 'X969', 'X1456',
                           'X1436', 'X1230', 'X1345', 'X1144', 'X1154', 'X1155',
                           'X1423', 'X1425', 'X1048', 'X1199', 'X1395', 'X1219', 'X1119']
        self.top_4_vars = ['X2', 'X1244', 'X352', 'X1400']
        
    def analyze_univariate(self, target_col: str = 'outcome') -> Dict:
        """
        Réalise l'analyse univariée des données.
        
        Args:
            target_col (str): Nom de la colonne cible
            
        Returns:
            Dict: Résultats de l'analyse univariée
        """
        results = {}
        
        # Distribution de la variable cible
        target_dist = self.df[target_col].value_counts()
        target_pct = self.df[target_col].value_counts(normalize=True) * 100
        
        results['target_distribution'] = {
            'counts': target_dist,
            'percentages': target_pct
        }
        
        # Valeurs manquantes
        missing_values = self.df.isnull().sum()
        missing_pct = (missing_values / len(self.df)) * 100
        
        results['missing_values'] = {
            'counts': missing_values,
            'percentages': missing_pct
        }
        
        return results
    
    def analyze_correlations(self, variables: Optional[List[str]] = None,
                           threshold: float = 0.5) -> Tuple[pd.DataFrame, List]:
        """
        Analyse les corrélations entre variables.
        
        Args:
            variables (List[str]): Liste des variables à analyser
            threshold (float): Seuil de corrélation
            
        Returns:
            Tuple[pd.DataFrame, List]: Matrice de corrélation et liste des corrélations fortes
        """
        if variables is None:
            variables = self.top_20_vars
            
        corr_matrix = self.df[variables].corr()
        
        # Visualisation
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix,
                   annot=True,
                   cmap='coolwarm',
                   center=0,
                   fmt='.2f',
                   square=True)
        plt.title('Matrice de Corrélation des Variables')
        plt.tight_layout()
        plt.show()
        
        # Identification des corrélations fortes
        high_corr = np.where(np.abs(corr_matrix) > threshold)
        high_corr_list = [(variables[i], variables[j], corr_matrix.iloc[i,j])
                         for i, j in zip(*high_corr) if i < j]
        
        return corr_matrix, high_corr_list
    
    def analyze_distributions(self, variables: Optional[List[str]] = None,
                            target_col: str = 'outcome') -> None:
        """
        Analyse la distribution des variables.
        
        Args:
            variables (List[str]): Liste des variables à analyser
            target_col (str): Nom de la colonne cible
        """
        if variables is None:
            variables = self.top_4_vars
            
        n_rows = (len(variables) + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5*n_rows))
        axes = axes.ravel()
        
        for idx, var in enumerate(variables):
            sns.histplot(data=self.df, x=var, hue=target_col, ax=axes[idx])
            axes[idx].set_title(f'Distribution de {var}')
        
        for idx in range(len(variables), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Statistiques descriptives
        print("\nStatistiques descriptives:")
        print(self.df[variables].describe())
    
    def create_polynomial_features(self, variables: Optional[List[str]] = None,
                                 degree: int = 2) -> pd.DataFrame:
        """
        Crée des features polynomiales.
        
        Args:
            variables (List[str]): Liste des variables
            degree (int): Degré maximum du polynôme
            
        Returns:
            pd.DataFrame: DataFrame avec les nouvelles features
        """
        if variables is None:
            variables = self.top_4_vars
            
        new_features = {}
        for var in variables:
            for d in range(2, degree + 1):
                new_features[f"{var}_pow{d}"] = self.df[var] ** d
        
        return pd.DataFrame(new_features)
    
    def create_interaction_features(self, variables: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Crée des features d'interaction.
        
        Args:
            variables (List[str]): Liste des variables
            
        Returns:
            pd.DataFrame: DataFrame avec les nouvelles features
        """
        if variables is None:
            variables = self.top_4_vars
            
        new_features = {}
        for i, var1 in enumerate(variables):
            for var2 in variables[i+1:]:
                new_features[f"{var1}_{var2}_interact"] = self.df[var1] * self.df[var2]
        
        return pd.DataFrame(new_features)
    
    def enhance_features(self) -> pd.DataFrame:
        """
        Crée toutes les nouvelles features et les combine.
        
        Returns:
            pd.DataFrame: DataFrame avec toutes les features
        """
        poly_features = self.create_polynomial_features()
        interact_features = self.create_interaction_features()
        
        enhanced_df = pd.concat([self.df, poly_features, interact_features], axis=1)
        return enhanced_df
    
    def analyze_feature_importance(self, enhanced_df: pd.DataFrame,
                                 target_col: str = 'outcome') -> pd.DataFrame:
        """
        Analyse l'importance des features.
        
        Args:
            enhanced_df (pd.DataFrame): DataFrame avec les features augmentées
            target_col (str): Nom de la colonne cible
            
        Returns:
            pd.DataFrame: DataFrame avec les importances des features
        """
        correlations = enhanced_df.corr()[target_col].sort_values(ascending=False)
        importance_df = pd.DataFrame({
            'feature': correlations.index,
            'correlation': correlations.values
        })
        
        return importance_df 