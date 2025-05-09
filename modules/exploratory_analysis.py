"""
Module d'analyse exploratoire pour le projet STA211.
Ce module contient des fonctions pour analyser les données du dataset Internet Advertisements.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency
import prince  # Si tu fais AFM / MCA


warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

__all__ = [
    'univariate_analysis',
    'bivariate_analysis',
    'multivariate_analysis',
    'dimension_reduction',
    'umap_visualization',
    'summary_statistics',
    'advanced_dimension_reduction',
    'perform_exploratory_analysis',
    'compare_visualization_methods',
    'analyze_feature_importance',
    'create_polynomial_features',
    'create_interaction_features',
    'enhance_features',
    'optimize_hyperparameters',
    'evaluate_optimized_models',
    'analyze_categorical_binaries_vs_target',
    'save_fig'
]



def save_fig(fname: str, directory: str = None, dpi: int = 150, figsize=(5, 3.5), **kwargs):
    """
    Sauvegarde la figure matplotlib courante dans directory/fname avec taille personnalisable.
    
    - figsize : tuple (largeur, hauteur) en pouces
    - dpi : résolution
    """
    if directory is None:
        directory = os.getenv('FIGURES_DIR', os.getcwd())

    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, fname)

    # Appliquer la taille de figure
    fig = plt.gcf()
    fig.set_size_inches(figsize)

    plt.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
    plt.show()

    
def univariate_analysis(data):
    """Effectue l'analyse univariée des données."""
    print("\n=== Analyse Univariée ===")
    
    # Distribution de la variable cible
    target_dist = data['outcome'].value_counts()
    print("\nDistribution de la variable cible :")
    print(target_dist)
    print(f"\nPourcentage de la classe majoritaire : {(target_dist.max() / len(data)) * 100:.2f}%")
    
    # Statistiques descriptives des variables numériques
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    print("\nStatistiques descriptives des variables numériques :")
    print(data[numeric_cols].describe())
    
    # Valeurs manquantes
    missing_values = data.isnull().sum()
    if missing_values.any():
        print("\nValeurs manquantes par colonne :")
        print(missing_values[missing_values > 0])

def bivariate_analysis(data, display_correlations=True):
    """
    Effectue l'analyse bivariée des données.
    
    Args:
        data (pd.DataFrame): DataFrame contenant les données
        display_correlations (bool): Si True, affiche les corrélations, sinon les retourne
        
    Returns:
        tuple: (DataFrame des corrélations avec la cible, liste des paires de variables fortement corrélées)
    """
    print("\n=== Analyse Bivariée ===")
    
    # Encoder la variable cible (0 pour 'noad.' et 1 pour 'ad.')
    target_numeric = (data['outcome'] == 'ad.').astype(int)
    
    # Sélectionner uniquement les colonnes numériques
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    numeric_data = data[numeric_cols]
    
    # Calculer les corrélations avec la variable cible
    correlations = pd.DataFrame()
    correlations['feature'] = numeric_cols
    correlations['correlation'] = [numeric_data[col].corr(target_numeric) for col in numeric_cols]
    correlations = correlations.sort_values('correlation', key=abs, ascending=False)
    
    if display_correlations:
        print("\nTop 10 variables les plus corrélées avec la variable cible :")
        print(correlations.head(10))
    
    # Identifier les paires de variables hautement corrélées
    corr_matrix = numeric_data.corr()
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i,j]) > 0.8:  # Seuil de corrélation fixé à 0.8
                high_corr_pairs.append((corr_matrix.columns[i], 
                                      corr_matrix.columns[j], 
                                      corr_matrix.iloc[i,j]))
    
    if display_correlations and high_corr_pairs:
        print("\nPaires de variables fortement corrélées (|corr| > 0.8):")
        for var1, var2, corr in high_corr_pairs:
            print(f"{var1} - {var2}: {corr:.3f}")
    
    return correlations, high_corr_pairs


def analyze_categorical_binaries_vs_target(data, target_col='outcome', show_top=20, pval_threshold=0.05):
    """
    Analyse les relations entre les variables binaires (0/1) et une variable cible catégorielle.

    Affiche les p-values du test du chi^2 pour chaque variable binaire vs. la cible,
    et sélectionne celles en-dessous du seuil pval_threshold.

    Args:
        data (pd.DataFrame): le DataFrame à analyser.
        target_col (str): nom de la variable cible.
        show_top (int): nombre de variables les plus significatives à afficher.
        pval_threshold (float): seuil de significativité des p-values.

    Returns:
        pd.DataFrame: tableau des variables sélectionnées triées par p-value croissante.
    """
    print("\n=== Analyse Bivariée : Variables Binaires vs Cible Catégorielle ===")

    binary_cols = [col for col in data.columns 
                   if col != target_col and data[col].dropna().nunique() == 2]

    results = []

    for col in binary_cols:
        contingency = pd.crosstab(data[col], data[target_col])
        if contingency.shape[0] == 2 and contingency.shape[1] > 1:
            chi2, p, dof, expected = chi2_contingency(contingency)
            if p < pval_threshold:
                results.append({'variable': col, 'p_value': p, 'chi2': chi2})

    results_df = pd.DataFrame(results).sort_values("p_value")

    print(f"\nTop {show_top} variables binaires avec p-value < {pval_threshold} :")
    print(results_df.head(show_top))

    return results_df

def multivariate_analysis(data):
    """
    Effectue l'analyse multivariée des données.
    
    Args:
        data (pd.DataFrame): DataFrame contenant les données
        
    Returns:
        list: Liste des paires de variables fortement corrélées (var1, var2, corr)
    """
    print("\n=== Analyse Multivariée ===")
    try:
        # Encoder la variable cible
        target_numeric = (data['outcome'] == 'ad.').astype(int)
        
        # Sélectionner uniquement les colonnes numériques (sauf 'outcome')
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        numeric_cols = [col for col in numeric_cols if col != 'outcome']
        
        # Créer un DataFrame avec les variables numériques et la cible encodée
        analysis_data = data[numeric_cols].copy()
        analysis_data['target'] = target_numeric
        
        # Calculer la matrice de corrélation
        corr_matrix = analysis_data.corr()
        
        # Trouver les paires de variables avec une forte corrélation (> 0.8)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i,j]) > 0.8:
                    high_corr_pairs.append((corr_matrix.columns[i], 
                                          corr_matrix.columns[j], 
                                          corr_matrix.iloc[i,j]))
        
        # Trier par valeur absolue de corrélation
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return high_corr_pairs
            
    except Exception as e:
        print(f"Erreur lors de l'analyse multivariée : {str(e)}")
        return []

def dimension_reduction(data, correlation_threshold=0.8, variance_threshold=0.95, display_info=False):
    """
    Réduit la dimensionnalité des données en utilisant la corrélation et l'ACP.
    
    Args:
        data (pd.DataFrame): DataFrame contenant les données
        correlation_threshold (float): Seuil de corrélation pour éliminer les variables redondantes
        variance_threshold (float): Seuil de variance expliquée pour l'ACP
        display_info (bool): Si True, affiche les informations détaillées
        
    Returns:
        tuple: (DataFrame réduit, modèle ACP, liste des variables conservées)
    """
    if display_info:
        print("\n=== Réduction de Dimensionnalité ===")
    
    # 1. Préparation des données
    # Séparer la variable cible
    target = data['outcome'].copy()
    target_numeric = (target == 'ad.').astype(int)
    
    # Sélectionner les variables numériques
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = [col for col in numeric_cols if col != 'outcome']
    X = data[numeric_cols].copy()
    
    # Gérer les valeurs manquantes
    if display_info:
        print("\nGestion des valeurs manquantes:")
        missing_counts = X.isnull().sum()
        print(f"Nombre de colonnes avec valeurs manquantes: {(missing_counts > 0).sum()}")
        print(f"Pourcentage moyen de valeurs manquantes: {missing_counts.mean()*100:.2f}%")
    
    # Imputer les valeurs manquantes par la moyenne
    X = X.fillna(X.mean())
    
    # 2. Élimination des variables fortement corrélées
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Identifier les colonnes à supprimer
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    
    # Garder les variables les plus corrélées avec la cible
    correlations_with_target = pd.Series({
        col: X[col].corr(target_numeric) for col in to_drop
    }).abs()
    
    # Trier les variables par corrélation avec la cible
    to_drop = [col for col in correlations_with_target.sort_values(ascending=True).index]
    
    if display_info:
        print(f"\nNombre de variables éliminées par corrélation : {len(to_drop)}")
        print(f"Variables restantes : {len(X.columns) - len(to_drop)}")
    
    # Créer le DataFrame réduit
    X_reduced = X.drop(columns=to_drop)
    
    # 3. Application de l'ACP
    # Standardisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)
    
    # Appliquer l'ACP
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculer la variance expliquée cumulée
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance_ratio >= variance_threshold) + 1
    
    if display_info:
        print(f"\nNombre de composantes principales retenues : {n_components}")
        print(f"Variance expliquée : {cumulative_variance_ratio[n_components-1]*100:.2f}%")
    
    # Créer le DataFrame final avec les composantes principales
    pca_reduced = PCA(n_components=n_components)
    X_pca_reduced = pca_reduced.fit_transform(X_scaled)
    
    # Créer les noms des colonnes pour les composantes principales
    pca_cols = [f"PC{i+1}" for i in range(n_components)]
    X_final = pd.DataFrame(X_pca_reduced, columns=pca_cols, index=X.index)
    
    # Ajouter la variable cible
    X_final['outcome'] = target
    
    # 4. Afficher les composantes principales et leur importance
    if display_info:
        print("\nImportance des composantes principales :")
        for i in range(n_components):
            print(f"PC{i+1}: {pca_reduced.explained_variance_ratio_[i]*100:.2f}%")
    
    # 5. Visualisation des deux premières composantes principales
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca_reduced[:, 0], X_pca_reduced[:, 1], 
                         c=target_numeric, cmap='viridis', alpha=0.6)
    plt.xlabel(f"Première composante principale ({pca_reduced.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"Deuxième composante principale ({pca_reduced.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("Projection des données sur les deux premières composantes principales")
    plt.colorbar(scatter, label='Classe (0: non-pub, 1: pub)')
    plt.tight_layout()
    plt.show()
    
    return X_final, pca_reduced, X_reduced.columns.tolist()

def umap_visualization(df, target_col='outcome'):
    """
    Visualise les données avec UMAP en optimisant les paramètres pour une meilleure séparation des classes.
    """
    try:
        # Copie du DataFrame pour éviter de modifier l'original
        df_clean = df.copy()
        
        # Encodage de la variable cible
        df_clean[target_col] = df_clean[target_col].map({'ad.': 1, 'noad.': 0})
        
        # Sélection des colonnes numériques
        numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
        X = df_clean[numeric_cols]
        
        # Analyse des valeurs manquantes
        missing_values = X.isnull().sum()
        print("\n=== Analyse des valeurs manquantes ===")
        print("Nombre de valeurs manquantes par colonne :")
        print(missing_values[missing_values > 0])
        
        # Imputation adaptative
        for col in X.columns:
            missing_pct = missing_values[col] / len(X) * 100
            if missing_pct > 30:
                # Suppression des variables avec plus de 30% de valeurs manquantes
                X = X.drop(columns=[col])
            elif missing_pct > 5:
                # Imputation par k-NN pour les variables avec 5-30% de valeurs manquantes
                imputer = KNNImputer(n_neighbors=5)
                X[col] = imputer.fit_transform(X[[col]])
            else:
                # Imputation par médiane pour les variables avec moins de 5% de valeurs manquantes
                imputer = SimpleImputer(strategy='median')
                X[col] = imputer.fit_transform(X[[col]])
        
        # Standardisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Optimisation des paramètres UMAP
        n_neighbors = min(50, len(X_scaled) // 10)  # Ajustement dynamique du nombre de voisins
        min_dist = 0.1  # Distance minimale entre les points
        
        # Application de UMAP
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='euclidean',
            random_state=42
        )
        embedding = reducer.fit_transform(X_scaled)
        
        # Création du graphique
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=df_clean[target_col],
            cmap=plt.cm.viridis,
            alpha=0.6,
            s=20
        )
        
        # Ajout des centroids des classes
        for label in [0, 1]:
            mask = df_clean[target_col] == label
            centroid = np.mean(embedding[mask], axis=0)
            plt.scatter(
                centroid[0],
                centroid[1],
                c='red' if label == 1 else 'blue',
                marker='*',
                s=200,
                edgecolor='black'
            )
        
        # Calcul des métriques de qualité
        silhouette = silhouette_score(embedding, df_clean[target_col])
        calinski = calinski_harabasz_score(embedding, df_clean[target_col])
        
        plt.title(f'Visualisation UMAP (Silhouette: {silhouette:.3f}, Calinski-Harabasz: {calinski:.3f})')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.colorbar(scatter, label='Classe (0: noad., 1: ad.)')
        
        # Ajout d'une légende pour les centroids
        plt.legend(
            handles=[
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='blue', markersize=10, label='Centroid noad.'),
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=10, label='Centroid ad.')
            ],
            loc='upper right'
        )
        
        plt.tight_layout()
        plt.show()
        
        # Analyse de la séparation des classes
        print("\n=== Analyse de la séparation des classes ===")
        print(f"Score de silhouette : {silhouette:.3f}")
        print(f"Score de Calinski-Harabasz : {calinski:.3f}")
        
        if silhouette > 0.5:
            print("Bonne séparation des classes")
        elif silhouette > 0.25:
            print("Séparation modérée des classes")
        else:
            print("Faible séparation des classes")
            
    except Exception as e:
        print(f"Erreur lors de la visualisation UMAP : {str(e)}")

def summary_statistics(data, pca, target_col='outcome'):
    """Affiche un résumé des statistiques et résultats."""
    print("\n=== Résumé des Résultats ===")
    print(f"Nombre total d'observations : {len(data)}")
    print(f"Nombre de variables : {data.shape[1]}")
    
    if target_col and target_col in data.columns:
        print(f"Distribution des classes :\n{data[target_col].value_counts(normalize=True)}")
    else:
        print(f"Attention : La colonne cible '{target_col}' n'a pas été trouvée dans le DataFrame.")
    
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    print(f"Nombre de variables numériques : {len(numeric_cols)}")
    print(f"Pourcentage de valeurs manquantes : {data[numeric_cols].isnull().mean().mean()*100:.2f}%")
    
    if pca is not None:
        print("\n=== Analyse des Composantes Principales ===")
        print(f"Variance expliquée par les deux premières composantes : {pca.explained_variance_ratio_.sum()*100:.2f}%")
        print(f"Variance expliquée par la première composante : {pca.explained_variance_ratio_[0]*100:.2f}%")
        print(f"Variance expliquée par la deuxième composante : {pca.explained_variance_ratio_[1]*100:.2f}%")

def advanced_dimension_reduction(data, n_features_to_select=50):
    """
    Applique des techniques avancées de réduction de dimensionnalité et de sélection de variables.
    
    Args:
        data (pd.DataFrame): DataFrame contenant les données
        n_features_to_select (int): Nombre de variables à sélectionner avec Random Forest
        
    Returns:
        tuple: (DataFrame avec variables sélectionnées, modèles de réduction, visualisations)
    """
    print("\n=== Réduction de Dimensionnalité Avancée ===")
    
    # 1. Préparation des données
    target = data['outcome'].copy()
    target_numeric = (target == 'ad.').astype(int)
    
    # Sélectionner les variables numériques
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = [col for col in numeric_cols if col != 'outcome']
    X = data[numeric_cols].copy()
    
    # Imputer les valeurs manquantes
    X = X.fillna(X.mean())
    
    # Standardisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 2. Sélection de variables avec Random Forest
    print("\nSélection de variables avec Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, target_numeric)
    
    # Sélectionner les variables les plus importantes
    selector = SelectFromModel(rf, max_features=n_features_to_select, prefit=True)
    feature_mask = selector.get_support()
    selected_features = X.columns[feature_mask].tolist()
    
    print(f"Nombre de variables sélectionnées : {len(selected_features)}")
    
    # Afficher les 10 variables les plus importantes
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 variables les plus importantes :")
    print(feature_importance.head(10))
    
    # 3. Réduction avec UMAP
    print("\nApplication de UMAP...")
    reducer_umap = umap.UMAP(random_state=42)
    X_umap = reducer_umap.fit_transform(X_scaled)
    
    # 4. Réduction avec t-SNE
    print("\nApplication de t-SNE...")
    reducer_tsne = TSNE(n_components=2, random_state=42)
    X_tsne = reducer_tsne.fit_transform(X_scaled)
    
    # 5. Visualisations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # UMAP
    scatter1 = ax1.scatter(X_umap[:, 0], X_umap[:, 1], 
                          c=target_numeric, cmap='viridis', alpha=0.6)
    ax1.set_title("Projection UMAP")
    ax1.set_xlabel("UMAP1")
    ax1.set_ylabel("UMAP2")
    plt.colorbar(scatter1, ax=ax1, label='Classe (0: non-pub, 1: pub)')
    
    # t-SNE
    scatter2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                          c=target_numeric, cmap='viridis', alpha=0.6)
    ax2.set_title("Projection t-SNE")
    ax2.set_xlabel("t-SNE1")
    ax2.set_ylabel("t-SNE2")
    plt.colorbar(scatter2, ax=ax2, label='Classe (0: non-pub, 1: pub)')
    
    plt.tight_layout()
    plt.show()
    
    # 6. Créer le DataFrame final avec les variables sélectionnées
    X_selected = X_scaled_df[selected_features].copy()
    X_selected['outcome'] = target
    
    # 7. Analyse des clusters naturels avec UMAP
    print("\nAnalyse des clusters naturels...")
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_umap)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], 
                         c=clusters, cmap='viridis', alpha=0.6)
    plt.title("Clusters naturels identifiés par UMAP")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.colorbar(scatter, label='Cluster')
    plt.show()
    
    # 8. Distribution des classes dans chaque cluster
    cluster_dist = pd.DataFrame({
        'cluster': clusters,
        'class': target_numeric
    }).groupby('cluster')['class'].value_counts(normalize=True).unstack()
    
    print("\nDistribution des classes par cluster :")
    print(cluster_dist)
    
    return X_selected, {
        'umap': reducer_umap,
        'tsne': reducer_tsne,
        'rf': rf,
        'selected_features': selected_features
    }

def perform_exploratory_analysis(data, target_col='outcome'):
    """
    Effectue une analyse exploratoire complète des données.
    
    Args:
        data (pd.DataFrame): DataFrame contenant les données à analyser
        target_col (str, optional): Nom de la colonne cible. Defaults to 'outcome'.
    """
    print("=== Début de l'analyse exploratoire ===\n")
    
    # Analyses existantes
    univariate_analysis(data)
    correlations, high_corr_pairs = bivariate_analysis(data)
    multivariate_analysis(data)
    
    # Réduction de dimension et visualisations
    X_final, pca_model, selected_features = dimension_reduction(data)
    umap_visualization(data, target_col)  # Ajout de la visualisation UMAP
    
    # Résumé des statistiques
    summary_statistics(data, pca_model, target_col)
    
    print("\n=== Fin de l'analyse exploratoire ===")
    return X_final, selected_features

def compare_visualization_methods(df, target_col='outcome'):
    """
    Compare différentes méthodes de visualisation (UMAP, t-SNE, PCA) et leurs performances.
    """
    try:
        # Préparation des données
        df_clean = df.copy()
        df_clean[target_col] = df_clean[target_col].map({'ad.': 1, 'noad.': 0})
        numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
        X = df_clean[numeric_cols]
        
        # Gestion des valeurs manquantes
        missing_values = X.isnull().sum()
        for col in X.columns:
            missing_pct = missing_values[col] / len(X) * 100
            if missing_pct > 30:
                X = X.drop(columns=[col])
            elif missing_pct > 5:
                imputer = KNNImputer(n_neighbors=5)
                X[col] = imputer.fit_transform(X[[col]])
            else:
                imputer = SimpleImputer(strategy='median')
                X[col] = imputer.fit_transform(X[[col]])
        
        # Standardisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Création de la figure
        plt.figure(figsize=(18, 6))
        
        # 1. UMAP
        plt.subplot(131)
        reducer = umap.UMAP(n_neighbors=min(50, len(X_scaled) // 10), min_dist=0.1)
        embedding = reducer.fit_transform(X_scaled)
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=df_clean[target_col], cmap=plt.cm.viridis, alpha=0.6)
        plt.title(f'UMAP (Silhouette: {silhouette_score(embedding, df_clean[target_col]):.3f})')
        plt.colorbar(scatter)
        
        # 2. t-SNE
        plt.subplot(132)
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        embedding_tsne = tsne.fit_transform(X_scaled)
        scatter = plt.scatter(embedding_tsne[:, 0], embedding_tsne[:, 1], c=df_clean[target_col], cmap=plt.cm.viridis, alpha=0.6)
        plt.title(f't-SNE (Silhouette: {silhouette_score(embedding_tsne, df_clean[target_col]):.3f})')
        plt.colorbar(scatter)
        
        # 3. PCA
        plt.subplot(133)
        pca = PCA(n_components=2)
        embedding_pca = pca.fit_transform(X_scaled)
        scatter = plt.scatter(embedding_pca[:, 0], embedding_pca[:, 1], c=df_clean[target_col], cmap=plt.cm.viridis, alpha=0.6)
        plt.title(f'PCA (Silhouette: {silhouette_score(embedding_pca, df_clean[target_col]):.3f})')
        plt.colorbar(scatter)
        
        plt.tight_layout()
        plt.show()
        
        # Comparaison des performances
        print("\n=== Comparaison des méthodes de visualisation ===")
        methods = {
            'UMAP': embedding,
            't-SNE': embedding_tsne,
            'PCA': embedding_pca
        }
        
        for method, emb in methods.items():
            silhouette = silhouette_score(emb, df_clean[target_col])
            calinski = calinski_harabasz_score(emb, df_clean[target_col])
            print(f"\n{method}:")
            print(f"- Score de silhouette: {silhouette:.3f}")
            print(f"- Score de Calinski-Harabasz: {calinski:.3f}")
            
    except Exception as e:
        print(f"Erreur lors de la comparaison des méthodes de visualisation : {str(e)}")

def analyze_feature_importance(df, target_col='outcome', include_enhanced=False):
    """
    Analyse l'importance des features, incluant optionnellement les features augmentées.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données
        target_col (str): Nom de la colonne cible
        include_enhanced (bool): Si True, inclut les features polynomiales et d'interaction
        
    Returns:
        pd.DataFrame: DataFrame avec les importances des features
    """
    try:
        # Create a copy of the DataFrame
        data = df.copy()
        
        # Convert target to binary (0 for 'noad.' and 1 for 'ad.')
        data[target_col] = (data[target_col] == 'ad.').astype(int)
        
        # Get numeric columns excluding the target
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = numeric_cols[numeric_cols != target_col]
        
        # Handle missing values
        for col in numeric_cols:
            missing_pct = data[col].isnull().mean()
            if missing_pct > 0.3:
                numeric_cols = numeric_cols.drop(col)
            elif missing_pct > 0.05:
                imputer = KNNImputer(n_neighbors=5)
                data[col] = imputer.fit_transform(data[[col]])[:, 0]
            else:
                data[col] = data[col].fillna(data[col].median())
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(data[numeric_cols])
        y = data[target_col]
        
        # Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = pd.Series(rf.feature_importances_, index=numeric_cols)
        
        # ANOVA F-scores
        f_scores, _ = f_classif(X, y)
        f_importance = pd.Series(f_scores, index=numeric_cols)
        
        # Correlation analysis
        correlations = data[numeric_cols].corrwith(data[target_col]).abs()
        
        # Combine scores
        importance_results = pd.DataFrame({
            'feature': numeric_cols,
            'RF_Importance': rf_importance / rf_importance.max(),
            'F_Score': f_importance / f_importance.max(),
            'Correlation': correlations / correlations.max()
        })
        
        importance_results['Combined_Score'] = importance_results[['RF_Importance', 'F_Score', 'Correlation']].mean(axis=1)
        importance_results = importance_results.sort_values('Combined_Score', ascending=False)
        
        # Get top features
        top_features = importance_results.head(20)
        
        # Plotting
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(top_features)), top_features['Combined_Score'])
        plt.xticks(range(len(top_features)), top_features['feature'], rotation=45, ha='right')
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()
        plt.show()
        
        if include_enhanced:
            enhanced_df = enhance_features(df, top_features['feature'].tolist())
            # Mettre à jour l'analyse avec les nouvelles features
            correlations = enhanced_df.corr()[target_col].sort_values(ascending=False)
            enhanced_importance = pd.DataFrame({
                'feature': correlations.index,
                'correlation': correlations.values
            })
            # Combiner les résultats originaux avec les résultats des features augmentées
            importance_results = pd.concat([importance_results, enhanced_importance], axis=0)
        
        return importance_results
        
    except Exception as e:
        print(f"Erreur lors de l'analyse de l'importance des variables : {str(e)}")
        # Return empty DataFrame with expected columns in case of error
        return pd.DataFrame(columns=['feature', 'RF_Importance', 'F_Score', 'Correlation', 'Combined_Score'])

def create_polynomial_features(data, variables=None, degree=2):
    """
    Crée des features polynomiales.
    
    Args:
        data (pd.DataFrame): DataFrame contenant les données
        variables (List[str]): Liste des variables à transformer
        degree (int): Degré maximum du polynôme
        
    Returns:
        pd.DataFrame: DataFrame avec les nouvelles features
    """
    if variables is None:
        # Si pas de variables spécifiées, utiliser les colonnes numériques
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = numeric_cols[numeric_cols != 'outcome']
        # Prendre les 4 premières colonnes numériques
        variables = numeric_cols[:4].tolist()
    
    new_features = {}
    for var in variables:
        if var in data.columns:  # Vérifier que la variable existe dans le DataFrame
            for d in range(2, degree + 1):
                new_features[f"{var}_pow{d}"] = data[var] ** d
    
    return pd.DataFrame(new_features)

def create_interaction_features(data, variables=None):
    """
    Crée des features d'interaction.
    
    Args:
        data (pd.DataFrame): DataFrame contenant les données
        variables (List[str]): Liste des variables pour créer les interactions
        
    Returns:
        pd.DataFrame: DataFrame avec les nouvelles features
    """
    if variables is None:
        # Si pas de variables spécifiées, utiliser les colonnes numériques
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = numeric_cols[numeric_cols != 'outcome']
        # Prendre les 4 premières colonnes numériques
        variables = numeric_cols[:4].tolist()
    
    new_features = {}
    for i, var1 in enumerate(variables):
        if var1 not in data.columns:  # Skip if variable doesn't exist
            continue
        for var2 in variables[i+1:]:
            if var2 not in data.columns:  # Skip if variable doesn't exist
                continue
            new_features[f"{var1}_{var2}_interact"] = data[var1] * data[var2]
    
    return pd.DataFrame(new_features)

def enhance_features(data, variables=None):
    """
    Crée toutes les nouvelles features et les combine.
    
    Args:
        data (pd.DataFrame): DataFrame contenant les données
        variables (List[str]): Liste des variables à utiliser
        
    Returns:
        pd.DataFrame: DataFrame avec toutes les features
    """
    try:
        poly_features = create_polynomial_features(data, variables)
        interact_features = create_interaction_features(data, variables)
        
        # Combine all features
        enhanced_df = pd.concat([data, poly_features, interact_features], axis=1)
        return enhanced_df
        
    except Exception as e:
        print(f"Erreur lors de la création des features augmentées : {str(e)}")
        return data  # Return original data if enhancement fails

def optimize_hyperparameters(X, y):
    """
    Optimise les hyperparamètres pour plusieurs modèles de classification.
    
    Parameters:
    -----------
    X : array-like
        Features matrix
    y : array-like
        Target variable
    
    Returns:
    --------
    dict
        Dictionary containing optimized models and their parameters
    """
    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Handle imbalanced data
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    # Define scoring metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }
    
    # Define parameter grids
    param_grids = {
        'RandomForest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'DecisionTree': {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        },
        'LogisticRegression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
    }
    
    # Initialize models
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42)
    }
    
    # Perform grid search for each model
    results = {}
    for name, model in models.items():
        print(f"\nOptimizing {name}...")
        grid_search = GridSearchCV(
            model,
            param_grids[name],
            cv=5,
            scoring=scoring,
            refit='f1',
            n_jobs=-1
        )
        grid_search.fit(X_resampled, y_resampled)
        
        results[name] = {
            'best_model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'cv_results': pd.DataFrame(grid_search.cv_results_)
        }
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best F1 score: {grid_search.best_score_:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    model_names = list(results.keys())
    f1_scores = [result['cv_results']['mean_test_f1'].max() for result in results.values()]
    
    plt.bar(model_names, f1_scores)
    plt.title('Model Comparison - F1 Scores')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1)
    
    for i, score in enumerate(f1_scores):
        plt.text(i, score, f'{score:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return results

def evaluate_optimized_models(models_dict, X, y):
    """
    Evaluate optimized models on test data.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary containing optimized models
    X : array-like
        Test features
    y : array-like
        Test target
    
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Evaluate each model
    results = {}
    for name, model_info in models_dict.items():
        model = model_info['best_model']
        y_pred = model.predict(X_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred)
        }
        
        results[name] = metrics
        
        print(f"\n{name.upper()} Model Evaluation:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(metrics))
    width = 0.2
    
    for i, (name, model_results) in enumerate(results.items()):
        plt.bar(x + i*width, [model_results[m] for m in metrics], width, label=name)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Comparison on Test Set')
    plt.xticks(x + width, metrics)
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
    
    return results
    

def main():
    try:
        # Load data
        print("\n=== Loading Data ===")
        df = load_data('data_train.csv')
        print("\nDataset dimensions:", df.shape)
        print("\nColumns:", df.columns.tolist())
        print("\nData types:\n", df.dtypes)
        print("\nFirst few rows:\n", df.head())
        
        # Analyze missing values
        print("\n=== Missing Values Analysis ===")
        missing_analysis = analyze_missing_values(df)
        
        # Compare visualization methods
        print("\n=== Comparing Visualization Methods ===")
        compare_visualization_methods(df)
        
        # Feature importance analysis
        print("\n=== Feature Importance Analysis ===")
        importance_results = analyze_feature_importance(df)
        if importance_results is not None:
            print("\nTop 20 most important features:")
            print(importance_results.nlargest(20, 'Combined_Score'))
        
        # UMAP visualization with top features
        if importance_results is not None:
            top_features = importance_results.nlargest(20, 'Combined_Score').index.tolist()
            print("\n=== UMAP Visualization with Top Features ===")
            umap_visualization(df[top_features + ['outcome']], 'outcome')
        
    except Exception as e:
        print(f"Error in main analysis: {str(e)}")

if __name__ == "__main__":
    main() 