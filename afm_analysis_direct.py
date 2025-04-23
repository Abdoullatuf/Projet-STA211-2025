import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import prince
import logging
from scipy import stats
from sklearn.impute import SimpleImputer, KNNImputer

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_missing_values(data, method='median'):
    """
    Prétraite les valeurs manquantes dans les données.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Le jeu de données à prétraiter
    method : str
        Méthode d'imputation ('median', 'mean', 'knn')
        
    Returns:
    --------
    data_imputed : pandas.DataFrame
        Le jeu de données avec les valeurs manquantes imputées
    """
    logger.info("Début du prétraitement des valeurs manquantes")
    
    # Vérification des valeurs manquantes
    missing_values = data.isnull().sum()
    if missing_values.sum() == 0:
        logger.info("Aucune valeur manquante détectée")
        return data
    
    logger.info(f"Valeurs manquantes par colonne :\n{missing_values[missing_values > 0]}")
    
    # Choix de la méthode d'imputation
    if method == 'median':
        imputer = SimpleImputer(strategy='median')
    elif method == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    else:
        raise ValueError(f"Méthode d'imputation non reconnue : {method}")
    
    # Application de l'imputation
    data_imputed = pd.DataFrame(
        imputer.fit_transform(data),
        columns=data.columns,
        index=data.index
    )
    
    logger.info("Imputation des valeurs manquantes terminée")
    return data_imputed

def perform_afm_analysis(data, target_column='outcome', missing_values_method='median'):
    """
    Effectue une Analyse Factorielle Multiple sur les données déjà encodées.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Le jeu de données à analyser (déjà encodé)
    target_column : str
        Le nom de la colonne cible
    missing_values_method : str
        Méthode d'imputation des valeurs manquantes ('median', 'mean', 'knn')
        
    Returns:
    --------
    mfa : prince.MFA
        L'objet MFA ajusté
    """
    # Vérification des données d'entrée
    if data.empty:
        raise ValueError("Le DataFrame d'entrée est vide")
    
    logger.info(f"Données d'entrée : {data.shape[0]} lignes, {data.shape[1]} colonnes")
    logger.info(f"Types de données :\n{data.dtypes}")

    # Prétraitement des valeurs manquantes
    data_processed = preprocess_missing_values(data, method=missing_values_method)

    # Préparation des données
    X = data_processed.drop(columns=[target_column]).copy()
    y = data_processed[target_column]

    # Vérification des colonnes
    if X.empty:
        raise ValueError("Aucune colonne explicative trouvée après suppression de la colonne cible")
    
    logger.info(f"Données finales : {X.shape}")

    # Identification des variables quantitatives et qualitatives
    quantitative_vars = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    qualitative_vars = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

    logger.info(f"Variables quantitatives : {len(quantitative_vars)}")
    logger.info(f"Variables qualitatives : {len(qualitative_vars)}")

    # Réorganisation des colonnes pour correspondre aux groupes
    X = X[quantitative_vars + qualitative_vars]

    # Création des groupes
    groups = []
    if quantitative_vars:
        groups.append(len(quantitative_vars))
    if qualitative_vars:
        groups.append(len(qualitative_vars))

    logger.info(f"Groupes définis : {groups}")

    # Application de l'AFM
    mfa = prince.MFA(
        n_components=2,
        n_iter=3,
        copy=True,
        check_input=True,
        random_state=42
    )

    # Ajustement du modèle
    logger.info("Début de l'ajustement du modèle MFA")
    try:
        mfa = mfa.fit(X, groups=groups)
        logger.info("Modèle MFA ajusté avec succès")
    except Exception as e:
        logger.error(f"Erreur lors de l'ajustement du modèle : {str(e)}")
        raise

    return mfa, X, y

def plot_afm_results(mfa, X, y, output_path=None):
    """
    Visualise les résultats de l'Analyse Factorielle Multiple.
    
    Parameters:
    -----------
    mfa : prince.MFA
        L'objet MFA ajusté
    X : pandas.DataFrame
        Les données explicatives
    y : pandas.Series
        La variable cible
    output_path : str, optional
        Chemin pour sauvegarder les graphiques
    """
    logger.info("Création des visualisations AFM")
    
    # Figure pour les composantes principales
    plt.figure(figsize=(12, 10))
    
    # Tracé des variables sur les deux premières composantes
    ax = plt.subplot(2, 2, 1)
    mfa.plot_partial_factor_map(ax=ax)
    plt.title("Carte factorielle des variables")
    
    # Tracé des individus colorés par la variable cible
    ax = plt.subplot(2, 2, 2)
    coords = mfa.row_coordinates(X)
    
    if isinstance(y.dtype, pd.CategoricalDtype) or y.dtype == 'object':
        # Pour une cible catégorielle
        scatter = sns.scatterplot(x=coords[0], y=coords[1], hue=y, palette='viridis', ax=ax)
    else:
        # Pour une cible numérique
        scatter = sns.scatterplot(x=coords[0], y=coords[1], hue=y, palette='coolwarm', ax=ax)
    
    plt.title("Projection des individus")
    plt.xlabel(f"Composante 1 ({mfa.eigenvalues_[0]:.1%})")
    plt.ylabel(f"Composante 2 ({mfa.eigenvalues_[1]:.1%})")
    
    # Tracé de la contribution des variables à chaque composante
    ax = plt.subplot(2, 2, 3)
    contributions = pd.DataFrame(
        mfa.column_contributions_,
        index=X.columns,
        columns=[f"Comp_{i+1}" for i in range(mfa.n_components)]
    )
    contributions.sort_values("Comp_1", ascending=False).head(10).plot(
        kind='barh', ax=ax
    )
    plt.title("Contribution des variables à la composante 1")
    
    # Tracé de l'inertie expliquée
    ax = plt.subplot(2, 2, 4)
    pd.Series(
        mfa.eigenvalues_,
        index=[f"Comp_{i+1}" for i in range(len(mfa.eigenvalues_))]
    ).plot(kind='bar', ax=ax)
    plt.title("Inertie expliquée par composante")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Visualisations sauvegardées dans {output_path}")
    
    plt.show()
    
    return

def interpret_afm_results(mfa, X, y):
    """
    Interprète les résultats de l'Analyse Factorielle Multiple.
    
    Parameters:
    -----------
    mfa : prince.MFA
        L'objet MFA ajusté
    X : pandas.DataFrame
        Les données explicatives
    y : pandas.Series
        La variable cible
    
    Returns:
    --------
    dict
        Un dictionnaire contenant les interprétations
    """
    logger.info("Interprétation des résultats AFM")
    
    # Récupération des coordonnées des variables
    var_coords = pd.DataFrame(
        mfa.column_coordinates_,
        index=X.columns,
        columns=[f"Comp_{i+1}" for i in range(mfa.n_components)]
    )
    
    # Contributions des variables aux composantes
    var_contrib = pd.DataFrame(
        mfa.column_contributions_,
        index=X.columns,
        columns=[f"Comp_{i+1}" for i in range(mfa.n_components)]
    )
    
    # Variables les plus contributives pour chaque composante
    top_vars_comp1 = var_contrib.sort_values("Comp_1", ascending=False).head(5)
    top_vars_comp2 = var_contrib.sort_values("Comp_2", ascending=False).head(5)
    
    # Corrélation entre les composantes principales et la variable cible
    components = mfa.row_coordinates(X)
    components.columns = [f"Comp_{i+1}" for i in range(mfa.n_components)]
    
    # Jointure avec la variable cible
    components_with_y = pd.DataFrame({
        "Comp_1": components[0],
        "Comp_2": components[1],
        "target": y
    })
    
    # Calcul des corrélations
    if pd.api.types.is_numeric_dtype(y):
        correlations = components_with_y.corr()["target"].drop("target")
        corr_analysis = f"La composante 1 a une corrélation de {correlations['Comp_1']:.2f} avec la cible.\n"
        corr_analysis += f"La composante 2 a une corrélation de {correlations['Comp_2']:.2f} avec la cible."
    else:
        # Pour les cibles catégorielles, calcul de l'ANOVA
        f_values = {}
        p_values = {}
        
        for comp in ["Comp_1", "Comp_2"]:
            groups = [components_with_y[components_with_y["target"] == cat][comp] 
                     for cat in components_with_y["target"].unique()]
            f_val, p_val = stats.f_oneway(*groups)
            f_values[comp] = f_val
            p_values[comp] = p_val
        
        corr_analysis = f"ANOVA pour Comp_1: F={f_values['Comp_1']:.2f}, p={p_values['Comp_1']:.4f}\n"
        corr_analysis += f"ANOVA pour Comp_2: F={f_values['Comp_2']:.2f}, p={p_values['Comp_2']:.4f}"
    
    # Construction du rapport d'interprétation
    interpretation = {
        "inertie_expliquee": {
            "Comp_1": mfa.eigenvalues_[0],
            "Comp_2": mfa.eigenvalues_[1],
            "Total": mfa.eigenvalues_[0] + mfa.eigenvalues_[1]
        },
        "variables_contributives": {
            "Comp_1": top_vars_comp1.index.tolist(),
            "Comp_2": top_vars_comp2.index.tolist()
        },
        "correlation_avec_cible": corr_analysis
    }
    
    # Affichage des résultats
    print("\n=== INTERPRÉTATION DES RÉSULTATS DE L'AFM ===\n")
    print(f"Inertie expliquée par la composante 1: {mfa.eigenvalues_[0]:.2%}")
    print(f"Inertie expliquée par la composante 2: {mfa.eigenvalues_[1]:.2%}")
    print(f"Inertie totale expliquée: {mfa.eigenvalues_[0] + mfa.eigenvalues_[1]:.2%}\n")
    
    print("Variables les plus contributives à la composante 1:")
    for var, contrib in top_vars_comp1.iterrows():
        print(f"  - {var}: {contrib['Comp_1']:.2%}")
    
    print("\nVariables les plus contributives à la composante 2:")
    for var, contrib in top_vars_comp2.iterrows():
        print(f"  - {var}: {contrib['Comp_2']:.2%}")
    
    print(f"\nRelation avec la variable cible:\n{corr_analysis}")
    
    return interpretation