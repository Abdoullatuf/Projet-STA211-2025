import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model_cv(model, X, y, cv, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']):
    """
    Évalue un modèle en utilisant la validation croisée avec plusieurs métriques.
    
    Parameters:
    -----------
    model : estimator object
        Le modèle à évaluer
    X : array-like
        Les features
    y : array-like
        La variable cible
    cv : cross-validation generator
        La stratégie de validation croisée
    scoring : list
        Liste des métriques à calculer
        
    Returns:
    --------
    dict
        Dictionnaire contenant les scores moyens et écarts-types
    """
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    
    results = {}
    for metric in scoring:
        mean_score = np.mean(scores[f'test_{metric}'])
        std_score = np.std(scores[f'test_{metric}'])
        results[metric] = {'mean': mean_score, 'std': std_score}
    
    return results

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Visualise l'importance des variables pour un modèle.
    
    Parameters:
    -----------
    model : estimator object
        Le modèle entraîné avec feature_importances_
    feature_names : array-like
        Les noms des variables
    top_n : int
        Nombre de variables à afficher
    """
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    
    importances = importances.sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importances, x='importance', y='feature')
    plt.title(f'Top {top_n} variables les plus importantes')
    plt.xlabel('Importance')
    plt.ylabel('Variable')
    plt.tight_layout()
    plt.show()

def plot_learning_curves(model, X, y, cv):
    """
    Trace les courbes d'apprentissage pour analyser le compromis biais-variance.
    
    Parameters:
    -----------
    model : estimator object
        Le modèle à évaluer
    X : array-like
        Les features
    y : array-like
        La variable cible
    cv : cross-validation generator
        La stratégie de validation croisée
    """
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes,
        scoring='accuracy', n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Score d\'entraînement')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.plot(train_sizes, val_mean, label='Score de validation')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    
    plt.xlabel('Taille de l\'échantillon d\'entraînement')
    plt.ylabel('Score')
    plt.title('Courbes d\'apprentissage')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def compare_models(models_dict, X, y, cv):
    """
    Compare plusieurs modèles sur différentes métriques.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionnaire des modèles à comparer
    X : array-like
        Les features
    y : array-like
        La variable cible
    cv : cross-validation generator
        La stratégie de validation croisée
    """
    results = {}
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    for name, model in models_dict.items():
        scores = evaluate_model_cv(model, X, y, cv, metrics)
        results[name] = scores
    
    # Création d'un DataFrame pour la visualisation
    comparison_df = pd.DataFrame()
    for model_name, model_scores in results.items():
        for metric, scores in model_scores.items():
            comparison_df.loc[model_name, f'{metric}_mean'] = scores['mean']
            comparison_df.loc[model_name, f'{metric}_std'] = scores['std']
    
    return comparison_df 