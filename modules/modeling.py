"""
Module de modélisation pour le projet STA211.
Contient les fonctions liées à la création de variables, à la sélection de features, à l'entraînement et à l'évaluation des modèles.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer, SimpleImputer

__all__ = [
    'analyze_feature_importance',
    'create_polynomial_features',
    'create_interaction_features',
    'enhance_features',
    'optimize_hyperparameters',
    'evaluate_optimized_models'
]

def analyze_feature_importance(df, target_col='outcome', include_enhanced=False):
    try:
        data = df.copy()
        data[target_col] = (data[target_col] == 'ad.').astype(int)
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = numeric_cols[numeric_cols != target_col]

        for col in numeric_cols:
            missing_pct = data[col].isnull().mean()
            if missing_pct > 0.3:
                numeric_cols = numeric_cols.drop(col)
            elif missing_pct > 0.05:
                imputer = KNNImputer(n_neighbors=5)
                data[col] = imputer.fit_transform(data[[col]])[:, 0]
            else:
                data[col] = data[col].fillna(data[col].median())

        scaler = StandardScaler()
        X = scaler.fit_transform(data[numeric_cols])
        y = data[target_col]

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = pd.Series(rf.feature_importances_, index=numeric_cols)

        f_scores, _ = f_classif(X, y)
        f_importance = pd.Series(f_scores, index=numeric_cols)

        correlations = data[numeric_cols].corrwith(data[target_col]).abs()

        importance_results = pd.DataFrame({
            'feature': numeric_cols,
            'RF_Importance': rf_importance / rf_importance.max(),
            'F_Score': f_importance / f_importance.max(),
            'Correlation': correlations / correlations.max()
        })

        importance_results['Combined_Score'] = importance_results[['RF_Importance', 'F_Score', 'Correlation']].mean(axis=1)
        importance_results = importance_results.sort_values('Combined_Score', ascending=False)

        top_features = importance_results.head(20)

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(top_features)), top_features['Combined_Score'])
        plt.xticks(range(len(top_features)), top_features['feature'], rotation=45, ha='right')
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()
        plt.show()

        if include_enhanced:
            enhanced_df = enhance_features(df, top_features['feature'].tolist())
            correlations = enhanced_df.corr()[target_col].sort_values(ascending=False)
            enhanced_importance = pd.DataFrame({
                'feature': correlations.index,
                'correlation': correlations.values
            })
            importance_results = pd.concat([importance_results, enhanced_importance], axis=0)

        return importance_results

    except Exception as e:
        print(f"Erreur lors de l'analyse de l'importance des variables : {str(e)}")
        return pd.DataFrame(columns=['feature', 'RF_Importance', 'F_Score', 'Correlation', 'Combined_Score'])

def create_polynomial_features(data, variables=None, degree=2):
    if variables is None:
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = numeric_cols[numeric_cols != 'outcome']
        variables = numeric_cols[:4].tolist()

    new_features = {}
    for var in variables:
        if var in data.columns:
            for d in range(2, degree + 1):
                new_features[f"{var}_pow{d}"] = data[var] ** d

    return pd.DataFrame(new_features)

def create_interaction_features(data, variables=None):
    if variables is None:
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = numeric_cols[numeric_cols != 'outcome']
        variables = numeric_cols[:4].tolist()

    new_features = {}
    for i, var1 in enumerate(variables):
        if var1 not in data.columns:
            continue
        for var2 in variables[i+1:]:
            if var2 not in data.columns:
                continue
            new_features[f"{var1}_{var2}_interact"] = data[var1] * data[var2]

    return pd.DataFrame(new_features)

def enhance_features(data, variables=None):
    try:
        poly_features = create_polynomial_features(data, variables)
        interact_features = create_interaction_features(data, variables)
        enhanced_df = pd.concat([data, poly_features, interact_features], axis=1)
        return enhanced_df
    except Exception as e:
        print(f"Erreur lors de la création des features augmentées : {str(e)}")
        return data

def optimize_hyperparameters(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }

    param_grids = {
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        },
        'DecisionTree': {
            'max_depth': [None, 10],
            'min_samples_split': [2, 5],
            'criterion': ['gini', 'entropy']
        },
        'LogisticRegression': {
            'C': [0.01, 1],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
    }

    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42)
    }

    results = {}
    for name, model in models.items():
        print(f"
Optimizing {name}...")
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
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

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

        print(f"
{name.upper()} Model Evaluation:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")

    plt.figure(figsize=(15, 5))
    metrics_list = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(metrics_list))
    width = 0.2

    for i, (name, model_results) in enumerate(results.items()):
        plt.bar(x + i*width, [model_results[m] for m in metrics_list], width, label=name)

    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Comparison on Test Set')
    plt.xticks(x + width, metrics_list)
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

    return results
