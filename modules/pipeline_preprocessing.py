
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import chi2
from scipy.stats import boxcox, normaltest
import numpy as np
import pandas as pd
import prince

def preprocessing_pipeline(df, target, imputation='knn', k_neighbors=5, alpha=0.05, corr_threshold=0.99,
                           use_afm=False, afm_threshold=0.9, output_csv='data_cleaned.csv'):
    df = df.copy()
    
    # 1. Séparation des variables et de la cible
    y = df[target]
    X = df.drop(columns=[target])
    
    # 2. Imputation des valeurs manquantes
    if imputation == 'knn':
        imputer = KNNImputer(n_neighbors=k_neighbors)
    elif imputation == 'multiple':
        imputer = IterativeImputer(random_state=42)
    else:
        raise ValueError("Méthode d'imputation non reconnue. Choisir 'knn' ou 'multiple'.")
    
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # 3. Transformations log(1+x) et Box-Cox sur X1, X2, X3
    for col in ['X1', 'X2', 'X3']:
        if col in X_imputed.columns:
            X_imputed[col] = np.log1p(X_imputed[col])
            _, pval = normaltest(X_imputed[col])
            if pval < 0.05:
                X_imputed[col], _ = boxcox(X_imputed[col] + 1)
    
    # 4. Sélection des variables binaires par chi²
    binary_cols = X_imputed.columns[(X_imputed.nunique() == 2)]
    chi2_scores, p_values = chi2(X_imputed[binary_cols], y)
    selected_binaries = binary_cols[p_values < alpha]
    X_selected = X_imputed[selected_binaries].copy()
    
    # 5. Suppression des variables redondantes (corrélation)
    corr_matrix = X_selected.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    X_final = X_selected.drop(columns=to_drop)
    
    # 6. AFM (Analyse Factorielle Multiple) si demandé
    if use_afm and not X_final.empty:
        mca = prince.MCA(n_components=len(X_final.columns), random_state=42)
        mca = mca.fit(X_final)
        cumulative_variance = mca.explained_inertia_.cumsum()
        n_components = np.argmax(cumulative_variance >= afm_threshold) + 1
        X_afm = mca.transform(X_final).iloc[:, :n_components]
        X_final = X_afm
    
    # 7. Ajout de la cible et export
    final_df = pd.concat([X_final.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    final_df.to_csv(output_csv, index=False)
    
    return final_df
