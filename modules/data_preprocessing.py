# modules/data_preprocessing.py

"""
Module de prétraitement des données pour le projet STA211.
Ce module contient des fonctions pour :
  - détecter l'environnement Colab (facultatif)
  - charger et nettoyer un CSV,
  - analyser les valeurs manquantes,
  - rechercher k optimal pour KNN,
  - imputer les valeurs manquantes (simple, KNN, multiple).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from IPython.display import display

# Pour l'imputation multiple
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

__all__ = [
    'is_colab',
    'clean_data',
    'load_data',
    'analyze_missing_values',
    'find_optimal_k',
    'handle_missing_values'
]


def is_colab() -> bool:
    """Détecte si le code s'exécute dans Google Colab (utile ailleurs si besoin)."""
    try:
        import google.colab  # noqa
        return True
    except ImportError:
        return False


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie un DataFrame :
      - strip + supprime guillemets des noms de colonnes,
      - strip + supprime guillemets des valeurs string.
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace('"', '')
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip().str.replace('"', '')
    return df


# ---------------------------------------------------------------------
# On résout dynamiquement la racine de projet et le dossier data/raw
MODULE_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODULE_DIR)
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
# ---------------------------------------------------------------------


def load_data(file_path: str, require_outcome: bool = True) -> pd.DataFrame:
    """
    Charge un CSV (tab ou virgule) et nettoie :
      - file_path : 
         * chemin ABSOLU → utilisé tel quel
         * chemin RELATIF → cherché dans data/raw/ à la racine du projet
      - require_outcome : si True, lève une erreur si la colonne 'outcome' manque.
    Affiche shape, info() et head().
    """
    # 1) Détection du chemin réel
    if os.path.isabs(file_path):
        real_path = file_path
    else:
        real_path = os.path.join(RAW_DATA_DIR, file_path)

    if not os.path.exists(real_path):
        raise FileNotFoundError(f"📂 Fichier introuvable : {real_path}")

    # 2) Lecture CSV (priorité tabulation puis virgule)
    try:
        df = pd.read_csv(real_path, sep='\t')
        if df.shape[1] == 1:
            df = pd.read_csv(real_path, sep=',')
    except Exception as e:
        raise RuntimeError(f"❌ Erreur lecture {real_path}: {e}")

    # 3) Vérification outcome
    if require_outcome and 'outcome' not in df.columns:
        raise ValueError("🚨 La colonne 'outcome' est manquante.")

    # 4) Nettoyage
    df = clean_data(df)

    # 5) Diagnostics
    print(f"Dimensions du dataset: {df.shape}\n")
    print("Infos colonnes :")
    df.info()
    print("\nAperçu des données :")
    display(df.head())

    return df


def analyze_missing_values(df: pd.DataFrame) -> dict:
    """
    Analyse des valeurs manquantes :
      - Total et % global
      - Par colonne : high (>30%), medium (5-30%), low (≤5%)
      - Top 5 colonnes manquantes
    Retourne un dict de statistiques.
    """
    total = df.size
    miss = df.isnull().sum().sum()
    pct = miss / total * 100

    by_col = df.isnull().sum()
    by_col = by_col[by_col > 0]
    pct_col = by_col / len(df) * 100

    high = pct_col[pct_col > 30]
    med  = pct_col[(pct_col > 5) & (pct_col <= 30)]
    low  = pct_col[pct_col <= 5]

    print(f"Total missing       : {miss} ({pct:.2f}%)")
    print(f"Colonnes affectées  : {len(by_col)} "
          f"(haut: {len(high)}, moyen: {len(med)}, bas: {len(low)})")
    print("Top 5 colonnes manquantes :")
    print(pct_col.sort_values(ascending=False).head())

    return {
        'total_missing':      int(miss),
        'percent_missing':    pct,
        'cols_missing':       by_col.to_dict(),
        'percent_per_col':    pct_col.to_dict(),
        'high_missing':       high.to_dict(),
        'medium_missing':     med.to_dict(),
        'low_missing':        low.to_dict(),
    }


def find_optimal_k(
    data: pd.DataFrame,
    continuous_cols: list[str],
    k_range: range = range(1, 21),
    cv_folds: int = 5,
    sample_size: int = 1000
) -> int:
    """
    Trouve k optimal pour KNNImputer via CV sur un petit échantillon.
    Affiche la courbe MSE vs k, et renvoie le k minimisant le MSE.
    """
    X = data[continuous_cols].copy()
    if len(X) > sample_size:
        X = X.sample(n=sample_size, random_state=42)

    rng = np.random.RandomState(42)
    mask = rng.rand(*X.shape) < 0.2
    X_missing = X.copy()
    X_missing[mask] = np.nan

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    mse_scores = []

    for k in k_range:
        fold_mse = []
        imputer = KNNImputer(n_neighbors=k)
        for train_idx, val_idx in cv.split(X):
            X_tr   = X_missing.iloc[train_idx]
            X_val  = X_missing.iloc[val_idx]
            X_true = X.iloc[val_idx]

            imputer.fit(X_tr)
            X_imp = pd.DataFrame(
                imputer.transform(X_val),
                columns=X_val.columns,
                index=X_val.index
            )

            mask_val = mask[val_idx]
            y_true = X_true.to_numpy(dtype=float)[mask_val]
            y_pred = X_imp.to_numpy(dtype=float)[mask_val]
            valid = (~np.isnan(y_true)) & (~np.isnan(y_pred))
            if valid.any():
                fold_mse.append(
                    mean_squared_error(y_true[valid], y_pred[valid])
                )

        mse_scores.append(np.mean(fold_mse))

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(list(k_range), mse_scores, marker='o')
    plt.xlabel('k (nombre de voisins)')
    plt.ylabel('Mean Squared Error')
    plt.title('KNN Imputation Performance')
    plt.grid(True)
    plt.show()

    best_k = int(k_range[np.argmin(mse_scores)])
    print(f"→ k optimal : {best_k}")
    return best_k

from typing import Optional

import pandas as pd
import numpy as np
import os
import joblib
from typing import Optional
from sklearn.impute import KNNImputer, IterativeImputer

def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'mixed_mar_mcar',
    mar_method: str = 'knn',
    knn_k: Optional[int] = None,
    display_info: bool = True,
    save_results: bool = True,
    processed_data_dir: Optional[str] = None,
    models_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Imputation des données manquantes selon la stratégie choisie :
    - 'all_median'       : Imputation par la médiane sur toutes les colonnes numériques.
    - 'mixed_mar_mcar'   : KNN ou imputation multiple pour X1-X3, médiane pour X4.

    Paramètres :
    - df : DataFrame à traiter.
    - strategy : 'all_median' ou 'mixed_mar_mcar'.
    - mar_method : 'knn' ou 'multiple' (utilisé uniquement si strategy='mixed_mar_mcar').
    - knn_k : nombre de voisins pour le KNN, ou None pour détection automatique.
    - display_info : bool, afficher des informations intermédiaires.
    - save_results : bool, enregistrer le résultat dans un fichier CSV et l’imputer.
    - processed_data_dir : chemin du dossier où sauvegarder le fichier CSV (requis si save_results=True).
    - models_dir : chemin du dossier où sauvegarder les objets (imputer, etc.).
    """
    df_proc = df.copy()
    suffix = ''

    if strategy == 'all_median':
        num_cols = df_proc.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            df_proc[col] = df_proc[col].fillna(df_proc[col].median())
        suffix = 'median_all'
        if display_info:
            print(f"→ Imputation par médiane sur {len(num_cols)} colonnes numériques.")

    elif strategy == 'mixed_mar_mcar':
        mar_cols = ['X1', 'X2', 'X3']
        mcar_col = 'X4'

        if all(c in df_proc.columns for c in mar_cols):
            if mar_method == 'knn':
                if knn_k is None:
                    from data_preprocessing import find_optimal_k
                    knn_k = find_optimal_k(df_proc, mar_cols)
                    if display_info:
                        print(f"→ k optimal automatiquement détecté : {knn_k}")
                imputer = KNNImputer(n_neighbors=knn_k)
                df_proc[mar_cols] = imputer.fit_transform(df_proc[mar_cols])
                suffix = f'knn_k{knn_k}'

                if save_results and models_dir:
                    os.makedirs(models_dir, exist_ok=True)
                    imp_path = os.path.join(models_dir, f"imputer_{suffix}.pkl")
                    joblib.dump(imputer, imp_path)

            elif mar_method == 'multiple':
                imputer = IterativeImputer(random_state=42, max_iter=10)
                df_proc[mar_cols] = imputer.fit_transform(df_proc[mar_cols])
                suffix = 'multiple'

                if save_results and models_dir:
                    os.makedirs(models_dir, exist_ok=True)
                    imp_path = os.path.join(models_dir, f"imputer_{suffix}.pkl")
                    joblib.dump(imputer, imp_path)

            else:
                raise ValueError("mar_method doit être 'knn' ou 'multiple'")

        if mcar_col in df_proc.columns:
            median_val = df_proc[mcar_col].median()
            df_proc[mcar_col] = df_proc[mcar_col].fillna(median_val)
            if display_info:
                print(f"→ Médiane imputée pour {mcar_col} (valeur = {median_val:.4f})")

    else:
        raise ValueError("strategy doit être 'all_median' ou 'mixed_mar_mcar'")

    if save_results:
        if processed_data_dir is None:
            raise ValueError("processed_data_dir doit être fourni si save_results=True.")
        filename = f"df_imputed_{suffix}.csv"
        filepath = os.path.join(processed_data_dir, filename)
        df_proc.to_csv(filepath, index=False)
        if display_info:
            print(f"✔ Données imputées sauvegardées dans '{filepath}'")

    return df_proc




