import pandas as pd
import numpy as np
import joblib
import os

from data_preprocessing import clean_data, is_colab
from transformations import transform_selected_variables

def preprocess_test_data(file_path, model_dir="models", missing_strategy="impute", imputer_file=None):
    """
    Prétraite un fichier de test en reproduisant le pipeline du train.

    :param file_path: chemin vers le fichier CSV de test (sans outcome).
    :param model_dir: répertoire contenant les objets entraînés (imputer, scaler, etc.).
    :param missing_strategy: "impute" (par défaut) ou "drop" pour supprimer les lignes incomplètes.
    :param imputer_file: nom de fichier imputer KNN si connu, sinon détection automatique.
    :return: DataFrame prêt pour prédiction.
    """
    df = pd.read_csv(file_path)
    df = clean_data(df)

    if imputer_file is None:
        imputer_file = next(f for f in os.listdir(model_dir) if f.startswith("imputer_knn_k") and f.endswith(".pkl"))
    imputer = joblib.load(os.path.join(model_dir, imputer_file))

    drop_cols = joblib.load(os.path.join(model_dir, "drop_cols.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler_knn.pkl"))
    columns_used = joblib.load(os.path.join(model_dir, "columns_used.pkl"))

    if missing_strategy == "drop":
        df = df.dropna()
    elif missing_strategy == "impute":
        required_cols = ['X1', 'X2', 'X3']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise KeyError(f"❌ Colonnes manquantes dans les données de test : {missing}")
        df[required_cols] = imputer.transform(df[required_cols])
        df['X4'] = df['X4'].fillna(df['X4'].median())
    else:
        raise ValueError("missing_strategy doit être 'impute' ou 'drop'.")

    df = transform_selected_variables(df)
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
    df = df[columns_used]
    df_scaled = scaler.transform(df)

    return pd.DataFrame(df_scaled, columns=columns_used)

def generate_submission(
    test_file="data_test.csv",
    imputer_file=None,
    scaler_file="scaler_knn.pkl",
    model_file="best_model.joblib",
    columns_file="columns_used.pkl",
    drop_cols_file="drop_cols.pkl",
    save_path="my_pred.csv",
    raw_data_dir=os.path.join(os.path.expanduser("~"), "OneDrive", "Documents", "Projects", "STA211_Challenge_2025", "data", "raw") if not is_colab() else "/content/drive/Othercomputers/Mon_pc_hp/Documents/Projects/STA211_Challenge_2025/data/raw",
    processed_data_dir=os.path.join(os.path.expanduser("~"), "OneDrive", "Documents", "Projects", "STA211_Challenge_2025", "data", "processed") if not is_colab() else "/content/drive/Othercomputers/Mon_pc_hp/Documents/Projects/STA211_Challenge_2025/data/processed",
    models_dir=os.path.join(os.path.expanduser("~"), "OneDrive", "Documents", "Projects", "STA211_Challenge_2025", "models") if not is_colab() else "/content/drive/Othercomputers/Mon_pc_hp/Documents/Projects/STA211_Challenge_2025/models",
    root_dir=os.path.join(os.path.expanduser("~"), "OneDrive", "Documents", "Projects", "STA211_Challenge_2025") if not is_colab() else "/content/drive/Othercomputers/Mon_pc_hp/Documents/Projects/STA211_Challenge_2025""."
):
    """
    Génère un fichier de prédiction my_pred.csv à partir du fichier data_test.csv
    en reproduisant le pipeline de prétraitement du train.
    """
    from data_preprocessing import load_data
    df = load_data(os.path.join(raw_data_dir, test_file), require_outcome=False)

    if imputer_file is None:
        imputer_file = next(f for f in os.listdir(models_dir) if f.startswith("imputer_knn_k") and f.endswith(".pkl"))
    imputer = joblib.load(os.path.join(models_dir, imputer_file))

    required_cols = ['X1', 'X2', 'X3']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"❌ Colonnes manquantes dans les données de test : {missing}")
    df[required_cols] = imputer.transform(df[required_cols])
    df['X4'] = df['X4'].fillna(df['X4'].median())

    df = transform_selected_variables(df)

    drop_path = os.path.join(models_dir, drop_cols_file)
    if os.path.exists(drop_path):
        drop_cols = joblib.load(drop_path)
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    columns_used = joblib.load(os.path.join(processed_data_dir, columns_file))
    df = df[columns_used]

    scaler = joblib.load(os.path.join(processed_data_dir, scaler_file))
    df_scaled = scaler.transform(df)

    model = joblib.load(os.path.join(models_dir, model_file))
    y_pred = model.predict(df_scaled)
    labels = np.where(y_pred == 1, "ad.", "noad.")

    pred_path = os.path.join(root_dir, save_path)
    pd.Series(labels).to_csv(pred_path, index=False, header=False)
    print(f"✅ Fichier de prédiction généré : {pred_path}")
