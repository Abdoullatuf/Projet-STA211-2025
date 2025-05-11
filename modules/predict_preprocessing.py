import pandas as pd
import numpy as np
import joblib
import os


from data_preprocessing import clean_data, is_colab, load_data
from transformations import transform_selected_variables

def preprocess_test_data(file_path, model_dir="models", missing_strategy="impute", imputer_file=None):
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
    root_dir=os.path.join(os.path.expanduser("~"), "OneDrive", "Documents", "Projects", "STA211_Challenge_2025") if not is_colab() else "/content/drive/Othercomputers/Mon_pc_hp/Documents/Projects/STA211_Challenge_2025"
):
    print("🔵 Chargement des données de test...")
    df = load_data(os.path.join(raw_data_dir, test_file), require_outcome=False)

    print("🔵 Chargement de l'imputer...")
    if imputer_file is None:
        imputer_file = next(f for f in os.listdir(models_dir) if f.startswith("imputer_knn_k") and f.endswith(".pkl"))
    imputer = joblib.load(os.path.join(models_dir, imputer_file))

    # Imputation sur colonnes X1, X2, X3
    required_cols = ['X1', 'X2', 'X3']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"❌ Colonnes manquantes dans les données de test : {missing}")
    df[required_cols] = imputer.transform(df[required_cols])

    # Remplir X4 si besoin
    df['X4'] = df['X4'].fillna(df['X4'].median())

    print("🔵 Application des transformations (log, boxcox)...")
    df = transform_selected_variables(df)

    # Suppression des colonnes inutiles
    drop_path = os.path.join(models_dir, drop_cols_file)
    if os.path.exists(drop_path):
        drop_cols = joblib.load(drop_path)
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
        print(f"🔵 {len(drop_cols)} colonnes supprimées après nettoyage.")

    # Sélectionner les colonnes finales
    if columns_file == "features_rf_top40.pkl":
        columns_used = joblib.load(os.path.join(models_dir, columns_file))
        apply_scaler = False  # 🔥 Pas de scaling pour RandomForest Top 40
    else:
        columns_used = joblib.load(os.path.join(processed_data_dir, columns_file))
        apply_scaler = True  # 🔥 Scaling pour Stacking et autres

    print(f"🔵 Sélection de {len(columns_used)} colonnes finales pour la prédiction...")
    df = df[columns_used]

    # Appliquer le scaler seulement si besoin
    if apply_scaler:
        print("🔵 Chargement du scaler...")
        scaler = joblib.load(os.path.join(processed_data_dir, scaler_file))
        df_scaled = scaler.transform(df)
    else:
        print("🛠 Pas d'application de scaler pour RandomForest Top 40.")
        df_scaled = df.values  # Juste les valeurs

    # Prédiction
    print("🔵 Chargement du modèle...")
    model = joblib.load(os.path.join(models_dir, model_file))

    print("🛠 Génération des prédictions...")
    y_pred = model.predict(df_scaled)

    # Mapping 0/1 vers "noad." / "ad."
    labels = np.where(y_pred == 1, "ad.", "noad.")

    # Sauvegarde
    os.makedirs(os.path.dirname(os.path.join(root_dir, save_path)), exist_ok=True)
    pred_path = os.path.join(root_dir, save_path)
    pd.Series(labels).to_csv(pred_path, index=False, header=False)
    print(f"✅ Fichier de prédiction généré : {pred_path}")

def generate_submission_mfa(
    test_file="data_test.csv",
    imputer_file=None,
    scaler_file="scaler_knn.pkl",
    model_file="best_model_mfa.joblib",
    mfa_file="mfa_model.pkl",
    save_path="my_pred_mfa.csv",
    raw_data_dir=os.path.join(os.path.expanduser("~"), "OneDrive", "Documents", "Projects", "STA211_Challenge_2025", "data", "raw") if not is_colab() else "/content/drive/Othercomputers/Mon_pc_hp/Documents/Projects/STA211_Challenge_2025/data/raw",
    models_dir=os.path.join(os.path.expanduser("~"), "OneDrive", "Documents", "Projects", "STA211_Challenge_2025", "models") if not is_colab() else "/content/drive/Othercomputers/Mon_pc_hp/Documents/Projects/STA211_Challenge_2025/models",
    root_dir=os.path.join(os.path.expanduser("~"), "OneDrive", "Documents", "Projects", "STA211_Challenge_2025") if not is_colab() else "/content/drive/Othercomputers/Mon_pc_hp/Documents/Projects/STA211_Challenge_2025"
):
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

    mfa_model = joblib.load(os.path.join(models_dir, mfa_file))

    # Séparer variables quantitatives et binaires
    quantitative_vars = ['X1_log', 'X2_boxcox', 'X3_boxcox', 'X4']
    binary_vars = [col for col in df.columns if col not in quantitative_vars and df[col].nunique() == 2]

    # Créer MultiIndex attendu par prince.MFA
    mfa_input = df[quantitative_vars + binary_vars].copy()
    mfa_input.columns = pd.MultiIndex.from_tuples(
        [('Quantitatives', col) if col in quantitative_vars else ('Binaires', col) for col in mfa_input.columns]
    )

    df_mfa = mfa_model.transform(mfa_input)
    df_mfa.columns = [f"AFM_{i+1}" for i in range(df_mfa.shape[1])]

    scaler = joblib.load(os.path.join(models_dir, scaler_file))
    df_scaled = scaler.transform(df_mfa)

    model = joblib.load(os.path.join(models_dir, model_file))
    y_pred = model.predict(df_scaled)
    labels = np.where(y_pred == 1, "ad.", "noad.")

    pred_path = os.path.join(root_dir, save_path)
    pd.Series(labels).to_csv(pred_path, index=False, header=False)
    print(f"✅ Fichier de prédiction (MFA) généré : {pred_path}")

