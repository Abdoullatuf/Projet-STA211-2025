#!/usr/bin/env python
"""
Script de prétraitement des données.
Exécute le pipeline de nettoyage et d'imputation des données.
"""

import sys
import os
import argparse
from pathlib import Path

# Ajouter le dossier src au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from src.data_preprocessing import load_data, handle_missing_values

def main():
    # Configuration des arguments en ligne de commande
    parser = argparse.ArgumentParser(description='Prétraitement des données')
    parser.add_argument('--file', choices=['train', 'test'], default='train',
                      help='Fichier à traiter: train (data_train.csv) ou test (data_test.csv)')
    args = parser.parse_args()

    # Déterminer les noms de fichiers
    input_filename = f"data_{args.file}.csv"
    output_filename = f"data_{args.file}_processed.csv"

    # Chemins des fichiers
    data_dir = Path(__file__).parent.parent / "data"
    raw_data = data_dir / "raw" / input_filename
    processed_data = data_dir / "processed" / output_filename
    
    # Vérifier que le fichier de données existe
    if not raw_data.exists():
        print(f"Erreur: Le fichier de données {raw_data} n'existe pas.")
        print("Veuillez placer vos fichiers de données dans le dossier data/raw/")
        print("Les fichiers doivent s'appeler 'data_train.csv' et 'data_test.csv'")
        sys.exit(1)
    
    # Créer les dossiers si nécessaire
    processed_data.parent.mkdir(parents=True, exist_ok=True)
    
    # Charger les données
    print(f"Chargement des données {input_filename}...")
    try:
        # Requérir la colonne outcome uniquement pour les données d'entraînement
        require_outcome = (args.file == 'train')
        df = load_data(str(raw_data), require_outcome=require_outcome)
        if df is None:
            print("Erreur lors du chargement des données.")
            sys.exit(1)
    except Exception as e:
        print(f"Erreur lors du chargement des données: {str(e)}")
        sys.exit(1)
    
    # Imputer les valeurs manquantes
    print("Imputation des valeurs manquantes...")
    try:
        df_processed = handle_missing_values(
            df,
            strategy='advanced',
            display_info=True,
            save_results=True,
            output_path=str(processed_data)
        )
        print(f"Données traitées enregistrées dans {processed_data}")
    except Exception as e:
        print(f"Erreur lors de l'imputation des données: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 