# Architecture du projet STA211 – Internet Advertisements Challenge

## 1. Contexte et objectifs

- **Brève description du challenge**  
  L’Internet Advertisements Challenge consiste à construire un modèle capable de distinguer, à partir d’un ensemble de descripteurs extraits d’images web, si chaque image est une **publicité** (`ad.`) ou **non-publicité** (`noad.`). L’enjeu pédagogique est de maîtriser l’ensemble du pipeline Data Science : exploration, pré-traitement, modélisation supervisée, validation et déploiement.

- **Contraintes**  
  - **Jeu de données** :  
    - Fichier CSV (`data_train.csv`) de 820 exemples, 1558 variables numériques et une colonne cible `outcome` contenant les valeurs `ad.` ou `noad.`.  
    - Présence de données manquantes (NA) sur plusieurs variables, classes déséquilibrées (~30 % de pubs).  
  - **Format** :  
    - Entrée : DataFrame pandas, Jupyter Notebooks pour l’exploration et la modélisation.  
    - Sortie :  
      1. Module Python `data_preprocessing.py` (nettoyage, imputation).  
      2. Notebooks (`01_preprocessing.ipynb`, `02_exploration.ipynb`, `03_modeling.ipynb`).  
      3. Fichier de prédictions `my_pred.csv` (820 lignes, `ad.`/`noad.`).  
  - **Livrables attendus** :  
    - Code Python structuré dans `module/`.  
    - Présentation claire des méthodes et des résultats (rapport Markdown ou PDF).  
    - Tests unitaires pour les fonctions critiques.  (optionnel)
    - Documentation de l’architecture et du flux de données.  


## 2. Schéma général
![Diagramme haut-niveau](assets/archi-overview.png)
> Un diagramme montrant les grandes étapes :  
> `Import → Pré-traitement → EDA → Modélisation → Évaluation → Export de prédictions`

## 3. Arborescence du dépôt
```text
.
├── data/                      # jeux de données bruts et intermédiaires
│   ├── raw/                   # données d’origine
│   └── processed/             # sorties de `handle_missing_values`
├── module/                    # code métier (pré-traitement, EDA, modèles)
│   ├── data_preprocessing.py  # nettoyage & imputation
│   ├── exploratory_analysis.py
│   └── modeling.py
├── notebooks/                 # Jupyter notebooks d’exploration et de rapport
│   ├── 01_preprocessing.ipynb
│   ├── 02_exploration.ipynb
│   └── 03_modeling.ipynb
├── reports/                   # supports finaux (PDF, slides, etc.)
├── scripts/                   # scripts d’exécution batch
│   └── run_all.sh
├── tests/                     # tests unitaires
├── requirements.txt
├── architecture.md            # ce fichier
└── README.md

