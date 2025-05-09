# STA211 – Internet Advertisements Challenge (UE STA211, CNAM – Master Science des Données)

## 1. Contexte  
Ce challenge est réalisé dans le cadre de l’UE STA211 du Conservatoire National des Arts et Métiers (CNAM), pour le Master Sciences des Données. L’objectif est de construire un modèle capable de distinguer automatiquement une page web « publicité » d’une page « non-publicité », à partir du jeu de données **Internet Advertisements** (820 observations, plusieurs dizaines de variables numériques et catégorielles).

### Contraintes  
- **Format** : CSV (`data_train.csv` / `data_test.csv`), colonnes explicatives + `outcome` (`ad.` / `noad.`).  
- **Livrables** :  
  1. Un **Jupyter Notebook** commenté (exploration, pré-traitement, modélisation, évaluation).  
  2. Un module Python `data_preprocessing.py` pour le pré-traitement et l’imputation.  
  3. Un diagramme d’architecture (`assets/archi-overview.png`) référencé dans `architecture.md`.  
  4. Un script ou notebook produisant un fichier de soumission `my_pred.csv` (820 lignes, valeurs `ad.`/`noad.`).  

## 2. Objectifs  
- **Pré-traitement**  
  - Nettoyage, gestion des valeurs manquantes (médiane, KNN, multiple).  
  - Transformation (log, Box-Cox), discrétisation, encodage.  
- **Exploration**  
  - Analyses univariée, bivariée, multivariée (ACP, AFM, clustering).  
- **Modélisation**  
  - Régression logistique, arbres CART, forêts aléatoires, éventuellement SVM/KNN.  
  - Validation croisée ou split 80/20 stratifié.  
  - Gestion du déséquilibre (pondération, SMOTE).  
- **Évaluation & livraison**  
  - Précision, rappel, F1-score, AUC, matrice de confusion.  
  - Export des prédictions sous `my_pred.csv` selon le format requis.

## 3. Architecture haute-niveau  
![Diagramme haut-niveau](assets/archi-overview.png)

> Voir **architecture.md** pour le détail de chaque étape.

## 4. Arborescence du projet  
```text
├── README.md
├── architecture.md
├── assets/
│   └── archi-overview.png
├── data/
│   ├── data_train.csv
│   └── data_test.csv
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── exploratory_analysis.py
│   └── modeling.py
├── requirements.txt
└── my_pred.csv        # Exemple de fichier de soumission

