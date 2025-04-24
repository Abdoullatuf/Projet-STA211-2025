import nbformat as nbf
nb = nbf.v4.new_notebook()

# Title and Introduction with Navigation Menu
nb.cells.append(nbf.v4.new_markdown_cell("""# Advertisement Click Prediction Modeling

Ce notebook se concentre sur le développement et l'évaluation de modèles de machine learning pour prédire les clics sur les publicités.

<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin: 10px 0;">
    <h2 style="color: #2874A6;">Table des Matières</h2>
    <ul style="list-style-type: none;">
        <li><a href="#data_loading">1. Chargement et Préparation des Données</a></li>
        <li><a href="#model_dev">2. Développement des Modèles</a></li>
        <li><a href="#model_eval">3. Évaluation des Modèles</a></li>
        <li><a href="#hyperopt">4. Optimisation des Hyperparamètres</a></li>
        <li><a href="#final_model">5. Modèle Final et Prédictions</a></li>
    </ul>
</div>

<div style="background-color: #e8f4f9; padding: 10px; border-radius: 5px; margin: 10px 0;">
    <p><strong>Navigation:</strong> Utilisez les liens ci-dessus pour naviguer dans le notebook. À la fin de chaque section, vous trouverez un lien pour revenir à la table des matières.</p>
</div>
"""))

# Import Libraries
nb.cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
"""))

# Data Loading Section
nb.cells.append(nbf.v4.new_markdown_cell("""<a id="data_loading"></a>
## 1. Chargement et Préparation des Données

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px;">
    <p><strong>Objectifs de cette section:</strong></p>
    <ul>
        <li>Charger les données prétraitées</li>
        <li>Séparer les features et la variable cible</li>
        <li>Diviser en ensembles d'entraînement et de test</li>
        <li>Standardiser les features</li>
        <li>Gérer le déséquilibre des classes avec SMOTE</li>
    </ul>
</div>
"""))

nb.cells.append(nbf.v4.new_code_cell("""# Load the imputed data
df = pd.read_csv('data_train_processed.csv')

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""<div style="text-align: right"><a href="#top">Retour en haut ↑</a></div>"""))

# Model Development Section
nb.cells.append(nbf.v4.new_markdown_cell("""<a id="model_dev"></a>
## 2. Développement des Modèles

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px;">
    <p><strong>Modèles implémentés:</strong></p>
    <ul>
        <li>Régression Logistique</li>
        <li>Random Forest</li>
        <li>Gradient Boosting</li>
        <li>XGBoost</li>
    </ul>
</div>
"""))

nb.cells.append(nbf.v4.new_code_cell("""# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test_scaled)
    results[name] = {
        'accuracy': model.score(X_test_scaled, y_test),
        'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]),
        'classification_report': classification_report(y_test, y_pred)
    }
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""<div style="text-align: right"><a href="#top">Retour en haut ↑</a></div>"""))

# Model Evaluation Section
nb.cells.append(nbf.v4.new_markdown_cell("""<a id="model_eval"></a>
## 3. Évaluation des Modèles

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px;">
    <p><strong>Métriques d'évaluation:</strong></p>
    <ul>
        <li>Accuracy</li>
        <li>ROC AUC Score</li>
        <li>Classification Report (Precision, Recall, F1-Score)</li>
    </ul>
</div>
"""))

nb.cells.append(nbf.v4.new_code_cell("""# Compare model performances
performance_df = pd.DataFrame({
    name: {
        'Accuracy': results[name]['accuracy'],
        'ROC AUC': results[name]['roc_auc']
    }
    for name in models.keys()
}).T

# Plot results
plt.figure(figsize=(10, 6))
performance_df.plot(kind='bar')
plt.title('Comparaison des Performances des Modèles')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""<div style="text-align: right"><a href="#top">Retour en haut ↑</a></div>"""))

# Hyperparameter Optimization Section
nb.cells.append(nbf.v4.new_markdown_cell("""<a id="hyperopt"></a>
## 4. Optimisation des Hyperparamètres

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px;">
    <p><strong>Processus d'optimisation:</strong></p>
    <ul>
        <li>Définition des grilles de paramètres</li>
        <li>Validation croisée (5-fold)</li>
        <li>Sélection du meilleur modèle</li>
    </ul>
</div>
"""))

nb.cells.append(nbf.v4.new_code_cell("""# Define parameter grids for the best performing model
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train_balanced, y_train_balanced)

print("Meilleurs paramètres:", grid_search.best_params_)
print("Meilleur score:", grid_search.best_score_)
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""<div style="text-align: right"><a href="#top">Retour en haut ↑</a></div>"""))

# Final Model Section
nb.cells.append(nbf.v4.new_markdown_cell("""<a id="final_model"></a>
## 5. Modèle Final et Prédictions

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px;">
    <p><strong>Étapes finales:</strong></p>
    <ul>
        <li>Entraînement du modèle final</li>
        <li>Évaluation des performances</li>
        <li>Analyse des features importantes</li>
        <li>Génération des prédictions</li>
    </ul>
</div>
"""))

nb.cells.append(nbf.v4.new_code_cell("""# Train final model with best parameters
final_model = RandomForestClassifier(**grid_search.best_params_, random_state=42)
final_model.fit(X_train_balanced, y_train_balanced)

# Make predictions
y_pred_final = final_model.predict(X_test_scaled)
y_pred_proba_final = final_model.predict_proba(X_test_scaled)[:, 1]

# Print final results
print("Classification Report:")
print(classification_report(y_test, y_pred_final))

print("\\nROC AUC Score:", roc_auc_score(y_test, y_pred_proba_final))

# Plot feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
plt.title('Top 20 Variables les Plus Importantes')
plt.xlabel('Importance')
plt.tight_layout()
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""<div style="text-align: right"><a href="#top">Retour en haut ↑</a></div>"""))

# Save the notebook
with open('02_Modeling.ipynb', 'w') as f:
    nbf.write(nb, f) 