# 🎬 ML Engineering Pipeline - Films

## 🚀 Projet Refactorisé et Optimisé

Ce projet de machine learning se concentre exclusivement sur l'analyse et la prédiction de données de films (IMDb et CSV). Toutes les références Twitter ont été supprimées et tous les bugs corrigés.

### ✅ Corrections Effectuées

1. **TypeError résolu**: `can only concatenate str (not "int") to str`
   - Conversion robuste des colonnes numériques avec `pd.to_numeric()`
   - Gestion des erreurs de type dans `feature_engineering.py`

2. **Nettoyage complet du code**
   - Suppression de toutes les références Twitter
   - Focus exclusif sur les données de films
   - Pipeline entièrement fonctionnel

3. **Optimisation des fichiers**
   - Suppression des fichiers de données Twitter
   - Nettoyage du cache Python
   - Suppression des cellules de debug temporaires
   - Documentation consolidée

### 🎯 Fonctionnalités

- 📊 **Collecte de données** : IMDb API et fichiers CSV
- 🧹 **Nettoyage des données** : Robuste avec gestion des types
- 🔧 **Feature Engineering** : Scaling, one-hot, interactions, polynomiales
- 🤖 **Modèles ML** : Random Forest, Gradient Boosting, Linear Regression
- 🌐 **API REST** : FastAPI pour les prédictions
- 📈 **Visualisations** : Matplotlib et Seaborn

### 🗂️ Structure du Projet

```
ml-engineering-pipeline/
├── api/                    # API REST avec FastAPI
├── data/                   # Données (ignorées par Git)
│   ├── raw/               # Données brutes
│   ├── processed/         # Données nettoyées
│   └── features/          # Features générées
├── models/                # Modèles entraînés (ignorés par Git)
├── notebooks/             # Jupyter notebooks
├── src/                   # Code source principal
├── .gitignore            # Configuration Git optimisée
├── requirements.txt      # Dépendances Python
└── launcher.bat         # Script de lancement
```

### 🛡️ Sécurité des Données

- Toutes les données sont protégées par `.gitignore`
- Modèles entraînés exclus du versioning
- Structure des dossiers préservée avec `.gitkeep`

### 🚀 Utilisation

1. **Installation** : `pip install -r requirements.txt`
2. **Notebook** : `jupyter notebook notebooks/data_engineering_ml_pipeline.ipynb`
3. **API** : `python launch_api.py`
4. **Launcher** : `launcher.bat`

### 🎬 Focus Films

Le projet traite uniquement des données de films :
- Informations IMDb (titre, note, votes, genre, acteurs)
- Métadonnées (budget, box office, durée)
- Prédictions de notes et success commercial

**Projet optimisé, nettoyé et entièrement fonctionnel pour l'analyse de films !**
