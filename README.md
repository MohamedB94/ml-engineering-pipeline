# ML Engineering Pipeline

Ce projet implémente un pipeline complet de data engineering et machine learning avec les étapes suivantes :

1. **Collecte de données** : Extraction de données depuis différentes sources (IMDb, Twitter, CSV)
2. **Nettoyage des données** : Prétraitement et nettoyage
3. **Feature Engineering** : Création de nouvelles caractéristiques pour améliorer les modèles
4. **Entraînement de modèle** : Utilisation de scikit-learn pour entraîner un modèle ML
5. **Stockage du modèle** : Sauvegarde avec joblib
6. **Déploiement API** : Exposition du modèle via FastAPI

## Structure du projet

```
│
├── data/                       # Dossier de données
│   ├── raw/                    # Données brutes
│   └── processed/              # Données traitées
│
├── src/                        # Code source
│   ├── data_collection.py      # Scripts de collecte de données
│   ├── data_cleaning.py        # Scripts de nettoyage
│   ├── feature_engineering.py  # Scripts pour la création de features
│   └── model_training.py       # Scripts d'entraînement de modèle
│
├── models/                     # Modèles entraînés
│
├── api/                        # Code de l'API
│   ├── main.py                 # Point d'entrée de l'API
│   └── endpoints.py            # Endpoints de l'API
│
├── notebooks/                  # Jupyter notebooks pour l'exploration
│
├── requirements.txt            # Dépendances du projet
│
└── .gitignore                  # Fichiers à ignorer par Git
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

1. Collecte de données :
```bash
python src/data_collection.py
```

2. Nettoyage et feature engineering :
```bash
python src/data_cleaning.py
python src/feature_engineering.py
```

3. Entraînement du modèle :
```bash
python src/model_training.py
```

4. Démarrage de l'API :
```bash
cd api
uvicorn main:app --reload
```

## Accès à l'API

Une fois démarrée, l'API est accessible à l'adresse : http://localhost:8000

Documentation de l'API : http://localhost:8000/docs
