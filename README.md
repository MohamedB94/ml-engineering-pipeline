# 🎬 ML Engineering Pipeline - Film Analysis

[![GitHub](https://img.shields.io/badge/GitHub-MohamedB94%2Fml--engineering--pipeline-blue?style=flat-square&logo=github)](https://github.com/MohamedB94/ml-engineering-pipeline)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success?style=flat-square)](https://github.com/MohamedB94/ml-engineering-pipeline)

Un pipeline complet de machine learning pour l'analyse et la prédiction de données de films.

## 🚀 Vue d'ensemble

Ce projet implémente un pipeline end-to-end pour :
- Collecte de données de films (IMDb, CSV)
- Nettoyage et preprocessing des données
- Feature engineering avancé
- Entraînement de modèles ML
- API REST pour les prédictions
- Visualisations et analyses

## ✨ Fonctionnalités

### 📊 Données Supportées
- **IMDb** : Films, notes, votes, genres, acteurs
- **CSV** : Budget, box office, durée, métadonnées

### 🔧 Techniques ML
- Preprocessing robuste avec gestion des types
- Feature engineering (scaling, one-hot, interactions)
- Modèles : Random Forest, Gradient Boosting, Régression
- Validation croisée et métriques de performance

### 🌐 API REST
- Prédictions en temps réel
- Interface FastAPI
- Documentation automatique

## 🛠️ Installation

```bash
# Cloner le repository
git clone https://github.com/MohamedB94/ml-engineering-pipeline.git
cd ml-engineering-pipeline

# Créer un environnement virtuel
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt
```

## 🚀 Utilisation

### 1. Notebook Jupyter
```bash
jupyter notebook notebooks/data_engineering_ml_pipeline.ipynb
```

### 2. API REST
```bash
python launch_api.py
```
Accéder à : http://localhost:8000/docs

### 3. Launcher Windows
```bash
launcher.bat
```
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
