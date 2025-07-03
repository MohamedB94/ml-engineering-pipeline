# 🎯 Quick Start Guide

## Installation Rapide

```bash
# 1. Cloner le repository
git clone https://github.com/MohamedB94/ml-engineering-pipeline.git
cd ml-engineering-pipeline

# 2. Créer l'environnement virtuel
python -m venv .venv
.venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer le projet
launcher.bat  # ou python launch_api.py
```

## 🚀 Utilisation en 3 minutes

### Option 1 : Notebook Interactif
```bash
jupyter notebook notebooks/data_engineering_ml_pipeline.ipynb
```
👉 Exécuter toutes les cellules pour voir le pipeline complet

### Option 2 : API REST
```bash
python launch_api.py
```
👉 Ouvrir http://localhost:8000/docs pour tester l'API

### Option 3 : Launcher Windows
```bash
launcher.bat
```
👉 Menu interactif pour choisir notebook ou API

## 📊 Données Exemple

Le projet inclut des données d'exemple :
- Films IMDb (Avatar, Titanic, Avengers...)
- Métadonnées (budget, box office, notes)
- Features automatiquement générées

## 🎬 Résultats Attendus

- ✅ Prédiction de notes IMDb
- ✅ Analyse de sentiment des synopsis
- ✅ Visualisations interactives
- ✅ Modèles ML entraînés
- ✅ API prête pour production

**Temps total : ~5 minutes pour un pipeline ML complet !**
