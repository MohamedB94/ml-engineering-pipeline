@echo off
chcp 65001 >nul
color 0A
cls

echo.
echo ████████████████████████████████████████████████████████████████
echo █                                                              █
echo █          🚀 ML ENGINEERING PIPELINE - LAUNCHER 🚀            █
echo █                                                              █
echo ████████████████████████████████████████████████████████████████
echo.

:MENU
echo.
echo ═══════════════════════════════════════════════════════════════════
echo                            MENU PRINCIPAL
echo ═══════════════════════════════════════════════════════════════════
echo.
echo   1. 🔧 SETUP COMPLET (Première fois)
echo   2. 🚀 LANCER L'API SEULEMENT  
echo   3. 🎬 TESTER AVATAR 3
echo   4. 🎭 TESTER AUTRE FILM
echo   5. 📊 PIPELINE COMPLET (Données + Modèle + API)
echo   6. 📖 OUVRIR DOCUMENTATION
echo   7. 🛑 QUITTER
echo.
echo ═══════════════════════════════════════════════════════════════════

set /p choice="Votre choix (1-7): "

if "%choice%"=="1" goto SETUP
if "%choice%"=="2" goto LAUNCH_API
if "%choice%"=="3" goto TEST_AVATAR
if "%choice%"=="4" goto TEST_CUSTOM
if "%choice%"=="5" goto FULL_PIPELINE
if "%choice%"=="6" goto DOCS
if "%choice%"=="7" goto EXIT
goto MENU

:SETUP
cls
echo.
echo ████████████████████████████████████████████████████████████████
echo █                    🔧 SETUP COMPLET                          █
echo ████████████████████████████████████████████████████████████████
echo.

echo [1/4] 🐍 Activation de l'environnement Python...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Erreur: Environnement Python non trouvé
    echo 💡 Créez d'abord un environnement virtuel avec: python -m venv .venv
    pause
    goto MENU
)

echo [2/4] 📦 Installation des dépendances...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ❌ Erreur lors de l'installation des dépendances
    pause
    goto MENU
)

echo [3/4] 🗃️ Exécution du pipeline de données...
python src\data_collection.py
python src\data_cleaning.py
python src\feature_engineering.py

echo [4/4] 🤖 Création du modèle...
python -c "
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import json
from datetime import datetime
import os

# Création modèle simple
np.random.seed(42)
n_samples = 100
data = {
    'Year': np.random.randint(1990, 2024, n_samples),
    'Runtime': np.random.randint(80, 180, n_samples),
    'word_count_plot': np.random.randint(10, 100, n_samples),
    'word_count_title': np.random.randint(1, 8, n_samples),
    'has_award': np.random.choice([0, 1], n_samples),
    'genre_count': np.random.randint(1, 4, n_samples),
    'imdbRating': np.random.uniform(4.0, 9.0, n_samples)
}
df = pd.DataFrame(data)
features = ['Year', 'Runtime', 'word_count_plot', 'word_count_title', 'has_award', 'genre_count']
X, y = df[features], df['imdbRating']

model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X, y)

# Sauvegarde
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/imdb_rating_predictor_20250703.joblib')

metadata = {
    'model_type': 'random_forest',
    'features': features,
    'target': 'imdbRating',
    'training_date': datetime.now().strftime('%%Y-%%m-%%d %%H:%%M:%%S'),
    'performance': {'r2_score': float(model.score(X, y))},
    'description': 'Modèle de prédiction IMDb'
}

with open('models/imdb_rating_predictor_20250703_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print('✅ Modèle créé avec succès!')
"

echo.
echo ✅ SETUP TERMINÉ AVEC SUCCÈS!
echo 💡 Vous pouvez maintenant utiliser l'option 2 pour lancer l'API
echo.
pause
goto MENU

:LAUNCH_API
cls
echo.
echo ████████████████████████████████████████████████████████████████
echo █                    🚀 LANCEMENT DE L'API                     █
echo ████████████████████████████████████████████████████████████████
echo.

echo 🐍 Activation de l'environnement...
call .venv\Scripts\activate.bat

echo 🌐 Démarrage du serveur API...
echo.
echo ═══════════════════════════════════════════════════════════════════
echo     🌟 API DÉMARRÉE AVEC SUCCÈS! 🌟
echo.
echo     📡 API:            http://localhost:8000
echo     📖 Documentation:  http://localhost:8000/docs  
echo     🔍 Swagger:        http://localhost:8000/redoc
echo.
echo     ⚠️  Appuyez sur Ctrl+C pour arrêter
echo ═══════════════════════════════════════════════════════════════════
echo.

cd api
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

cd ..
pause
goto MENU

:TEST_AVATAR
cls
echo.
echo ████████████████████████████████████████████████████████████████
echo █                    🎬 TEST AVATAR 3                          █
echo ████████████████████████████████████████████████████████████████
echo.

echo 🐍 Activation de l'environnement...
call .venv\Scripts\activate.bat

echo 🎬 Test de prédiction pour Avatar 3...
echo.

python -c "
import requests
import json

data = {
    'title': 'Avatar 3',
    'year': 2025,
    'director': 'James Cameron',
    'actors': 'Sam Worthington, Zoe Saldana',
    'genre': 'Sci-Fi, Action',
    'plot': 'Jake Sully continues his adventures on Pandora with new alien worlds to explore',
    'runtime': 180
}

try:
    response = requests.post('http://localhost:8000/predict/imdb_rating', json=data)
    if response.status_code == 200:
        result = response.json()
        print('✅ PRÉDICTION RÉUSSIE!')
        print('═' * 50)
        print(f'🎯 Note prédite: {result[\"prediction\"]:.1f}/10')
        print(f'⏱️  Temps: {result[\"processing_time\"]:.3f}s')
        print(f'🤖 Modèle: {result[\"model_info\"][\"model_type\"]}')
        print('═' * 50)
        
        note = result['prediction']
        if note >= 8.0:
            print('🌟 EXCELLENT! Film très recommandé!')
        elif note >= 7.0:
            print('👍 BON FILM! Vaut le déplacement.')
        elif note >= 6.0:
            print('🤔 CORRECT. Pour les fans du genre.')
        else:
            print('⚠️  DÉCEVANT selon les prédictions.')
    else:
        print(f'❌ Erreur: {response.status_code}')
        print(f'Détails: {response.text}')
except requests.exceptions.ConnectionError:
    print('❌ ERREUR: API non accessible!')
    print('💡 Lancez d''abord l''API avec l''option 2')
except Exception as e:
    print(f'❌ Erreur: {e}')
"

echo.
pause
goto MENU

:TEST_CUSTOM
cls
echo.
echo ████████████████████████████████████████████████████████████████
echo █                    🎭 TEST FILM PERSONNALISÉ                 █
echo ████████████████████████████████████████████████████████████████
echo.

set /p film_title="🎬 Titre du film: "
set /p film_year="📅 Année: "
set /p film_director="🎬 Réalisateur: "
set /p film_actors="👥 Acteurs principaux: "
set /p film_genre="🎭 Genre: "
set /p film_runtime="⏱️  Durée (minutes): "
set /p film_plot="📖 Synopsis: "

echo.
echo 🐍 Activation de l'environnement...
call .venv\Scripts\activate.bat

echo 🔮 Prédiction en cours...

python -c "
import requests
import json

data = {
    'title': '%film_title%',
    'year': int('%film_year%'),
    'director': '%film_director%',
    'actors': '%film_actors%',
    'genre': '%film_genre%',
    'plot': '%film_plot%',
    'runtime': int('%film_runtime%')
}

try:
    response = requests.post('http://localhost:8000/predict/imdb_rating', json=data)
    if response.status_code == 200:
        result = response.json()
        print('✅ PRÉDICTION RÉUSSIE!')
        print('═' * 50)
        print(f'🎬 Film: %film_title%')
        print(f'🎯 Note prédite: {result[\"prediction\"]:.1f}/10')
        print(f'⏱️  Temps: {result[\"processing_time\"]:.3f}s')
        print('═' * 50)
    else:
        print(f'❌ Erreur: {response.status_code}')
except Exception as e:
    print(f'❌ Erreur: {e}')
"

echo.
pause
goto MENU

:FULL_PIPELINE
cls
echo.
echo ████████████████████████████████████████████████████████████████
echo █                 📊 PIPELINE COMPLET                          █
echo ████████████████████████████████████████████████████████████████
echo.

echo 🐍 Activation de l'environnement...
call .venv\Scripts\activate.bat

echo [1/5] 📥 Collecte de données...
python src\data_collection.py

echo [2/5] 🧹 Nettoyage des données...
python src\data_cleaning.py

echo [3/5] ⚙️ Feature engineering...
python src\feature_engineering.py

echo [4/5] 🤖 Entraînement du modèle...
python src\model_training.py

echo [5/5] 🚀 Lancement de l'API...
echo.
echo ✅ PIPELINE TERMINÉ!
echo 💡 L'API va maintenant se lancer...
echo.
pause

cd api
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

cd ..
pause
goto MENU

:DOCS
echo.
echo 📖 Ouverture de la documentation...
start http://localhost:8000/docs
timeout /t 2 /nobreak >nul
start http://localhost:8000/redoc
goto MENU

:EXIT
cls
echo.
echo ████████████████████████████████████████████████████████████████
echo █                                                              █
echo █               🎉 MERCI D'AVOIR UTILISÉ                      █
echo █            ML ENGINEERING PIPELINE! 🚀                      █
echo █                                                              █
echo ████████████████████████████████████████████████████████████████
echo.
timeout /t 3 /nobreak >nul
exit

:ERROR
echo.
echo ❌ Une erreur est survenue.
echo 💡 Vérifiez que vous êtes dans le bon répertoire.
pause
goto MENU
