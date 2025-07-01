"""
Module principal de l'API FastAPI.
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

# Ajout du répertoire parent au path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Import des modules locaux
from src.data_cleaning import clean_imdb_data
from src.feature_engineering import engineer_imdb_features

# Chemin des modèles
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Fonction pour charger le modèle le plus récent
def load_latest_model(prefix):
    """
    Charge le modèle le plus récent avec le préfixe spécifié.
    
    Args:
        prefix (str): Préfixe du nom du modèle
        
    Returns:
        tuple: (modèle, métadonnées)
    """
    # Liste des fichiers de modèle
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.joblib') and f.startswith(prefix)]
    
    if not model_files:
        return None, None
    
    # Tri par date (les modèles sont nommés avec un timestamp)
    model_files.sort(reverse=True)
    model_path = os.path.join(MODELS_DIR, model_files[0])
    
    # Chargement du modèle
    model = joblib.load(model_path)
    
    # Chargement des métadonnées si disponibles
    metadata_file = model_files[0].replace('.joblib', '_metadata.json')
    metadata_path = os.path.join(MODELS_DIR, metadata_file)
    
    metadata = None
    if os.path.exists(metadata_path):
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return model, metadata

# Chargement du modèle lors du démarrage
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chargement du modèle IMDb
    app.state.imdb_model, app.state.imdb_metadata = load_latest_model('imdb_rating_predictor')
    
    # Chargement du modèle Twitter
    app.state.twitter_model, app.state.twitter_metadata = load_latest_model('twitter_sentiment_classifier')
    
    yield
    
    # Nettoyage lors de la fermeture
    app.state.imdb_model = None
    app.state.twitter_model = None

# Initialisation de l'application FastAPI
app = FastAPI(
    title="Data Engineering & ML Pipeline API",
    description="API pour l'accès aux modèles entraînés dans notre pipeline de data engineering et machine learning",
    version="1.0.0",
    lifespan=lifespan
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèles Pydantic pour les données
class MovieInput(BaseModel):
    """Modèle pour les entrées de prédiction de film."""
    title: str = Field(..., example="The Shawshank Redemption")
    year: int = Field(..., example=1994)
    director: str = Field(..., example="Frank Darabont")
    actors: str = Field(..., example="Tim Robbins, Morgan Freeman")
    genre: str = Field(..., example="Drama")
    plot: str = Field(..., example="Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.")
    runtime: Optional[int] = Field(None, example=142)

class TweetInput(BaseModel):
    """Modèle pour les entrées de prédiction de sentiment de tweet."""
    text: str = Field(..., example="I love data science and machine learning!")

class PredictionResponse(BaseModel):
    """Modèle pour les réponses de prédiction."""
    prediction: Any
    confidence: Optional[float] = None
    model_info: Dict[str, Any]
    features_used: List[str]
    processing_time: float

# Routes API

@app.get("/")
async def root():
    """Page d'accueil de l'API."""
    return {
        "message": "Bienvenue sur l'API du pipeline de Data Engineering et Machine Learning",
        "docs_url": "/docs",
        "models_available": {
            "imdb_rating_predictor": app.state.imdb_model is not None,
            "twitter_sentiment_classifier": app.state.twitter_model is not None
        }
    }

@app.get("/models")
async def get_models():
    """Liste les modèles disponibles et leurs métadonnées."""
    models = {
        "imdb_rating_predictor": app.state.imdb_metadata,
        "twitter_sentiment_classifier": app.state.twitter_metadata
    }
    return models

@app.get("/models/{model_name}")
async def get_model_info(model_name: str = Path(..., description="Nom du modèle")):
    """Obtient les informations détaillées sur un modèle spécifique."""
    if model_name == "imdb_rating_predictor":
        if app.state.imdb_metadata:
            return app.state.imdb_metadata
        else:
            raise HTTPException(status_code=404, detail="Modèle IMDb non trouvé")
    elif model_name == "twitter_sentiment_classifier":
        if app.state.twitter_metadata:
            return app.state.twitter_metadata
        else:
            raise HTTPException(status_code=404, detail="Modèle Twitter non trouvé")
    else:
        raise HTTPException(status_code=404, detail="Modèle non trouvé")

@app.post("/predict/imdb_rating", response_model=PredictionResponse)
async def predict_imdb_rating(movie: MovieInput):
    """
    Prédit la note IMDb d'un film en fonction de ses caractéristiques.
    """
    if app.state.imdb_model is None:
        raise HTTPException(status_code=503, detail="Modèle IMDb non disponible")
    
    start_time = datetime.now()
    
    # Préparation des données
    movie_data = movie.dict()
    movie_df = pd.DataFrame([movie_data])
    
    # Simulation du pipeline de nettoyage et feature engineering
    # Dans un système de production, cette partie pourrait être optimisée
    # pour éviter de réexécuter tout le pipeline
    
    # Adaptation des colonnes pour correspondre à celles attendues par le modèle
    if 'title' in movie_df.columns and 'Title' not in movie_df.columns:
        movie_df = movie_df.rename(columns={'title': 'Title'})
    if 'year' in movie_df.columns and 'Year' not in movie_df.columns:
        movie_df = movie_df.rename(columns={'year': 'Year'})
    if 'director' in movie_df.columns and 'Director' not in movie_df.columns:
        movie_df = movie_df.rename(columns={'director': 'Director'})
    if 'actors' in movie_df.columns and 'Actors' not in movie_df.columns:
        movie_df = movie_df.rename(columns={'actors': 'Actors'})
    if 'genre' in movie_df.columns and 'Genre' not in movie_df.columns:
        movie_df = movie_df.rename(columns={'genre': 'Genre'})
    if 'plot' in movie_df.columns and 'Plot' not in movie_df.columns:
        movie_df = movie_df.rename(columns={'plot': 'Plot'})
    if 'runtime' in movie_df.columns and 'Runtime' not in movie_df.columns:
        movie_df = movie_df.rename(columns={'runtime': 'Runtime'})
    
    # Récupération des métadonnées du modèle
    features = app.state.imdb_metadata.get('features', []) if app.state.imdb_metadata else []
    
    # Vérification que toutes les caractéristiques nécessaires sont disponibles
    # Si ce n'est pas le cas, on peut générer des valeurs par défaut
    
    # Préparation des données pour la prédiction
    # Dans un cas réel, il faudrait effectuer le même prétraitement que lors de l'entraînement
    X = movie_df.copy()
    
    # Réalisation de la prédiction
    try:
        prediction = app.state.imdb_model.predict(X)[0]
        
        # Si le modèle fournit des probabilités (classification)
        if hasattr(app.state.imdb_model, 'predict_proba'):
            confidence = np.max(app.state.imdb_model.predict_proba(X)[0])
        else:
            confidence = None
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return {
            "prediction": float(prediction),
            "confidence": confidence,
            "model_info": {
                "model_name": "imdb_rating_predictor",
                "model_type": app.state.imdb_metadata.get('model_type', 'unknown') if app.state.imdb_metadata else 'unknown',
                "training_date": app.state.imdb_metadata.get('training_date', 'unknown') if app.state.imdb_metadata else 'unknown'
            },
            "features_used": features,
            "processing_time": processing_time
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

@app.post("/predict/twitter_sentiment", response_model=PredictionResponse)
async def predict_twitter_sentiment(tweet: TweetInput):
    """
    Prédit le sentiment d'un tweet.
    """
    if app.state.twitter_model is None:
        raise HTTPException(status_code=503, detail="Modèle Twitter non disponible")
    
    start_time = datetime.now()
    
    # Préparation des données
    tweet_data = tweet.dict()
    tweet_df = pd.DataFrame([tweet_data])
    
    # Nettoyage du texte du tweet (simplifié)
    import re
    
    def clean_tweet(text):
        # Suppression des mentions (@user)
        text = re.sub(r'@\w+', '', text)
        # Suppression des URL
        text = re.sub(r'http\S+', '', text)
        # Suppression des hashtags
        text = re.sub(r'#\w+', '', text)
        # Suppression des caractères spéciaux et ponctuations
        text = re.sub(r'[^\w\s]', '', text)
        # Suppression des espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    tweet_df['clean_text'] = tweet_df['text'].apply(clean_tweet)
    
    # Récupération des métadonnées du modèle
    features = app.state.twitter_metadata.get('features', []) if app.state.twitter_metadata else []
    
    # Vérification que toutes les caractéristiques nécessaires sont disponibles
    # Si ce n'est pas le cas, on peut générer des valeurs par défaut
    
    # Préparation des données pour la prédiction
    # Dans un cas réel, il faudrait effectuer le même prétraitement que lors de l'entraînement
    
    # Simulation des caractéristiques générées lors du feature engineering
    tweet_df['word_count'] = tweet_df['clean_text'].apply(lambda x: len(str(x).split()))
    tweet_df['has_question'] = tweet_df['clean_text'].apply(lambda x: 1 if '?' in str(x) else 0)
    tweet_df['has_exclamation'] = tweet_df['clean_text'].apply(lambda x: 1 if '!' in str(x) else 0)
    tweet_df['capital_letter_ratio'] = tweet_df['clean_text'].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1))
    
    # Sélection des caractéristiques utilisées par le modèle
    X = tweet_df.copy()
    
    # Réalisation de la prédiction
    try:
        prediction = app.state.twitter_model.predict(X)[0]
        
        # Si le modèle fournit des probabilités (classification)
        if hasattr(app.state.twitter_model, 'predict_proba'):
            proba = app.state.twitter_model.predict_proba(X)[0]
            confidence = float(np.max(proba))
            # Récupération des classes pour l'interprétation
            classes = app.state.twitter_model.classes_
            class_probabilities = {str(classes[i]): float(proba[i]) for i in range(len(classes))}
        else:
            confidence = None
            class_probabilities = None
        
        # Conversion du résultat numérique en libellé
        sentiment_label = "positif" if prediction == 1 else "négatif"
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return {
            "prediction": sentiment_label,
            "confidence": confidence,
            "model_info": {
                "model_name": "twitter_sentiment_classifier",
                "model_type": app.state.twitter_metadata.get('model_type', 'unknown') if app.state.twitter_metadata else 'unknown',
                "training_date": app.state.twitter_metadata.get('training_date', 'unknown') if app.state.twitter_metadata else 'unknown',
                "class_probabilities": class_probabilities
            },
            "features_used": features,
            "processing_time": processing_time
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

# Démarrage de l'application si ce script est exécuté directement
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
