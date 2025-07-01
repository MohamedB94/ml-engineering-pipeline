"""
Module de collecte de données provenant de différentes sources :
- IMDb via API/scraping
- Twitter via API
- Fichiers CSV locaux
"""
import os
import json
import pandas as pd
import requests
from bs4 import BeautifulSoup
import tweepy
from dotenv import load_dotenv

# Chargement des variables d'environnement pour les clés API
load_dotenv()

# Chemin pour sauvegarder les données
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')


def collect_imdb_data(movie_ids=None):
    """
    Collecte des données de films depuis IMDb.
    
    Args:
        movie_ids (list): Liste d'identifiants IMDb de films
        
    Returns:
        pandas.DataFrame: Données de films
    """
    if not movie_ids:
        # Liste par défaut de films populaires
        movie_ids = ['tt0111161', 'tt0068646', 'tt0071562', 'tt0468569', 'tt0050083']
    
    movies_data = []
    
    for movie_id in movie_ids:
        # Exemple d'utilisation de l'API OMDb (nécessite une clé API)
        api_key = os.getenv('OMDB_API_KEY', '')
        if api_key:
            response = requests.get(f'http://www.omdbapi.com/?i={movie_id}&apikey={api_key}')
            if response.status_code == 200:
                movie_data = response.json()
                movies_data.append(movie_data)
                print(f"Données récupérées pour le film: {movie_data.get('Title', movie_id)}")
        else:
            print("Clé API OMDb non configurée. Utilisez le scraping comme alternative.")
            # Comme alternative, on pourrait implémenter du scraping ici
    
    # Sauvegarde des données brutes
    if movies_data:
        df = pd.DataFrame(movies_data)
        output_path = os.path.join(RAW_DATA_DIR, 'imdb_data.csv')
        df.to_csv(output_path, index=False)
        print(f"Données IMDb sauvegardées dans {output_path}")
        return df
    
    return pd.DataFrame()


def collect_twitter_data(query, count=100):
    """
    Collecte des données Twitter pour une requête donnée.
    
    Args:
        query (str): Terme de recherche
        count (int): Nombre de tweets à récupérer
        
    Returns:
        pandas.DataFrame: Données de tweets
    """
    # Authentification Twitter
    consumer_key = os.getenv('TWITTER_CONSUMER_KEY')
    consumer_secret = os.getenv('TWITTER_CONSUMER_SECRET')
    access_token = os.getenv('TWITTER_ACCESS_TOKEN')
    access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
    
    if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
        print("Informations d'authentification Twitter manquantes.")
        return pd.DataFrame()
    
    auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
    api = tweepy.API(auth)
    
    try:
        # Récupération des tweets
        tweets = api.search_tweets(q=query, count=count, tweet_mode='extended')
        
        # Extraction des données pertinentes
        tweets_data = []
        for tweet in tweets:
            tweet_data = {
                'id': tweet.id_str,
                'created_at': tweet.created_at,
                'text': tweet.full_text,
                'user': tweet.user.screen_name,
                'retweets': tweet.retweet_count,
                'favorites': tweet.favorite_count,
                'hashtags': [hashtag['text'] for hashtag in tweet.entities['hashtags']]
            }
            tweets_data.append(tweet_data)
        
        # Création du DataFrame
        df = pd.DataFrame(tweets_data)
        
        # Sauvegarde des données
        output_path = os.path.join(RAW_DATA_DIR, f'twitter_{query.replace(" ", "_")}.csv')
        df.to_csv(output_path, index=False)
        print(f"Données Twitter pour '{query}' sauvegardées dans {output_path}")
        
        return df
    
    except Exception as e:
        print(f"Erreur lors de la collecte des tweets: {str(e)}")
        return pd.DataFrame()


def load_csv_data(file_path):
    """
    Charge des données depuis un fichier CSV.
    
    Args:
        file_path (str): Chemin vers le fichier CSV
        
    Returns:
        pandas.DataFrame: Données du CSV
    """
    try:
        df = pd.read_csv(file_path)
        
        # Copie dans le dossier de données brutes
        filename = os.path.basename(file_path)
        output_path = os.path.join(RAW_DATA_DIR, filename)
        
        # Ne copie que si le fichier n'est pas déjà dans le dossier raw
        if file_path != output_path:
            df.to_csv(output_path, index=False)
            print(f"Fichier CSV copié dans {output_path}")
        
        return df
    
    except Exception as e:
        print(f"Erreur lors du chargement du fichier CSV: {str(e)}")
        return pd.DataFrame()


if __name__ == "__main__":
    # Création du dossier de données brutes s'il n'existe pas
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # Exemple de collecte de données IMDb
    imdb_data = collect_imdb_data()
    
    # Exemple de collecte de données Twitter
    twitter_data = collect_twitter_data("data science")
    
    # Exemple de chargement de CSV (remplacer par votre propre chemin)
    # csv_path = "path/to/your/data.csv"
    # csv_data = load_csv_data(csv_path)
    
    print("Collecte de données terminée.")
