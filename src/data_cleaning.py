"""
Module de nettoyage et prétraitement des données collectées.
"""
import os
import pandas as pd
import numpy as np
import re
from datetime import datetime

# Chemins des répertoires
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def clean_imdb_data(input_file='imdb_data.csv'):
    """
    Nettoie et transforme les données IMDb.
    
    Args:
        input_file (str): Nom du fichier d'entrée dans le répertoire raw
        
    Returns:
        pandas.DataFrame: Données nettoyées
    """
    # Vérification de l'existence du fichier
    file_path = os.path.join(RAW_DATA_DIR, input_file)
    if not os.path.exists(file_path):
        print(f"Fichier {file_path} introuvable.")
        return None
    
    # Chargement des données
    df = pd.read_csv(file_path)
    print(f"Nettoyage des données IMDb: {df.shape[0]} lignes chargées.")
    
    # Conversion des types de données
    if 'Year' in df.columns:
        # Extraction de l'année à partir du format "2021–2022" ou "(2021)"
        df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})').astype(float)
    
    if 'imdbRating' in df.columns:
        df['imdbRating'] = pd.to_numeric(df['imdbRating'], errors='coerce')
    
    if 'imdbVotes' in df.columns:
        # Conversion des votes de format "1,234,567" en nombre
        df['imdbVotes'] = df['imdbVotes'].str.replace(',', '').astype(float)
    
    if 'Runtime' in df.columns:
        # Extraction du temps en minutes à partir de "120 min"
        df['Runtime'] = df['Runtime'].str.extract(r'(\d+)').astype(float)
    
    # Gestion des valeurs manquantes
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        # Remplacement des valeurs manquantes par la médiane
        df[col] = df[col].fillna(df[col].median())
    
    # Nettoyage des colonnes textuelles
    text_columns = df.select_dtypes(include=['object']).columns
    for col in text_columns:
        # Remplacement des valeurs N/A par une chaîne vide
        df[col] = df[col].fillna('')
        
        # Suppression des caractères spéciaux inutiles
        if col in ['Plot', 'Title', 'Director', 'Actors']:
            df[col] = df[col].apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x)).strip())
    
    # Sauvegarde des données nettoyées
    output_path = os.path.join(PROCESSED_DATA_DIR, 'clean_imdb_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Données IMDb nettoyées sauvegardées dans {output_path}")
    
    return df

def clean_twitter_data(input_file):
    """
    Nettoie et transforme les données Twitter.
    
    Args:
        input_file (str): Nom du fichier d'entrée dans le répertoire raw
        
    Returns:
        pandas.DataFrame: Données nettoyées
    """
    # Vérification de l'existence du fichier
    file_path = os.path.join(RAW_DATA_DIR, input_file)
    if not os.path.exists(file_path):
        print(f"Fichier {file_path} introuvable.")
        return None
    
    # Chargement des données
    df = pd.read_csv(file_path)
    print(f"Nettoyage des données Twitter: {df.shape[0]} lignes chargées.")
    
    # Conversion de la date en format datetime
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    
    # Nettoyage du texte des tweets
    if 'text' in df.columns:
        # Fonction pour nettoyer le texte
        def clean_tweet(tweet):
            # Suppression des mentions (@user)
            tweet = re.sub(r'@\w+', '', tweet)
            # Suppression des URL
            tweet = re.sub(r'http\S+', '', tweet)
            # Suppression des hashtags
            tweet = re.sub(r'#\w+', '', tweet)
            # Suppression des caractères spéciaux et ponctuations
            tweet = re.sub(r'[^\w\s]', '', tweet)
            # Suppression des espaces multiples
            tweet = re.sub(r'\s+', ' ', tweet).strip()
            return tweet
        
        df['clean_text'] = df['text'].apply(clean_tweet)
    
    # Conversion des métriques en numériques
    for col in ['retweets', 'favorites']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Création d'une colonne pour la longueur du tweet
    if 'text' in df.columns:
        df['tweet_length'] = df['text'].apply(len)
    
    # Sauvegarde des données nettoyées
    output_file = 'clean_' + input_file
    output_path = os.path.join(PROCESSED_DATA_DIR, output_file)
    df.to_csv(output_path, index=False)
    print(f"Données Twitter nettoyées sauvegardées dans {output_path}")
    
    return df

def clean_csv_data(input_file, date_columns=None, numeric_columns=None, categorical_columns=None):
    """
    Nettoie et transforme des données génériques depuis un CSV.
    
    Args:
        input_file (str): Nom du fichier d'entrée dans le répertoire raw
        date_columns (list): Liste des colonnes à convertir en dates
        numeric_columns (list): Liste des colonnes à convertir en numériques
        categorical_columns (list): Liste des colonnes à convertir en catégorielles
        
    Returns:
        pandas.DataFrame: Données nettoyées
    """
    # Vérification de l'existence du fichier
    file_path = os.path.join(RAW_DATA_DIR, input_file)
    if not os.path.exists(file_path):
        print(f"Fichier {file_path} introuvable.")
        return None
    
    # Chargement des données
    df = pd.read_csv(file_path)
    print(f"Nettoyage des données CSV: {df.shape[0]} lignes chargées.")
    
    # Gestion des valeurs manquantes
    df = df.dropna(how='all')  # Suppression des lignes entièrement vides
    
    # Conversion des colonnes de dates
    if date_columns:
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Conversion des colonnes numériques
    if numeric_columns:
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Remplacement des valeurs manquantes par la médiane
                df[col] = df[col].fillna(df[col].median())
    
    # Conversion des colonnes catégorielles
    if categorical_columns:
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')
    
    # Sauvegarde des données nettoyées
    output_file = 'clean_' + input_file
    output_path = os.path.join(PROCESSED_DATA_DIR, output_file)
    df.to_csv(output_path, index=False)
    print(f"Données CSV nettoyées sauvegardées dans {output_path}")
    
    return df

if __name__ == "__main__":
    # Création du dossier des données traitées s'il n'existe pas
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Nettoyage des données IMDb
    clean_imdb_data()
    
    # Nettoyage des données Twitter
    twitter_files = [f for f in os.listdir(RAW_DATA_DIR) if f.startswith('twitter_')]
    for file in twitter_files:
        clean_twitter_data(file)
    
    # Exemple de nettoyage de CSV générique
    # clean_csv_data('your_data.csv', 
    #                date_columns=['date_column'], 
    #                numeric_columns=['numeric_column1', 'numeric_column2'],
    #                categorical_columns=['category_column'])
    
    print("Nettoyage des données terminé.")
