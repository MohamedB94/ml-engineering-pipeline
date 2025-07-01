"""
Module pour le feature engineering des données nettoyées.
Création de nouvelles caractéristiques pour améliorer les performances des modèles.
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import re
from datetime import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Téléchargement des ressources NLTK nécessaires
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    print("Avertissement: Impossible de télécharger les ressources NLTK. Certaines fonctionnalités pourraient ne pas être disponibles.")

# Chemins des répertoires
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
FEATURES_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, 'features')

def engineer_imdb_features(input_file='clean_imdb_data.csv'):
    """
    Crée des caractéristiques avancées à partir des données IMDb nettoyées.
    
    Args:
        input_file (str): Nom du fichier d'entrée dans le répertoire processed
        
    Returns:
        pandas.DataFrame: Données avec nouvelles caractéristiques
    """
    # Vérification de l'existence du fichier
    file_path = os.path.join(PROCESSED_DATA_DIR, input_file)
    if not os.path.exists(file_path):
        print(f"Fichier {file_path} introuvable.")
        return None
    
    # Chargement des données
    df = pd.read_csv(file_path)
    print(f"Feature engineering pour données IMDb: {df.shape[0]} lignes chargées.")
    
    # Création de nouvelles caractéristiques
    
    # 1. Âge du film (années depuis sa sortie)
    if 'Year' in df.columns:
        current_year = datetime.now().year
        df['movie_age'] = current_year - df['Year']
    
    # 2. Analyse du titre
    if 'Title' in df.columns:
        df['title_length'] = df['Title'].apply(len)
        df['title_word_count'] = df['Title'].apply(lambda x: len(str(x).split()))
    
    # 3. Génération de caractéristiques à partir du genre
    if 'Genre' in df.columns:
        # One-hot encoding des genres
        genres = df['Genre'].str.split(',', expand=True).stack()
        genres = genres.str.strip()
        genre_dummies = pd.get_dummies(genres, prefix='genre')
        genre_dummies = genre_dummies.groupby(level=0).sum()
        df = pd.concat([df, genre_dummies], axis=1)
    
    # 4. Nombre d'acteurs
    if 'Actors' in df.columns:
        df['actor_count'] = df['Actors'].apply(lambda x: len(str(x).split(',')))
    
    # 5. Ratio votes/note
    if 'imdbRating' in df.columns and 'imdbVotes' in df.columns:
        df['rating_votes_ratio'] = df['imdbRating'] / (df['imdbVotes'] + 1)  # Éviter division par zéro
    
    # 6. Analyse de sentiment pour le résumé (Plot)
    if 'Plot' in df.columns:
        try:
            sia = SentimentIntensityAnalyzer()
            df['plot_sentiment'] = df['Plot'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
            df['plot_length'] = df['Plot'].apply(len)
            df['plot_word_count'] = df['Plot'].apply(lambda x: len(str(x).split()))
        except:
            print("Avertissement: Analyse de sentiment indisponible.")
    
    # Sauvegarde des données avec nouvelles caractéristiques
    os.makedirs(FEATURES_DATA_DIR, exist_ok=True)
    output_path = os.path.join(FEATURES_DATA_DIR, 'featured_imdb_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Données IMDb avec nouvelles caractéristiques sauvegardées dans {output_path}")
    
    return df

def engineer_twitter_features(input_file):
    """
    Crée des caractéristiques avancées à partir des données Twitter nettoyées.
    
    Args:
        input_file (str): Nom du fichier d'entrée dans le répertoire processed
        
    Returns:
        pandas.DataFrame: Données avec nouvelles caractéristiques
    """
    # Vérification de l'existence du fichier
    file_path = os.path.join(PROCESSED_DATA_DIR, input_file)
    if not os.path.exists(file_path):
        print(f"Fichier {file_path} introuvable.")
        return None
    
    # Chargement des données
    df = pd.read_csv(file_path)
    print(f"Feature engineering pour données Twitter: {df.shape[0]} lignes chargées.")
    
    # Création de nouvelles caractéristiques
    
    # 1. Caractéristiques temporelles
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['hour_of_day'] = df['created_at'].dt.hour
        df['day_of_week'] = df['created_at'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 2. Analyse de sentiment
    if 'clean_text' in df.columns:
        try:
            sia = SentimentIntensityAnalyzer()
            df['sentiment_score'] = df['clean_text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
            df['sentiment_positive'] = df['sentiment_score'].apply(lambda x: 1 if x > 0.05 else 0)
            df['sentiment_negative'] = df['sentiment_score'].apply(lambda x: 1 if x < -0.05 else 0)
            df['sentiment_neutral'] = df['sentiment_score'].apply(lambda x: 1 if -0.05 <= x <= 0.05 else 0)
        except:
            print("Avertissement: Analyse de sentiment indisponible.")
    
    # 3. Caractéristiques du texte
    if 'clean_text' in df.columns:
        df['word_count'] = df['clean_text'].apply(lambda x: len(str(x).split()))
        df['has_question'] = df['clean_text'].apply(lambda x: 1 if '?' in str(x) else 0)
        df['has_exclamation'] = df['clean_text'].apply(lambda x: 1 if '!' in str(x) else 0)
        df['capital_letter_ratio'] = df['clean_text'].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1))
    
    # 4. Popularité relative
    if 'retweets' in df.columns and 'favorites' in df.columns:
        df['engagement'] = df['retweets'] + df['favorites']
        df['retweet_to_favorite_ratio'] = df['retweets'] / (df['favorites'] + 1)  # Éviter division par zéro
    
    # 5. TF-IDF pour les textes (version simplifiée)
    if 'clean_text' in df.columns and df.shape[0] > 10:  # Au moins 10 tweets pour avoir des résultats intéressants
        try:
            tfidf = TfidfVectorizer(max_features=10, stop_words='english')
            tfidf_matrix = tfidf.fit_transform(df['clean_text'].fillna(''))
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{i}' for i in range(10)])
            df = pd.concat([df, tfidf_df], axis=1)
        except:
            print("Avertissement: Vectorisation TF-IDF indisponible.")
    
    # Sauvegarde des données avec nouvelles caractéristiques
    os.makedirs(FEATURES_DATA_DIR, exist_ok=True)
    output_file = 'featured_' + input_file
    output_path = os.path.join(FEATURES_DATA_DIR, output_file)
    df.to_csv(output_path, index=False)
    print(f"Données Twitter avec nouvelles caractéristiques sauvegardées dans {output_path}")
    
    return df

def engineer_custom_features(input_file, config=None):
    """
    Crée des caractéristiques personnalisées pour des données génériques.
    
    Args:
        input_file (str): Nom du fichier d'entrée dans le répertoire processed
        config (dict): Configuration des transformations à appliquer
        
    Returns:
        pandas.DataFrame: Données avec nouvelles caractéristiques
    """
    # Configuration par défaut
    if config is None:
        config = {
            'scale_columns': [],
            'onehot_columns': [],
            'bin_columns': [],
            'interactions': []
        }
    
    # Vérification de l'existence du fichier
    file_path = os.path.join(PROCESSED_DATA_DIR, input_file)
    if not os.path.exists(file_path):
        print(f"Fichier {file_path} introuvable.")
        return None
    
    # Chargement des données
    df = pd.read_csv(file_path)
    print(f"Feature engineering personnalisé: {df.shape[0]} lignes chargées.")
    
    # 1. Normalisation/standardisation
    if config['scale_columns']:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[config['scale_columns']])
        scaled_df = pd.DataFrame(scaled_features, columns=[f"{col}_scaled" for col in config['scale_columns']])
        df = pd.concat([df, scaled_df], axis=1)
    
    # 2. One-hot encoding
    if config['onehot_columns']:
        for col in config['onehot_columns']:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
    
    # 3. Binning (discrétisation)
    if config['bin_columns']:
        for col_info in config['bin_columns']:
            col = col_info['column']
            bins = col_info.get('bins', 5)
            
            if col in df.columns:
                df[f"{col}_binned"] = pd.cut(df[col], bins=bins, labels=False)
    
    # 4. Interactions (produits croisés)
    if config['interactions']:
        for interaction in config['interactions']:
            col1, col2 = interaction
            if col1 in df.columns and col2 in df.columns:
                df[f"{col1}_{col2}_interaction"] = df[col1] * df[col2]
    
    # Sauvegarde des données avec nouvelles caractéristiques
    os.makedirs(FEATURES_DATA_DIR, exist_ok=True)
    output_file = 'featured_' + input_file
    output_path = os.path.join(FEATURES_DATA_DIR, output_file)
    df.to_csv(output_path, index=False)
    print(f"Données avec caractéristiques personnalisées sauvegardées dans {output_path}")
    
    return df

if __name__ == "__main__":
    # Création du dossier des caractéristiques s'il n'existe pas
    os.makedirs(FEATURES_DATA_DIR, exist_ok=True)
    
    # Feature engineering pour les données IMDb
    engineer_imdb_features()
    
    # Feature engineering pour les données Twitter
    twitter_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.startswith('clean_twitter_')]
    for file in twitter_files:
        engineer_twitter_features(file)
    
    # Exemple de feature engineering personnalisé
    # config = {
    #     'scale_columns': ['numeric_col1', 'numeric_col2'],
    #     'onehot_columns': ['category_col'],
    #     'bin_columns': [{'column': 'numeric_col3', 'bins': 4}],
    #     'interactions': [('numeric_col1', 'numeric_col2')]
    # }
    # engineer_custom_features('clean_your_data.csv', config)
    
    print("Feature engineering terminé.")
