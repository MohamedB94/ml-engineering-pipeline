"""
Module pour l'entraînement des modèles de machine learning
et l'évaluation de leurs performances.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# Chemins des répertoires
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FEATURES_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'features')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def load_training_data(input_file, target_column, features=None, test_size=0.2, random_state=42):
    """
    Charge les données d'entraînement et les prépare pour le modèle.
    
    Args:
        input_file (str): Nom du fichier d'entrée dans le répertoire features
        target_column (str): Nom de la colonne cible
        features (list): Liste des colonnes à utiliser comme caractéristiques
        test_size (float): Proportion des données pour le test
        random_state (int): Graine aléatoire pour la reproductibilité
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    # Vérification de l'existence du fichier
    file_path = os.path.join(FEATURES_DATA_DIR, input_file)
    if not os.path.exists(file_path):
        print(f"Fichier {file_path} introuvable.")
        return None
    
    # Chargement des données
    df = pd.read_csv(file_path)
    print(f"Chargement des données d'entraînement: {df.shape[0]} lignes.")
    
    # Vérification que la colonne cible existe
    if target_column not in df.columns:
        print(f"Colonne cible '{target_column}' introuvable dans les données.")
        return None
    
    # Sélection des caractéristiques
    if features is None:
        # Utilisation de toutes les colonnes sauf la cible
        features = [col for col in df.columns if col != target_column]
    
    # Filtrage des colonnes non-numériques
    numeric_features = []
    for feature in features:
        if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
            numeric_features.append(feature)
        elif feature in df.columns:
            print(f"Avertissement: La colonne '{feature}' n'est pas numérique et sera ignorée.")
    
    # Données pour l'entraînement
    X = df[numeric_features].copy()
    y = df[target_column].copy()
    
    # Gestion des valeurs manquantes
    X = X.fillna(X.mean())
    
    # Division en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Données divisées: {X_train.shape[0]} exemples d'entraînement, {X_test.shape[0]} exemples de test.")
    print(f"Caractéristiques utilisées: {len(numeric_features)}")
    
    return X_train, X_test, y_train, y_test, numeric_features

def train_regression_model(X_train, y_train, X_test, y_test, features, model_type='random_forest'):
    """
    Entraîne un modèle de régression et évalue ses performances.
    
    Args:
        X_train (pd.DataFrame): Caractéristiques d'entraînement
        y_train (pd.Series): Cible d'entraînement
        X_test (pd.DataFrame): Caractéristiques de test
        y_test (pd.Series): Cible de test
        features (list): Noms des caractéristiques
        model_type (str): Type de modèle ('linear', 'random_forest', 'gradient_boosting', 'svr')
        
    Returns:
        tuple: (modèle entraîné, performances)
    """
    print(f"Entraînement d'un modèle de régression de type '{model_type}'...")
    
    # Sélection du modèle
    if model_type == 'linear':
        model = LinearRegression()
        param_grid = {'fit_intercept': [True, False]}
    elif model_type == 'random_forest':
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    elif model_type == 'svr':
        model = SVR()
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    else:
        print(f"Type de modèle '{model_type}' non reconnu. Utilisation de RandomForest par défaut.")
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    
    # Pipeline avec standardisation et sélection de caractéristiques
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_regression, k=min(15, len(features)))),
        ('model', model)
    ])
    
    # Recherche des meilleurs hyperparamètres
    param_grid = {'model__' + key: value for key, value in param_grid.items()}
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Meilleurs hyperparamètres: {grid_search.best_params_}")
    
    # Évaluation du modèle
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Métriques de performance
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    performance = {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'best_params': grid_search.best_params_
    }
    
    print(f"Performance du modèle:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    # Visualisation des prédictions vs valeurs réelles
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Prédictions')
    plt.title('Prédictions vs Valeurs réelles')
    
    # Sauvegarde du graphique
    os.makedirs(MODELS_DIR, exist_ok=True)
    plot_path = os.path.join(MODELS_DIR, f'{model_type}_regression_plot.png')
    plt.savefig(plot_path)
    print(f"Graphique sauvegardé dans {plot_path}")
    
    # Importance des caractéristiques (si disponible)
    if hasattr(best_model['model'], 'feature_importances_'):
        # Récupération des indices des caractéristiques sélectionnées
        selected_indices = best_model['feature_selection'].get_support(indices=True)
        selected_features = [features[i] for i in selected_indices]
        
        # Récupération de l'importance des caractéristiques
        importances = best_model['model'].feature_importances_
        
        # Tri des caractéristiques par importance
        indices = np.argsort(importances)[::-1]
        
        # Visualisation
        plt.figure(figsize=(10, 6))
        plt.title('Importance des caractéristiques')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [selected_features[i] for i in indices], rotation=90)
        plt.tight_layout()
        
        # Sauvegarde du graphique
        feature_plot_path = os.path.join(MODELS_DIR, f'{model_type}_feature_importance.png')
        plt.savefig(feature_plot_path)
        print(f"Graphique d'importance des caractéristiques sauvegardé dans {feature_plot_path}")
    
    return best_model, performance

def train_classification_model(X_train, y_train, X_test, y_test, features, model_type='random_forest'):
    """
    Entraîne un modèle de classification et évalue ses performances.
    
    Args:
        X_train (pd.DataFrame): Caractéristiques d'entraînement
        y_train (pd.Series): Cible d'entraînement
        X_test (pd.DataFrame): Caractéristiques de test
        y_test (pd.Series): Cible de test
        features (list): Noms des caractéristiques
        model_type (str): Type de modèle ('logistic', 'random_forest', 'gradient_boosting', 'svc')
        
    Returns:
        tuple: (modèle entraîné, performances)
    """
    print(f"Entraînement d'un modèle de classification de type '{model_type}'...")
    
    # Vérification que la cible est catégorielle
    if pd.api.types.is_numeric_dtype(y_train) and len(np.unique(y_train)) > 10:
        print("Avertissement: La cible semble être continue. Considérez utiliser un modèle de régression.")
    
    # Sélection du modèle
    if model_type == 'logistic':
        model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    elif model_type == 'svc':
        model = SVC(random_state=42, probability=True)
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    else:
        print(f"Type de modèle '{model_type}' non reconnu. Utilisation de RandomForest par défaut.")
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    
    # Pipeline avec standardisation et sélection de caractéristiques
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(mutual_info_regression, k=min(15, len(features)))),
        ('model', model)
    ])
    
    # Recherche des meilleurs hyperparamètres
    param_grid = {'model__' + key: value for key, value in param_grid.items()}
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Meilleurs hyperparamètres: {grid_search.best_params_}")
    
    # Évaluation du modèle
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Métriques de performance
    accuracy = accuracy_score(y_test, y_pred)
    
    # Vérification si classification binaire ou multiclasse
    if len(np.unique(y_train)) == 2:
        # Classification binaire
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
    else:
        # Classification multiclasse
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
    
    performance = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'best_params': grid_search.best_params_
    }
    
    print(f"Performance du modèle:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # Matrice de confusion
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Prédictions')
    plt.ylabel('Valeurs réelles')
    plt.title('Matrice de confusion')
    
    # Sauvegarde du graphique
    os.makedirs(MODELS_DIR, exist_ok=True)
    plot_path = os.path.join(MODELS_DIR, f'{model_type}_confusion_matrix.png')
    plt.savefig(plot_path)
    print(f"Matrice de confusion sauvegardée dans {plot_path}")
    
    # Rapport de classification détaillé
    report = classification_report(y_test, y_pred)
    print("Rapport de classification détaillé:")
    print(report)
    
    # Importance des caractéristiques (si disponible)
    if hasattr(best_model['model'], 'feature_importances_'):
        # Récupération des indices des caractéristiques sélectionnées
        selected_indices = best_model['feature_selection'].get_support(indices=True)
        selected_features = [features[i] for i in selected_indices]
        
        # Récupération de l'importance des caractéristiques
        importances = best_model['model'].feature_importances_
        
        # Tri des caractéristiques par importance
        indices = np.argsort(importances)[::-1]
        
        # Visualisation
        plt.figure(figsize=(10, 6))
        plt.title('Importance des caractéristiques')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [selected_features[i] for i in indices], rotation=90)
        plt.tight_layout()
        
        # Sauvegarde du graphique
        feature_plot_path = os.path.join(MODELS_DIR, f'{model_type}_feature_importance.png')
        plt.savefig(feature_plot_path)
        print(f"Graphique d'importance des caractéristiques sauvegardé dans {feature_plot_path}")
    
    return best_model, performance

def save_model(model, model_name, metadata=None):
    """
    Sauvegarde un modèle entraîné et ses métadonnées.
    
    Args:
        model: Modèle entraîné
        model_name (str): Nom du modèle
        metadata (dict): Métadonnées supplémentaires
        
    Returns:
        str: Chemin du modèle sauvegardé
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Génération d'un nom de fichier avec horodatage
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = f"{model_name}_{timestamp}.joblib"
    model_path = os.path.join(MODELS_DIR, model_file)
    
    # Sauvegarde du modèle avec joblib
    joblib.dump(model, model_path)
    print(f"Modèle sauvegardé dans {model_path}")
    
    # Sauvegarde des métadonnées
    if metadata:
        metadata_file = f"{model_name}_{timestamp}_metadata.json"
        metadata_path = os.path.join(MODELS_DIR, metadata_file)
        
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=4)
        
        print(f"Métadonnées sauvegardées dans {metadata_path}")
    
    return model_path

if __name__ == "__main__":
    # Exemple d'utilisation pour un modèle de régression sur des données IMDb
    print("\n=== Entraînement d'un modèle de régression sur les données IMDb ===\n")
    
    # Charger les données
    X_train, X_test, y_train, y_test, features = load_training_data(
        'featured_imdb_data.csv',
        target_column='imdbRating',
        features=None,  # Utilisera toutes les colonnes numériques
        test_size=0.2,
        random_state=42
    )
    
    if X_train is not None:
        # Entraîner le modèle
        model, performance = train_regression_model(
            X_train, y_train, X_test, y_test, features,
            model_type='random_forest'
        )
        
        # Sauvegarder le modèle
        model_path = save_model(model, 'imdb_rating_predictor', {
            'performance': performance,
            'features': features,
            'target': 'imdbRating',
            'model_type': 'random_forest',
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'description': 'Modèle de prédiction de la note IMDb basé sur les caractéristiques du film'
        })
    
    # Exemple pour un modèle de classification sur des données Twitter
    # print("\n=== Entraînement d'un modèle de classification sur les données Twitter ===\n")
    
    # twitter_files = [f for f in os.listdir(FEATURES_DATA_DIR) if f.startswith('featured_clean_twitter_')]
    # if twitter_files:
    #     # Charger les données Twitter
    #     X_train, X_test, y_train, y_test, features = load_training_data(
    #         twitter_files[0],
    #         target_column='sentiment_positive',  # Colonne générée dans feature_engineering.py
    #         features=None,
    #         test_size=0.2,
    #         random_state=42
    #     )
        
    #     if X_train is not None:
    #         # Entraîner le modèle
    #         model, performance = train_classification_model(
    #             X_train, y_train, X_test, y_test, features,
    #             model_type='gradient_boosting'
    #         )
            
    #         # Sauvegarder le modèle
    #         model_path = save_model(model, 'twitter_sentiment_classifier', {
    #             'performance': performance,
    #             'features': features,
    #             'target': 'sentiment_positive',
    #             'model_type': 'gradient_boosting',
    #             'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    #             'description': 'Modèle de classification de sentiment pour les tweets'
    #         })
    
    print("\nEntraînement des modèles terminé.")
