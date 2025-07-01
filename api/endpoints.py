"""
Module contenant les routes et endpoints supplémentaires pour l'API.
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import pandas as pd
import os
import sys
import json
import tempfile
from datetime import datetime

# Ajout du répertoire parent au path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Import des modules locaux
from src.data_collection import collect_imdb_data, load_csv_data
from src.data_cleaning import clean_csv_data

# Création du router
router = APIRouter(prefix="/api/v1", tags=["Data Processing"])

# Modèles Pydantic
class DataProcessingResult(BaseModel):
    """Modèle pour les résultats de traitement de données."""
    status: str
    message: str
    rows_processed: int
    columns_processed: int
    processing_time: float
    preview: List[Dict[str, Any]]

class DataCleaningConfig(BaseModel):
    """Configuration pour le nettoyage de données."""
    date_columns: Optional[List[str]] = None
    numeric_columns: Optional[List[str]] = None
    categorical_columns: Optional[List[str]] = None

# Fonctions d'aide
def process_uploaded_csv(file_path: str, config: DataCleaningConfig = None):
    """
    Traite un fichier CSV téléchargé.
    
    Args:
        file_path (str): Chemin du fichier CSV
        config (DataCleaningConfig): Configuration de nettoyage
    
    Returns:
        pd.DataFrame: Données traitées
    """
    # Chargement des données
    df = load_csv_data(file_path)
    
    # Nettoyage selon la configuration
    if config:
        df = clean_csv_data(
            file_path,
            date_columns=config.date_columns,
            numeric_columns=config.numeric_columns,
            categorical_columns=config.categorical_columns
        )
    
    return df

# Routes
@router.post("/upload-csv", response_model=DataProcessingResult)
async def upload_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    clean_data: bool = Form(False),
    config: Optional[str] = Form(None)
):
    """
    Télécharge et traite un fichier CSV.
    
    Args:
        file (UploadFile): Fichier CSV à télécharger
        clean_data (bool): Effectuer le nettoyage des données
        config (str): Configuration JSON pour le nettoyage
    
    Returns:
        DataProcessingResult: Résultat du traitement
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Le fichier doit être au format CSV")
    
    start_time = datetime.now()
    
    # Sauvegarde temporaire du fichier
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    try:
        # Écriture du contenu du fichier
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        
        # Traitement du fichier
        cleaning_config = None
        if clean_data and config:
            try:
                config_dict = json.loads(config)
                cleaning_config = DataCleaningConfig(**config_dict)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Format de configuration JSON invalide")
        
        # Chargement et traitement des données
        df = process_uploaded_csv(temp_file.name, cleaning_config)
        
        # Génération du résultat
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Aperçu des données (max 5 lignes)
        preview = df.head(5).to_dict('records')
        
        # Ajout d'une tâche en arrière-plan pour sauvegarder les données
        # Cela permettrait de continuer le traitement après avoir retourné la réponse
        # background_tasks.add_task(save_processed_data, df, file.filename)
        
        return {
            "status": "success",
            "message": f"Fichier {file.filename} traité avec succès",
            "rows_processed": df.shape[0],
            "columns_processed": df.shape[1],
            "processing_time": processing_time,
            "preview": preview
        }
    
    finally:
        # Nettoyage du fichier temporaire
        os.unlink(temp_file.name)

@router.get("/data-sources", response_model=List[Dict[str, Any]])
async def get_data_sources():
    """
    Liste les sources de données disponibles.
    
    Returns:
        List[Dict[str, Any]]: Liste des sources de données
    """
    # Dans un système réel, cette information pourrait provenir d'une base de données
    return [
        {
            "id": "imdb",
            "name": "IMDb",
            "description": "Données de films de la base IMDb",
            "type": "api",
            "status": "available"
        },
        {
            "id": "twitter",
            "name": "Twitter",
            "description": "Données de tweets via l'API Twitter",
            "type": "api",
            "status": "requires_auth"
        },
        {
            "id": "csv",
            "name": "CSV Upload",
            "description": "Upload de fichiers CSV personnalisés",
            "type": "upload",
            "status": "available"
        }
    ]

@router.post("/collect-imdb-data", response_model=DataProcessingResult)
async def api_collect_imdb_data(movie_ids: List[str]):
    """
    Collecte des données depuis IMDb pour les identifiants spécifiés.
    
    Args:
        movie_ids (List[str]): Liste d'identifiants IMDb de films
    
    Returns:
        DataProcessingResult: Résultat de la collecte
    """
    if not movie_ids:
        raise HTTPException(status_code=400, detail="Aucun identifiant de film fourni")
    
    start_time = datetime.now()
    
    # Collecte des données
    df = collect_imdb_data(movie_ids)
    
    if df.empty:
        raise HTTPException(status_code=500, detail="Échec de la collecte des données IMDb")
    
    # Génération du résultat
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # Aperçu des données (max 5 lignes)
    preview = df.head(5).to_dict('records')
    
    return {
        "status": "success",
        "message": f"Données collectées pour {len(movie_ids)} films",
        "rows_processed": df.shape[0],
        "columns_processed": df.shape[1],
        "processing_time": processing_time,
        "preview": preview
    }

# Fonction pour enregistrer le router dans l'application principale
def include_router(app):
    """
    Ajoute le router à l'application FastAPI principale.
    
    Args:
        app: Application FastAPI
    """
    app.include_router(router)
