"""
Configuración centralizada para el proyecto de detección de mensajes de odio en YouTube.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Directorios del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
MLFLOW_DIR = BASE_DIR / "mlflow"
LOGS_DIR = BASE_DIR / "logs"

# Configuración de la API de YouTube
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Configuración del modelo
MODEL_CONFIG = {
    "model_name": "bert-base-multilingual-cased",
    "max_length": 128,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "weight_decay": 0.01,
    "seed": 42,
}

# Categorías de mensajes de odio
HATE_SPEECH_CATEGORIES = [
    "IsToxic", "IsAbusive", "IsThreat", "IsProvocative", 
    "IsObscene", "IsHatespeech", "IsRacist", "IsNationalist", 
    "IsSexist", "IsHomophobic", "IsReligiousHate", "IsRadicalism"
]

# Configuración de la base de datos
DB_URL = os.getenv("DATABASE_URL", "sqlite:///hate_speech.db")

# Crear directorios si no existen
for directory in [RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, 
                 MODELS_DIR, MLFLOW_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)