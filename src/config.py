"""
Configuración global para el proyecto de detección de mensajes de odio en YouTube.
Este archivo contiene constantes, rutas y parámetros utilizados en todo el proyecto.
"""

import os
from pathlib import Path

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
INTERIM_DATA_DIR = os.path.join(DATA_DIR, "interim")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Asegurar que las carpetas existan
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Rutas de datos
DATASET_PATH = os.path.join(RAW_DATA_DIR, "youtube_hate_speech_dataset.csv")
PROCESSED_DATASET_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_youtube_data.csv")

# Etiquetas de clasificación
HATE_LABELS = [
    'IsToxic', 'IsAbusive', 'IsThreat', 'IsProvocative', 
    'IsObscene', 'IsHatespeech', 'IsRacist', 'IsNationalist', 
    'IsSexist', 'IsHomophobic', 'IsReligiousHate', 'IsRadicalism'
]

# Categorías principales (basado en hallazgos de la fase 1)
PRIMARY_LABELS = ['IsToxic', 'IsAbusive', 'IsHatespeech', 'IsRacist']

# Parámetros de modelado
MODEL_PARAMS = {
    'bert': {
        'model_name': 'bert-base-multilingual-cased',
        'max_length': 256,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'epochs': 5,
        'weight_decay': 0.01,
    },
    'roberta': {
        'model_name': 'roberta-base',
        'max_length': 256,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'epochs': 5,
        'weight_decay': 0.01,
    },
    'xlm-roberta': {
        'model_name': 'xlm-roberta-base',
        'max_length': 256,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'epochs': 5,
        'weight_decay': 0.01,
    },
    'distilbert': {
        'model_name': 'distilbert-base-multilingual-cased',
        'max_length': 256,
        'batch_size': 32,  # Mayor tamaño de batch para modelo más pequeño
        'learning_rate': 5e-5,
        'epochs': 5,
        'weight_decay': 0.01,
    }
}

# Parámetros para validación cruzada
CV_PARAMS = {
    'n_splits': 5,
    'random_state': 42,
    'shuffle': True
}

# Parámetros para manejo de desbalance de clases
CLASS_IMBALANCE_PARAMS = {
    'use_class_weights': True,
    'focal_loss_gamma': 2.0,  # Para Focal Loss si se implementa
}

# Parámetros para optimización de hiperparámetros
HP_TUNING_PARAMS = {
    'n_trials': 20,
    'timeout': 3600,  # 1 hora
    'study_name': 'hate_speech_optimization'
}

# Configuraciones para MLflow
MLFLOW_TRACKING_URI = "mlruns"
MLFLOW_EXPERIMENT_NAME = "youtube_hate_speech_detection"

# Métricas para seguimiento
TRACKING_METRICS = [
    'accuracy', 'precision', 'recall', 'f1',
    'auc', 'average_precision', 'roc_auc'
]

# Parámetros de preprocesamiento
PREPROCESSING_PARAMS = {
    'remove_urls': True,
    'remove_mentions': True,
    'normalize_emojis': True,
    'language_detection': True,
    'lemmatization': True,
    'stopwords_removal': True
}