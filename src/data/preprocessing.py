"""
Módulo para preprocesamiento de texto para los modelos de detección de mensajes de odio.
Incluye funciones para limpieza, normalización y transformación de texto.
"""

import re
import pandas as pd
import numpy as np
import spacy
import nltk
from nltk.corpus import stopwords
import logging
from typing import List, Dict, Union, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Asegurar que los recursos de NLTK estén descargados
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Cargar modelos de spaCy
try:
    nlp_es = spacy.load('es_core_news_sm')
    nlp_en = spacy.load('en_core_web_sm')
    SPACY_LOADED = True
except OSError:
    logger.warning("Modelos de spaCy no encontrados. Usando procesamiento básico.")
    SPACY_LOADED = False


def detect_language(text: str) -> str:
    """
    Detecta el idioma del texto (simplificado para español e inglés).
    
    Args:
        text: Texto a analizar
        
    Returns:
        Código de idioma detectado ('es', 'en' o 'unknown')
    """
    if not text or pd.isna(text):
        return "unknown"
    
    # Enfoque simplificado usando stopwords
    es_stops = set(stopwords.words('spanish'))
    en_stops = set(stopwords.words('english'))
    
    words = set(text.lower().split())
    es_count = len(words.intersection(es_stops))
    en_count = len(words.intersection(en_stops))
    
    if es_count > en_count:
        return 'es'
    elif en_count > 0:
        return 'en'
    else:
        return 'unknown'


def clean_text(text: str, params: Dict[str, bool] = None) -> str:
    """
    Limpia y normaliza el texto.
    
    Args:
        text: Texto a limpiar
        params: Parámetros de configuración para la limpieza
        
    Returns:
        Texto limpio
    """
    if pd.isna(text) or not text:
        return ""
    
    if params is None:
        params = {
            'remove_urls': True,
            'remove_mentions': True,
            'normalize_emojis': True
        }
    
    # Convertir a minúsculas
    text = text.lower()
    
    # Eliminar URLs
    if params.get('remove_urls', True):
        text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    
    # Eliminar menciones (@usuario)
    if params.get('remove_mentions', True):
        text = re.sub(r'@\w+', ' MENTION ', text)
    
    # Normalizar emojis
    if params.get('normalize_emojis', True):
        text = re.sub(r'[^\w\s,.]', ' EMOJI ', text)
    
    # Eliminar caracteres no alfanuméricos y espacios múltiples
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def process_text(text: str, lang: str = 'unknown', 
                params: Dict[str, bool] = None) -> str:
    """
    Procesa el texto completo con limpieza, lematización y filtrado de stopwords.
    
    Args:
        text: Texto a procesar
        lang: Código de idioma ('es', 'en', 'unknown')
        params: Parámetros de configuración para el procesamiento
        
    Returns:
        Texto procesado
    """
    if pd.isna(text) or not text:
        return ""
    
    if params is None:
        params = {
            'lemmatization': True,
            'stopwords_removal': True
        }
    
    # Limpiar texto primero
    text = clean_text(text, params)
    
    # Si no tenemos spaCy o no queremos lematización, devolver el texto limpio
    if not SPACY_LOADED or not params.get('lemmatization', True):
        if params.get('stopwords_removal', True):
            # Eliminar stopwords sin spaCy
            if lang == 'es':
                stop_words = set(stopwords.words('spanish'))
            elif lang == 'en':
                stop_words = set(stopwords.words('english'))
            else:
                stop_words = set(stopwords.words('spanish') + stopwords.words('english'))
            
            tokens = text.split()
            tokens = [token for token in tokens if token not in stop_words]
            return " ".join(tokens)
        return text
    
    # Procesamiento con spaCy según idioma detectado
    if lang == 'es':
        doc = nlp_es(text)
    elif lang == 'en':
        doc = nlp_en(text)
    else:
        # Para lenguaje desconocido, usar el modelo español por defecto
        doc = nlp_es(text)
    
    # Lematización y filtrado de stopwords
    if params.get('stopwords_removal', True):
        tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    else:
        tokens = [token.lemma_ for token in doc if token.is_alpha]
    
    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame, 
                        text_column: str = 'Text',
                        language_column: str = 'language',
                        processed_column: str = 'processed_text',
                        params: Dict[str, bool] = None) -> pd.DataFrame:
    """
    Preprocesa un DataFrame completo con detección de idioma y procesamiento de texto.
    
    Args:
        df: DataFrame a procesar
        text_column: Nombre de la columna de texto
        language_column: Nombre de la columna donde guardar el idioma detectado
        processed_column: Nombre de la columna donde guardar el texto procesado
        params: Parámetros de configuración para el procesamiento
        
    Returns:
        DataFrame con el texto procesado
    """
    logger.info(f"Procesando DataFrame de {len(df)} filas")
    
    # Crear copias para evitar modificar el original
    result_df = df.copy()
    
    # Detectar idiomas
    if params and params.get('language_detection', True):
        logger.info("Detectando idiomas...")
        result_df[language_column] = result_df[text_column].apply(detect_language)
        
        # Estadísticas de idiomas
        lang_counts = result_df[language_column].value_counts()
        logger.info(f"Distribución de idiomas detectados: {lang_counts.to_dict()}")
    elif language_column not in result_df.columns:
        # Si no hay detección de idioma y no existe la columna, asignar 'unknown'
        result_df[language_column] = 'unknown'
    
    # Procesar textos
    logger.info("Procesando textos...")
    result_df[processed_column] = result_df.apply(
        lambda row: process_text(
            row[text_column], 
            row[language_column],
            params
        ), 
        axis=1
    )
    
    logger.info("Procesamiento completado")
    return result_df


def preprocess_for_transformer(texts: List[str], 
                            tokenizer,
                            max_length: int = 256,
                            truncation: bool = True,
                            padding: str = 'max_length') -> Dict:
    """
    Preprocesa textos para modelos transformer usando un tokenizador.
    
    Args:
        texts: Lista de textos a procesar
        tokenizer: Tokenizador de Hugging Face
        max_length: Longitud máxima de secuencia
        truncation: Si se debe truncar secuencias largas
        padding: Estrategia de padding ('max_length' o 'longest')
        
    Returns:
        Diccionario con tensores de entrada para el modelo
    """
    return tokenizer(
        texts,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        return_tensors='pt'  # PyTorch tensors
    )


if __name__ == "__main__":
    # Código de prueba
    test_text = "Hola @usuario, mira este link https://example.com ¡Es genial! 😊"
    print(f"Original: {test_text}")
    print(f"Limpio: {clean_text(test_text)}")
    print(f"Idioma detectado: {detect_language(test_text)}")
    print(f"Procesado: {process_text(test_text, 'es')}")