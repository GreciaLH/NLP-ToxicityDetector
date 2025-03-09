#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para preprocesamiento y limpieza de datos del dataset de comentarios de YouTube.
Este script realiza las siguientes tareas:
1. Carga el dataset original
2. Limpia y preprocesa el texto
3. Maneja valores faltantes
4. Guarda el dataset procesado
"""

import os
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import logging
import argparse
from pathlib import Path

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_nltk_resources():
    """Descarga recursos necesarios de NLTK"""
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
            logger.info(f"Recurso {resource} ya está descargado")
        except LookupError:
            logger.info(f"Descargando recurso {resource}")
            nltk.download(resource, quiet=False)
    
    # Asegurar que se descargue el tokenizador de oraciones
    try:
        nltk.data.find('tokenizers/punkt/english.pickle')
    except LookupError:
        nltk.download('punkt', quiet=False)

def load_data(input_filepath):
    """
    Carga el dataset desde un archivo CSV
    
    Args:
        input_filepath: Ruta al archivo CSV
        
    Returns:
        DataFrame con los datos cargados
    """
    logger.info(f"Cargando datos desde {input_filepath}")
    return pd.read_csv(input_filepath)

def clean_text(text):
    """
    Limpia y preprocesa el texto
    
    Args:
        text: Texto a limpiar
        
    Returns:
        Texto limpio
    """
    if not isinstance(text, str):
        return ""
    
    # Convertir a minúsculas
    text = text.lower()
    
    # Eliminar URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Eliminar etiquetas HTML
    text = re.sub(r'<.*?>', '', text)
    
    # Eliminar emojis y caracteres especiales
    text = re.sub(r'[^\w\s.,!?]', '', text)
    
    # Eliminar números
    text = re.sub(r'\d+', '', text)
    
    # Eliminar espacios adicionales
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

from nltk.tokenize import RegexpTokenizer

def lemmatize_text(text, lemmatizer, stop_words):
    """
    Tokeniza y lematiza el texto
    
    Args:
        text: Texto a lematizar
        lemmatizer: Instancia de WordNetLemmatizer
        stop_words: Conjunto de stopwords
        
    Returns:
        Texto lematizado
    """
    if not isinstance(text, str) or not text:
        return ""
    
    # Usar RegexpTokenizer en lugar de word_tokenize
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    
    # Eliminar stopwords y lematizar
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)

def preprocess_data(df, lemmatize=True):
    """
    Preprocesa todo el dataset
    
    Args:
        df: DataFrame con los datos
        lemmatize: Si se debe lematizar el texto
        
    Returns:
        DataFrame con los datos preprocesados
    """
    logger.info("Iniciando preprocesamiento de datos")
    
    # Crear copia para no modificar el original
    processed_df = df.copy()
    
    # Limpiar texto
    logger.info("Limpiando texto")
    processed_df['cleaned_text'] = processed_df['Text'].apply(clean_text)
    
    if lemmatize:
        # Descargar recursos NLTK si es necesario
        download_nltk_resources()
        
        # Inicializar lematizador y stopwords
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        # Lematizar texto
        logger.info("Lematizando texto")
        processed_df['lemmatized_text'] = processed_df['cleaned_text'].apply(
            lambda x: lemmatize_text(x, lemmatizer, stop_words)
        )
    
    # Verificar y manejar valores faltantes
    logger.info("Verificando valores faltantes")
    missing_values = processed_df.isnull().sum()
    if missing_values.sum() > 0:
        logger.warning(f"Se encontraron valores faltantes:\n{missing_values[missing_values > 0]}")
        
        # Manejar valores faltantes en texto
        if processed_df['cleaned_text'].isnull().sum() > 0:
            processed_df['cleaned_text'] = processed_df['cleaned_text'].fillna('')
        
        if lemmatize and processed_df['lemmatized_text'].isnull().sum() > 0:
            processed_df['lemmatized_text'] = processed_df['lemmatized_text'].fillna('')
    
    # Calcular longitud del texto
    processed_df['text_length'] = processed_df['cleaned_text'].apply(len)
    processed_df['word_count'] = processed_df['cleaned_text'].apply(lambda x: len(x.split()))
    
    logger.info("Preprocesamiento completado")
    return processed_df

def save_processed_data(df, output_filepath):
    """
    Guarda el dataset procesado en un archivo CSV
    
    Args:
        df: DataFrame con los datos procesados
        output_filepath: Ruta donde guardar el archivo
    """
    # Crear directorio si no existe
    output_dir = os.path.dirname(output_filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    logger.info(f"Guardando datos procesados en {output_filepath}")
    df.to_csv(output_filepath, index=False)
    logger.info("Datos guardados exitosamente")

def main(input_filepath, output_filepath, lemmatize=True):
    """Función principal para ejecutar el preprocesamiento"""
    # Cargar datos
    df = load_data(input_filepath)
    
    # Preprocesar datos
    processed_df = preprocess_data(df, lemmatize)
    
    # Guardar datos procesados
    save_processed_data(processed_df, output_filepath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocesamiento de datos para el detector de mensajes de odio')
    parser.add_argument('--input', type=str, default='../../data/raw/youtube_hate_speech_dataset.csv',
                        help='Ruta al archivo CSV de entrada')
    parser.add_argument('--output', type=str, default='data/processed/youtube_comments_processed.csv',
                        help='Ruta para guardar el archivo CSV procesado')
    parser.add_argument('--no-lemmatize', dest='lemmatize', action='store_false',
                        help='Desactivar lematización (reduce tiempo de procesamiento)')
    parser.set_defaults(lemmatize=True)
    
    args = parser.parse_args()
    
    main(args.input, args.output, args.lemmatize)