#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para dividir los datos en conjuntos de entrenamiento, validación y prueba.
Este script implementa:
1. División estratificada para mantener la distribución de clases
2. Opciones para manejo de desbalance de clases
3. Guardar los conjuntos resultantes
"""

import os
import pandas as pd
import numpy as np
import logging
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_processed_data(input_filepath):
    """
    Carga el dataset procesado
    
    Args:
        input_filepath: Ruta al archivo CSV procesado
        
    Returns:
        DataFrame con los datos procesados
    """
    logger.info(f"Cargando datos procesados desde {input_filepath}")
    return pd.read_csv(input_filepath)

def split_data(df, target_column='IsToxic', test_size=0.2, val_size=0.1, random_state=42):
    """
    Divide los datos en conjuntos de entrenamiento, validación y prueba
    
    Args:
        df: DataFrame con los datos procesados
        target_column: Columna objetivo para estratificación
        test_size: Proporción del conjunto de prueba
        val_size: Proporción del conjunto de validación
        random_state: Semilla para reproducibilidad
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test: Conjuntos divididos
    """
    logger.info("Dividiendo datos en conjuntos de entrenamiento, validación y prueba")
    
    # Preparar características y etiquetas
    if 'lemmatized_text' in df.columns:
        X = df[['lemmatized_text', 'text_length', 'word_count']]
    else:
        X = df[['cleaned_text', 'text_length', 'word_count']]
    
    # Obtener todas las columnas que comienzan con 'Is' para clasificación multi-etiqueta
    label_columns = [col for col in df.columns if col.startswith('Is')]
    y = df[label_columns]
    
    # Primera división: separar conjunto de prueba
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size,
        stratify=y[target_column],
        random_state=random_state
    )
    
    # Segunda división: separar conjuntos de entrenamiento y validación
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        stratify=y_temp[target_column],
        random_state=random_state
    )
    
    logger.info(f"Tamaño del conjunto de entrenamiento: {len(X_train)}")
    logger.info(f"Tamaño del conjunto de validación: {len(X_val)}")
    logger.info(f"Tamaño del conjunto de prueba: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def handle_class_imbalance(X_train, y_train, method='smote', target_column='IsToxic', random_state=42):
    """
    Maneja el desbalance de clases en el conjunto de entrenamiento
    
    Args:
        X_train: Características del conjunto de entrenamiento
        y_train: Etiquetas del conjunto de entrenamiento
        method: Método para manejar el desbalance ('smote', 'undersample', 'smotetomek' o 'none')
        target_column: Columna objetivo para el resampling
        random_state: Semilla para reproducibilidad
        
    Returns:
        X_resampled, y_resampled: Conjunto de entrenamiento rebalanceado
    """
    if method == 'none':
        logger.info("No se aplicará técnica para manejar el desbalance de clases")
        return X_train, y_train
    
    logger.info(f"Aplicando método {method} para manejar el desbalance de clases")
    
    # Preparar datos para resampling
    # Convertir a numpy array para compatibilidad con imblearn
    if 'lemmatized_text' in X_train.columns:
        text_column = 'lemmatized_text'
    else:
        text_column = 'cleaned_text'
    
    # Guardamos los textos originales ya que no podemos aplicar SMOTE directamente a texto
    texts = X_train[text_column].values
    
    # Usamos solo características numéricas para SMOTE
    X_numeric = X_train[['text_length', 'word_count']].values
    y_target = y_train[target_column].values
    
    # Aplicar método de resampling
    if method == 'smote':
        resampler = SMOTE(random_state=random_state)
    elif method == 'undersample':
        resampler = RandomUnderSampler(random_state=random_state)
    elif method == 'smotetomek':
        resampler = SMOTETomek(random_state=random_state)
    else:
        logger.error(f"Método {method} no reconocido")
        return X_train, y_train
    
    # Aplicar resampling solo a las características numéricas
    X_numeric_resampled, y_resampled = resampler.fit_resample(X_numeric, y_target)
    
    # Reconstruir DataFrame con todas las etiquetas
    y_resampled_df = pd.DataFrame(y_resampled, columns=[target_column])
    
    # Para las demás etiquetas, necesitamos reconstruirlas basadas en los índices originales
    if len(y_train.columns) > 1:
        # Obtener índices de las muestras originales que se mantuvieron
        sample_indices = resampler.sample_indices_ if hasattr(resampler, 'sample_indices_') else None
        
        if sample_indices is not None and len(sample_indices) > 0:
            for col in y_train.columns:
                if col != target_column:
                    y_resampled_df[col] = y_train.iloc[sample_indices][col].values
        else:
            # Si no podemos obtener los índices, simplemente repetimos los valores de la primera etiqueta
            for col in y_train.columns:
                if col != target_column:
                    y_resampled_df[col] = y_resampled
    
    # Crear un nuevo DataFrame para X
    if len(X_numeric_resampled) == len(texts):  # Si no se agregaron muestras sintéticas
        X_resampled_df = pd.DataFrame({
            text_column: texts,
            'text_length': X_numeric_resampled[:, 0],
            'word_count': X_numeric_resampled[:, 1]
        })
    else:
        # Si se agregaron muestras sintéticas, necesitamos crear textos "sintéticos"
        # Esto es una simplificación; en la práctica, podríamos usar técnicas más avanzadas
        synthetic_count = len(X_numeric_resampled) - len(texts)
        logger.info(f"Se generaron {synthetic_count} muestras sintéticas")
        
        # Para las muestras sintéticas, usamos textos existentes de la clase minoritaria
        minority_texts = texts[y_target == 1] if y_target.sum() < len(y_target) / 2 else texts[y_target == 0]
        
        if len(minority_texts) > 0:
            synthetic_texts = np.random.choice(minority_texts, synthetic_count, replace=True)
            all_texts = np.concatenate([texts, synthetic_texts])
        else:
            # Si no hay textos de la clase minoritaria, duplicamos textos al azar
            all_texts = np.concatenate([texts, np.random.choice(texts, synthetic_count, replace=True)])
        
        X_resampled_df = pd.DataFrame({
            text_column: all_texts,
            'text_length': X_numeric_resampled[:, 0],
            'word_count': X_numeric_resampled[:, 1]
        })
    
    logger.info(f"Distribución de clases después del resampling: {np.bincount(y_resampled)}")
    
    return X_resampled_df, y_resampled_df

def save_split_data(X_train, X_val, X_test, y_train, y_val, y_test, output_dir):
    """
    Guarda los conjuntos divididos en archivos CSV
    
    Args:
        X_train, X_val, X_test: Conjuntos de características
        y_train, y_val, y_test: Conjuntos de etiquetas
        output_dir: Directorio donde guardar los archivos
    """
    # Crear directorio si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logger.info(f"Guardando conjuntos divididos en {output_dir}")
    
    # Combinar características y etiquetas
    train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    val_df = pd.concat([X_val.reset_index(drop=True), y_val.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    
    # Guardar en archivos CSV
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    logger.info("Conjuntos guardados exitosamente")

def main(input_filepath, output_dir, imbalance_method='smote', target_column='IsToxic', 
         test_size=0.2, val_size=0.1, random_state=42):
    """Función principal para dividir los datos"""
    # Cargar datos procesados
    df = load_processed_data(input_filepath)
    
    # Dividir datos
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, target_column, test_size, val_size, random_state
    )
    
    # Manejar desbalance de clases en el conjunto de entrenamiento
    X_train_resampled, y_train_resampled = handle_class_imbalance(
        X_train, y_train, imbalance_method, target_column, random_state
    )
    
    # Guardar conjuntos divididos
    save_split_data(X_train_resampled, X_val, X_test, y_train_resampled, y_val, y_test, output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='División de datos para el detector de mensajes de odio')
    parser.add_argument('--input', type=str, default='data/processed/youtube_comments_processed.csv',
                        help='Ruta al archivo CSV procesado')
    parser.add_argument('--output-dir', type=str, default='data/processed/split',
                        help='Directorio para guardar los conjuntos divididos')
    parser.add_argument('--imbalance-method', type=str, default='smote', choices=['smote', 'undersample', 'smotetomek', 'none'],
                        help='Método para manejar el desbalance de clases')
    parser.add_argument('--target-column', type=str, default='IsToxic',
                        help='Columna objetivo para estratificación y resampling')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proporción del conjunto de prueba')
    parser.add_argument('--val-size', type=float, default=0.1,
                        help='Proporción del conjunto de validación')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Semilla para reproducibilidad')
    
    args = parser.parse_args()
    
    main(args.input, args.output_dir, args.imbalance_method, args.target_column,
         args.test_size, args.val_size, args.random_state)