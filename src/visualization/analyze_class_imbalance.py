#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para analizar el desbalance de clases en el dataset de comentarios de YouTube.
Este script realiza:
1. Visualización de la distribución de clases
2. Análisis del impacto del desbalance
3. Comparación de diferentes estrategias para manejar el desbalance
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import logging
import argparse
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def visualize_class_distribution(df, output_dir):
    """
    Visualiza la distribución de clases en el dataset
    
    Args:
        df: DataFrame con los datos
        output_dir: Directorio para guardar las visualizaciones
    """
    # Crear directorio si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logger.info("Visualizando distribución de clases")
    
    # Obtener columnas de etiquetas
    label_columns = [col for col in df.columns if col.startswith('Is')]
    
    # Calcular distribución de clases
    class_distribution = df[label_columns].mean().sort_values(ascending=False) * 100
    
    # Visualizar distribución mediante un gráfico de barras
    plt.figure(figsize=(12, 8))
    sns.barplot(x=class_distribution.index, y=class_distribution.values)
    plt.title('Distribución de Clases (%)')
    plt.xlabel('Categoría')
    plt.ylabel('Porcentaje')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    
    # Visualizar distribución como una tabla
    class_distribution_df = pd.DataFrame({
        'Categoría': class_distribution.index,
        'Porcentaje (%)': class_distribution.values.round(2),
        'Conteo': (df[label_columns].sum()).values
    })
    logger.info(f"Distribución de clases:\n{class_distribution_df}")
    
    # Guardar distribución en CSV
    class_distribution_df.to_csv(os.path.join(output_dir, 'class_distribution.csv'), index=False)
    
    # Analizar desbalance en clases combinadas (para el caso multi-etiqueta)
    plt.figure(figsize=(10, 6))
    label_counts = df[label_columns].sum(axis=1).value_counts().sort_index()
    sns.barplot(x=label_counts.index, y=label_counts.values)
    plt.title('Número de Etiquetas Positivas por Comentario')
    plt.xlabel('Número de Etiquetas')
    plt.ylabel('Número de Comentarios')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multi_label_distribution.png'))
    
    # Analizar correlación entre etiquetas
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[label_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlación entre Etiquetas')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'label_correlation.png'))

def analyze_imbalance_impact(df, text_column, target_column, output_dir, random_state=42):
    """
    Analiza el impacto del desbalance de clases en el rendimiento del modelo
    
    Args:
        df: DataFrame con los datos
        text_column: Columna que contiene el texto
        target_column: Columna objetivo
        output_dir: Directorio para guardar las visualizaciones
        random_state: Semilla para reproducibilidad
    """
    logger.info(f"Analizando impacto del desbalance en {target_column}")
    
    # Preparar datos
    X = df[text_column]
    y = df[target_column]
    
    # Convertir texto a características TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, min_df=5)
    X_tfidf = vectorizer.fit_transform(X)
    
    # Dividir en entrenamiento y prueba (70/30)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.3, stratify=y, random_state=random_state
    )
    
    # Entrenar modelo con datos desbalanceados
    logger.info("Entrenando modelo con datos desbalanceados")
    model_imbalanced = LogisticRegression(random_state=random_state, max_iter=1000)
    model_imbalanced.fit(X_train, y_train)
    y_pred_imbalanced = model_imbalanced.predict(X_test)
    
    # Evaluar modelo con datos desbalanceados
    imbalanced_report = classification_report(y_test, y_pred_imbalanced, output_dict=True)
    logger.info(f"Informe de clasificación (desbalanceado):\n{pd.DataFrame(imbalanced_report).T}")
    
    # Crear y visualizar matriz de confusión
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred_imbalanced)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión ({target_column} - Desbalanceado)')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{target_column}_confusion_matrix_imbalanced.png'))
    
    # Calcular y almacenar métricas para comparación posterior
    metrics_imbalanced = {
        'accuracy': imbalanced_report['accuracy'],
        'precision_0': imbalanced_report['False']['precision'],
        'precision_1': imbalanced_report['True']['precision'],
        'recall_0': imbalanced_report['False']['recall'],
        'recall_1': imbalanced_report['True']['recall'],
        'f1_0': imbalanced_report['False']['f1-score'],
        'f1_1': imbalanced_report['True']['f1-score'],
    }
    
    # Aplicar SMOTE para balancear clases
    logger.info("Aplicando SMOTE para balancear clases")
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Entrenar modelo con datos balanceados usando SMOTE
    model_smote = LogisticRegression(random_state=random_state, max_iter=1000)
    model_smote.fit(X_train_smote, y_train_smote)
    y_pred_smote = model_smote.predict(X_test)
    
    # Evaluar modelo con datos balanceados usando SMOTE
    smote_report = classification_report(y_test, y_pred_smote, output_dict=True)
    logger.info(f"Informe de clasificación (SMOTE):\n{pd.DataFrame(smote_report).T}")
    
    # Crear y visualizar matriz de confusión para SMOTE
    plt.figure(figsize=(8, 6))
    cm_smote = confusion_matrix(y_test, y_pred_smote)
    sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión ({target_column} - SMOTE)')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{target_column}_confusion_matrix_smote.png'))
    
    # Calcular y almacenar métricas para SMOTE
    metrics_smote = {
        'accuracy': smote_report['accuracy'],
        'precision_0': smote_report['False']['precision'],
        'precision_1': smote_report['True']['precision'],
        'recall_0': smote_report['False']['recall'],
        'recall_1': smote_report['True']['recall'],
        'f1_0': smote_report['False']['f1-score'],
        'f1_1': smote_report['True']['f1-score'],
    }
    
    # Aplicar UnderSampling para balancear clases
    logger.info("Aplicando UnderSampling para balancear clases")
    undersampler = RandomUnderSampler(random_state=random_state)
    X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)
    
    # Entrenar modelo con datos balanceados usando UnderSampling
    model_under = LogisticRegression(random_state=random_state, max_iter=1000)
    model_under.fit(X_train_under, y_train_under)
    y_pred_under = model_under.predict(X_test)
    
    # Evaluar modelo con datos balanceados usando UnderSampling
    under_report = classification_report(y_test, y_pred_under, output_dict=True)
    logger.info(f"Informe de clasificación (UnderSampling):\n{pd.DataFrame(under_report).T}")
    
    # Calcular y almacenar métricas para UnderSampling
    metrics_under = {
        'accuracy': under_report['accuracy'],
        'precision_0': under_report['False']['precision'],
        'precision_1': under_report['True']['precision'],
        'recall_0': under_report['False']['recall'],
        'recall_1': under_report['True']['recall'],
        'f1_0': under_report['False']['f1-score'],
        'f1_1': under_report['True']['f1-score'],
    }
    
    # Comparar métricas
    metrics_df = pd.DataFrame({
        'Desbalanceado': metrics_imbalanced,
        'SMOTE': metrics_smote,
        'UnderSampling': metrics_under
    }).T
    
    logger.info(f"Comparación de métricas:\n{metrics_df}")
    metrics_df.to_csv(os.path.join(output_dir, f'{target_column}_resampling_comparison.csv'))
    
    # Visualizar curvas ROC
    plt.figure(figsize=(10, 8))
    
    # ROC para modelo desbalanceado
    y_score_imbalanced = model_imbalanced.predict_proba(X_test)[:, 1]
    fpr_imbalanced, tpr_imbalanced, _ = roc_curve(y_test, y_score_imbalanced)
    roc_auc_imbalanced = auc(fpr_imbalanced, tpr_imbalanced)
    plt.plot(fpr_imbalanced, tpr_imbalanced, label=f'Desbalanceado (AUC = {roc_auc_imbalanced:.2f})')
    
    # ROC para modelo con SMOTE
    y_score_smote = model_smote.predict_proba(X_test)[:, 1]
    fpr_smote, tpr_smote, _ = roc_curve(y_test, y_score_smote)
    roc_auc_smote = auc(fpr_smote, tpr_smote)
    plt.plot(fpr_smote, tpr_smote, label=f'SMOTE (AUC = {roc_auc_smote:.2f})')
    
    # ROC para modelo con UnderSampling
    y_score_under = model_under.predict_proba(X_test)[:, 1]
    fpr_under, tpr_under, _ = roc_curve(y_test, y_score_under)
    roc_auc_under = auc(fpr_under, tpr_under)
    plt.plot(fpr_under, tpr_under, label=f'UnderSampling (AUC = {roc_auc_under:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curva ROC - {target_column}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{target_column}_roc_comparison.png'))
    
    # Visualizar precision y recall para diferentes estrategias
    metrics_to_plot = ['precision_1', 'recall_1', 'f1_1']
    plt.figure(figsize=(12, 6))
    metrics_df_subset = metrics_df[metrics_to_plot]
    metrics_df_subset.plot(kind='bar')
    plt.title(f'Comparación de Métricas para Clase Positiva - {target_column}')
    plt.ylabel('Valor')
    plt.ylim([0, 1])
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{target_column}_metrics_comparison.png'))
    
    return metrics_df

def main(input_filepath, output_dir, text_column='cleaned_text'):
    """Función principal para analizar el desbalance de clases"""
    # Cargar datos
    df = load_data(input_filepath)
    
    # Visualizar distribución de clases
    visualize_class_distribution(df, output_dir)
    
    # Analizar impacto del desbalance para cada etiqueta
    label_columns = [col for col in df.columns if col.startswith('Is')]
    
    # Seleccionar las etiquetas más relevantes para el análisis
    # Podemos analizar todas, pero para ahorrar tiempo analizamos las principales
    main_labels = ['IsToxic', 'IsHatespeech', 'IsAbusive', 'IsRacist']
    
    all_metrics = {}
    for label in main_labels:
        if label in df.columns:
            metrics = analyze_imbalance_impact(df, text_column, label, output_dir)
            all_metrics[label] = metrics
    
    logger.info("Análisis de desbalance de clases completado")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Análisis de desbalance de clases para el detector de mensajes de odio')
    parser.add_argument('--input', type=str, default='data/processed/youtube_comments_processed.csv',
                        help='Ruta al archivo CSV procesado')
    parser.add_argument('--output-dir', type=str, default='reports/class_imbalance',
                        help='Directorio para guardar las visualizaciones')
    parser.add_argument('--text-column', type=str, default='cleaned_text',
                        help='Columna que contiene el texto procesado')
    
    args = parser.parse_args()
    
    main(args.input, args.output_dir, args.text_column)