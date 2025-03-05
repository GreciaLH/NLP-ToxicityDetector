"""
Módulo para cargar y transformar datos para los modelos de detección de mensajes de odio.
Incluye clases y funciones para crear datasets y dataloaders de PyTorch.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizer
import logging
from typing import Dict, List, Tuple, Union, Optional
import os
import sys

# Agregar directorio raíz al path para importar módulos del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config import HATE_LABELS
from src.data.preprocessing import preprocess_for_transformer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HateSpeechDataset(Dataset):
    """
    Dataset personalizado para los datos de detección de mensajes de odio.
    """
    
    def __init__(self, 
                texts: List[str], 
                labels: np.ndarray, 
                tokenizer: PreTrainedTokenizer,
                max_length: int = 256):
        """
        Inicializa el dataset.
        
        Args:
            texts: Lista de textos procesados
            labels: Array con las etiquetas binarias
            tokenizer: Tokenizador para el modelo transformer
            max_length: Longitud máxima de secuencia
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx].astype(np.float32)
        
        # Tokenizar texto
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Quitamos la dimensión extra añadida por return_tensors='pt'
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }
        
        return item


def load_and_split_data(
    data_path: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify_column: str = 'IsToxic',
    text_column: str = 'processed_text'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga y divide los datos en conjuntos de entrenamiento, validación y prueba.
    
    Args:
        data_path: Ruta al archivo CSV con los datos
        test_size: Proporción de datos para el conjunto de prueba
        val_size: Proporción de datos para el conjunto de validación
        random_state: Semilla para reproducibilidad
        stratify_column: Columna para estratificar la división
        text_column: Nombre de la columna con el texto procesado
    
    Returns:
        Tupla con los DataFrames de entrenamiento, validación y prueba
    """
    logger.info(f"Cargando datos desde {data_path}")
    df = pd.read_csv(data_path)
    
    # Verificar que el texto procesado existe
    if text_column not in df.columns:
        raise ValueError(f"Columna '{text_column}' no encontrada en el dataset")
    
    # Verificar que las etiquetas existen
    for label in HATE_LABELS:
        if label not in df.columns:
            raise ValueError(f"Etiqueta '{label}' no encontrada en el dataset")
    
    # Primera división: train+val vs test
    if stratify_column in df.columns:
        stratify = df[stratify_column]
    else:
        stratify = None
        logger.warning(f"Columna de estratificación '{stratify_column}' no encontrada. No se estratificará.")
    
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify
    )
    
    # Segunda división: train vs val
    if stratify_column in train_val_df.columns:
        stratify = train_val_df[stratify_column]
    else:
        stratify = None
    
    val_size_adjusted = val_size / (1 - test_size)  # Ajustar para el tamaño de train_val
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=val_size_adjusted, 
        random_state=random_state,
        stratify=stratify
    )
    
    logger.info(f"División completada - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    text_column: str = 'processed_text',
    batch_size: int = 16,
    max_length: int = 256,
    label_columns: List[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crea dataloaders para entrenamiento, validación y prueba.
    
    Args:
        train_df: DataFrame de entrenamiento
        val_df: DataFrame de validación
        test_df: DataFrame de prueba
        tokenizer: Tokenizador para el modelo transformer
        text_column: Nombre de la columna con el texto procesado
        batch_size: Tamaño de batch
        max_length: Longitud máxima de secuencia
        label_columns: Lista de columnas de etiquetas
    
    Returns:
        Tupla con los dataloaders de entrenamiento, validación y prueba
    """
    if label_columns is None:
        label_columns = HATE_LABELS
    
    # Crear datasets
    train_texts = train_df[text_column].tolist()
    val_texts = val_df[text_column].tolist()
    test_texts = test_df[text_column].tolist()
    
    train_labels = train_df[label_columns].values
    val_labels = val_df[label_columns].values
    test_labels = test_df[label_columns].values
    
    train_dataset = HateSpeechDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = HateSpeechDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = HateSpeechDataset(test_texts, test_labels, tokenizer, max_length)
    
    # Crear dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    logger.info(f"Dataloaders creados - Batches: Train {len(train_loader)}, Val {len(val_loader)}, Test {len(test_loader)}")
    return train_loader, val_loader, test_loader


def compute_class_weights(df: pd.DataFrame, label_columns: List[str] = None) -> Dict[str, torch.Tensor]:
    """
    Calcula pesos para manejar el desbalance de clases.
    
    Args:
        df: DataFrame con los datos
        label_columns: Lista de columnas de etiquetas
    
    Returns:
        Diccionario con los pesos para cada etiqueta
    """
    if label_columns is None:
        label_columns = HATE_LABELS
    
    weights = {}
    for label in label_columns:
        # Contar positivos y negativos
        pos_count = df[label].sum()
        neg_count = len(df) - pos_count
        
        # Evitar división por cero
        if pos_count == 0:
            pos_weight = 1.0
        else:
            pos_weight = neg_count / pos_count
        
        weights[label] = torch.tensor([pos_weight], dtype=torch.float)
        
    return weights


if __name__ == "__main__":
    # Código de prueba
    from transformers import BertTokenizer
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    # Ejemplo con un DataFrame pequeño
    data = {
        'processed_text': ['este es un texto de prueba', 'otro texto no tóxico', 'texto con odio y toxicidad'],
        'IsToxic': [1, 0, 1],
        'IsAbusive': [0, 0, 1],
        'IsHatespeech': [0, 0, 1]
    }
    
    df = pd.DataFrame(data)
    
    # Prueba de la clase HateSpeechDataset
    texts = df['processed_text'].tolist()
    labels = df[['IsToxic', 'IsAbusive', 'IsHatespeech']].values
    
    dataset = HateSpeechDataset(texts, labels, tokenizer, max_length=128)
    print(f"Tamaño del dataset: {len(dataset)}")
    
    # Ver un item
    item = dataset[0]
    print(f"Item de ejemplo:")
    print(f"- input_ids shape: {item['input_ids'].shape}")
    print(f"- attention_mask shape: {item['attention_mask'].shape}")
    print(f"- labels: {item['labels']}")