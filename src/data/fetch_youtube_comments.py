"""
Módulo para obtener comentarios de videos de YouTube usando la API de YouTube.
"""

import os
import time
from typing import Dict, List, Optional

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm

from src.config import YOUTUBE_API_KEY, RAW_DATA_DIR


def create_youtube_client():
    """Crea un cliente para la API de YouTube."""
    if not YOUTUBE_API_KEY:
        print("YouTube API key no encontrada. Configure la variable YOUTUBE_API_KEY.")
        return None
    
    try:
        client = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        return client
    except Exception as e:
        print(f"Error al crear cliente de YouTube: {e}")
        return None


def fetch_video_comments(youtube_client, video_id, max_results=100):
    """Obtiene comentarios para un video específico de YouTube."""
    comments = []
    next_page_token = None
    
    try:
        while len(comments) < max_results:
            response = youtube_client.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=min(100, max_results - len(comments)),
                pageToken=next_page_token
            ).execute()
            
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'CommentId': item['id'],
                    'VideoId': video_id,
                    'Text': comment['textDisplay'],
                    'Author': comment['authorDisplayName'],
                    'PublishedAt': comment['publishedAt'],
                    'LikeCount': comment['likeCount'],
                })
            
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
            
            time.sleep(0.1)  # Evitar límites de tasa
    
    except HttpError as e:
        print(f"Error HTTP para el video {video_id}: {e}")
    except Exception as e:
        print(f"Error al obtener comentarios para el video {video_id}: {e}")
    
    return comments


def save_comments_to_csv(comments_df, output_path=None):
    """Guarda los comentarios obtenidos en un archivo CSV."""
    if comments_df.empty:
        print("No hay comentarios para guardar.")
        return
    
    if not output_path:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_path = RAW_DATA_DIR / f"youtube_comments_{timestamp}.csv"
    
    comments_df.to_csv(output_path, index=False)
    print(f"Guardados {len(comments_df)} comentarios en {output_path}")