import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import os
import pandas as pd
from sqlalchemy import create_engine

# Load environment variables
load_dotenv()

# Database connection parameters
DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')

# Create SQLAlchemy engine
engine = create_engine(f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}')


def store_prediction(comment, estoxico, is_toxic, is_abusive, is_provocative, is_obscene, is_hatespeech, is_racist, video_url=None):
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
        cursor = conn.cursor()
        
        video_id = None
        if video_url:
            # Check if the video URL already exists
            check_video_query = """
                SELECT videoid FROM video WHERE videourl = %s
            """
            cursor.execute(check_video_query, (video_url,))
            result = cursor.fetchone()
            
            if result:
                video_id = result[0]
            else:
                # Insert the video URL into the video table
                insert_video_query = """
                    INSERT INTO video (videourl)
                    VALUES (%s)
                    RETURNING videoid
                """
                cursor.execute(insert_video_query, (video_url,))
                video_id = cursor.fetchone()[0]
        
        # Check if the comment + videoid combination already exists
        check_comment_query = """
            SELECT comment_id FROM comment WHERE comment = %s AND videoid = %s
        """
        cursor.execute(check_comment_query, (comment, video_id))
        result = cursor.fetchone()
        
        if not result:
            # Insert the prediction into the comment table
            insert_comment_query = """
                INSERT INTO comment (videoid, comment, es_toxico, is_toxic, is_abusive, is_provocative, is_obscene, is_hatespeech, is_racist)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_comment_query, (video_id, comment, estoxico, is_toxic, is_abusive, is_provocative, is_obscene, is_hatespeech, is_racist))
            conn.commit()
        
    except Exception as e:
        print(f"Error storing prediction: {e}")
    finally:
        cursor.close()
        conn.close()

def get_es_toxico_distribution():
    try:
        query = "SELECT es_toxico, COUNT(*) FROM comment GROUP BY es_toxico"
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        print(f"Error fetching es_toxico distribution: {e}")
        return None

def get_score_distributions():
    """Get the distribution of scores for each toxicity metric in buckets of 0.1 intervals."""
    try:
        metrics = ['is_abusive', 'is_provocative', 'is_obscene', 'is_hatespeech', 'is_racist']
        distributions = {}
        
        for metric in metrics:
            query = f"""
                WITH bucket_counts AS (
                    SELECT 
                        width_bucket({metric}, 0, 1, 10) as bucket,
                        count(*) as count,
                        SUM(COUNT(*)) OVER () as total_count
                    FROM comment
                    GROUP BY bucket
                )
                SELECT 
                    bucket,
                    count,
                    (count * 100.0 / total_count) as percentage
                FROM bucket_counts
                ORDER BY bucket
            """
            df = pd.read_sql(query, engine)
            df['bucket_range'] = df['bucket'].apply(lambda x: f'{(x-1)/10:.1f}-{x/10:.1f}')
            distributions[metric] = df
            
        return distributions
    except Exception as e:
        print(f"Error fetching score distributions: {e}")
        return None