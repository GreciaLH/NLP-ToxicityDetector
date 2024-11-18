import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Database connection parameters
DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')

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

