import psycopg2
from psycopg2 import sql

# Database connection parameters
DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')


def store_prediction_tab1(comment, estoxico, is_toxic, is_abusive, is_provocative, is_obscene, is_hatespeech, is_racist):
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
        cursor = conn.cursor()
        
        # Insert the prediction into the database
        insert_query = sql.SQL("""
            INSERT INTO comment (comment, estoxico, is_toxic, is_abusive, is_provocative, is_obscene, is_hatespeech, is_racist)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """)
        cursor.execute(insert_query, (comment, estoxico, is_toxic, is_abusive, is_provocative, is_obscene, is_hatespeech, is_racist))
        
        # Commit the transaction
        conn.commit()
        
    except Exception as e:
        print(f"Error storing prediction for tab1: {e}")
    finally:
        cursor.close()
        conn.close()

def store_prediction_tab2(video_url, comments):
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
        cursor = conn.cursor()
        
        # Insert the video URL into the database
        insert_video_query = sql.SQL("""
            INSERT INTO predictions_tab2 (video_url)
            VALUES (%s)
            RETURNING id
        """)
        cursor.execute(insert_video_query, (video_url,))
        video_id = cursor.fetchone()[0]
        
        # Insert each comment prediction into the database
        insert_comment_query = sql.SQL("""
            INSERT INTO comments (video_id, comment, estoxico, is_toxic, is_abusive, is_provocative, is_obscene, is_hatespeech, is_racist)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """)
        for comment in comments:
            cursor.execute(insert_comment_query, (
                video_id, 
                comment['text'], 
                comment['estoxico'], 
                comment['is_toxic'], 
                comment['is_abusive'], 
                comment['is_provocative'], 
                comment['is_obscene'], 
                comment['is_hatespeech'], 
                comment['is_racist']
            ))
        
        # Commit the transaction
        conn.commit()
        
    except Exception as e:
        print(f"Error storing prediction for tab2: {e}")
    finally:
        cursor.close()
        conn.close()