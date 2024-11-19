import streamlit as st
import onnxruntime as ort
import pickle
import numpy as np
from googleapiclient.discovery import build
#from googleapiclient.errors import HttpError
import pandas as pd
import re
from urllib.parse import parse_qs, urlparse
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from dotenv import load_dotenv
import os
from db_management import store_prediction, get_es_toxico_distribution, get_score_distributions
import matplotlib.pyplot as plt

load_dotenv()  # load environment variables from .env
api_key = os.getenv('API_KEY')

# Load tokenizer
with open('models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the ONNX model
model_path = 'models/text_classification_model.onnx'
ort_session = ort.InferenceSession(model_path)

# Labels for classification
labels = ['IsToxic', 'IsAbusive', 'IsProvocative', 'IsObscene', 'IsHatespeech', 'IsRacist']


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # remove special characters
    return text


def get_video_id(url):
    """Extract video ID from YouTube URL"""
    query = parse_qs(urlparse(url).query)
    return query.get('v', [None])[0]



def get_video_comments(video_id, api_key):
    """Fetch comments from a YouTube video"""
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    comments = []
    next_page_token = None
    
    while len(comments) < 100:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        )
        response = request.execute()
        
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
            
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
    
    return comments[:100]

    
def predict_toxicity(text):
    try:
        # Preprocess
        text = preprocess_text(text)

        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=100)
        
        # Get input/output names from ONNX model
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        
        # Run inference
        prediction = ort_session.run([output_name], {input_name: padded.astype(np.float32)})[0]
        
        # Format results
        results = {}
        for label, prob in zip(labels, prediction[0]):
            results[label] = float(prob)
        
        return results
    
    except Exception as e:
        st.error(f"Error procesando texto: {str(e)}")
        return None
    

# Streamlit UI
st.title("Clasificador de toxicidad de comentarios")

# Create tabs for different scenarios
tab1, tab2, tab3 = st.tabs(["Analisis de un comentario", "Analisis de un enlace de YouTube", "Visualización de datos"])

with tab1:
    st.header("Analizar un comentario")
    input_text = st.text_area("Ingresar comentario a analizar:")
    
    if st.button("Analizar comentario"):
        if input_text:
            es_toxico = False
            input_text = preprocess_text(input_text)
            probabilities = predict_toxicity(input_text)
            es_toxico = any(prob > 0.5 for label, prob in probabilities.items() if label != 'IsToxic')
            # Display results
            st.subheader("Classification Results:")
            st.write(f"EsToxico: {es_toxico}")
            for label, prob in probabilities.items():  # Changed to .items()
                st.metric(label, f"{prob:.3f}")  # Changed from :.3% to :.3f

                        
            # Store prediction in the database
            store_prediction(
                comment=input_text,
                estoxico=es_toxico,
                is_toxic=probabilities['IsToxic'],
                is_abusive=probabilities['IsAbusive'],
                is_provocative=probabilities['IsProvocative'],
                is_obscene=probabilities['IsObscene'],
                is_hatespeech=probabilities['IsHatespeech'],
                is_racist=probabilities['IsRacist']
            )

        else:
            st.warning("Please enter some text to analyze.")

with tab2:
    st.header("Analizar un enlace de YouTube")
    #api_key = st.text_input("Enter your YouTube API Key:", type="password")
    video_url = st.text_input("Ingresar URL de YouTube:")
    
    if st.button("Analizar comentarios"):
        if api_key and video_url:
            video_id = get_video_id(video_url)
            if video_id:
                with st.spinner("Extrayendo y analizando comentarios..."):
                    try:
                        comments = get_video_comments(video_id, api_key)
                        results = []
                        
                        for comment in comments:
                            input_text = preprocess_text(comment)
                            es_toxico = False
                            probabilities = predict_toxicity(comment)
                            es_toxico = any(prob > 0.5 for label, prob in probabilities.items() if label != 'IsToxic')
                            result = {
                                'Comment': comment,
                                'EsToxico': es_toxico,
                                **{label: f"{prob:.3f}" for label, prob in probabilities.items()} 
                            }
                            results.append(result)
                        
                            # Store prediction in the database
                            store_prediction(
                                comment=comment,
                                estoxico=es_toxico,
                                is_toxic=probabilities['IsToxic'],
                                is_abusive=probabilities['IsAbusive'],
                                is_provocative=probabilities['IsProvocative'],
                                is_obscene=probabilities['IsObscene'],
                                is_hatespeech=probabilities['IsHatespeech'],
                                is_racist=probabilities['IsRacist'],
                                video_url=video_url
                            )


                        # Create and display DataFrame
                        df = pd.DataFrame(results).drop(columns=['IsToxic'])
                        # Set display options to show all columns without scrolling
                        # Adjust column width for 'Comment' column
                        st.dataframe(
                            df.style.set_properties(
                                subset=['Comment'], 
                                **{'width': '200px', 'font-size': '9px'}
                            ),
                            use_container_width=True,
                            height=500,
                            hide_index=True
                        )
                        
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.error("URL invalida. Por favor ingrese un enlace de YouTube valido.")
        else:
            st.warning("Ingrese un enlace de YouTube para continuar.")

with tab3:
    st.header("Visualización de datos")
    
    # Fetch data for es_toxico distribution
    es_toxico_df = get_es_toxico_distribution()
    if es_toxico_df is not None:
        st.subheader("Distribución de EsToxico")
        fig1, ax1 = plt.subplots()
        ax1.pie(es_toxico_df['count'], labels=es_toxico_df['es_toxico'], autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig1)
    

# Fetch and display score distributions
distributions = get_score_distributions()
if distributions is not None:
    st.subheader("Distribución de puntuaciones")
    metrics = ['is_abusive', 'is_provocative', 'is_obscene', 'is_hatespeech', 'is_racist']
    
    # Create a 3x2 grid of subplots
    fig3, axes = plt.subplots(3, 2, figsize=(15, 20))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics):
        df = distributions[metric]
        axes[idx].bar(df['bucket_range'], df['percentage'])
        axes[idx].set_title(f'Distribución de {metric}')
        axes[idx].set_xlabel('Rango de valores')
        axes[idx].set_ylabel('Porcentaje de comentarios (%)')
        axes[idx].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}%'.format(y)))
        plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45)
    
    # Remove the extra subplot
    fig3.delaxes(axes[5])
    plt.tight_layout()
    st.pyplot(fig3)

# Add this at the bottom of the notebook cell

def main():
    if __name__ == '__main__':
        main()

#print("Access the app at http://127.0.0.1:8501")