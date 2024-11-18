import streamlit as st
import onnxruntime as ort
import pickle
import numpy as np
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
import re
from urllib.parse import parse_qs, urlparse
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from dotenv import load_dotenv
import os

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
        st.error(f"Error processing text: {str(e)}")
        return None
    

# Streamlit UI
st.title("Text Toxicity Classifier")

# Create tabs for different scenarios
tab1, tab2 = st.tabs(["Single Text Analysis", "Youtube URL Comments Analysis"])

with tab1:
    st.header("Analyze Single Text")
    input_text = st.text_area("Enter text to analyze:")
    
    if st.button("Analyze Text"):
        if input_text:
            es_toxico = False
            probabilities = predict_toxicity(input_text)
            es_toxico = any(prob > 0.5 for _, prob in probabilities.items())
            # Display results
            st.subheader("Classification Results:")
            st.write(f"EsToxico: {es_toxico}")
            for label, prob in probabilities.items():  # Changed to .items()
                st.metric(label, f"{prob:.3f}")  # Changed from :.3% to :.3f
        else:
            st.warning("Please enter some text to analyze.")

with tab2:
    st.header("Analyze YouTube Comments")
    #api_key = st.text_input("Enter your YouTube API Key:", type="password")
    video_url = st.text_input("Enter YouTube Video URL:")
    
    if st.button("Analyze Comments"):
        if api_key and video_url:
            video_id = get_video_id(video_url)
            if video_id:
                with st.spinner("Fetching and analyzing comments..."):
                    try:
                        comments = get_video_comments(video_id, api_key)
                        results = []
                        
                        for comment in comments:
                            es_toxico = False
                            probabilities = predict_toxicity(comment)
                            es_toxico = any(prob > 0.5 for _, prob in probabilities.items())
                            result = {
                                'Comment': comment,
                                'EsToxico': es_toxico,
                                **{label: f"{prob:.3f}" for label, prob in probabilities.items()} 
                            }
                            results.append(result)
                        
                        # Create and display DataFrame
                        df = pd.DataFrame(results)
                        # Set display options to show all columns without scrolling
                        st.dataframe(
                            df.head(10),  # Show only first 10 rows
                            use_container_width=True,  # Makes dataframe use full width
                            height=500,  # Fixed height, adjust as needed
                            hide_index=True  # Optionally hide index for cleaner display
                        )
                        
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.error("Invalid YouTube URL")
        else:
            st.warning("Please enter both API key and video URL.")



# Add this at the bottom of the notebook cell

def main():
    if __name__ == '__main__':
        main()

#print("Access the app at http://127.0.0.1:8501")