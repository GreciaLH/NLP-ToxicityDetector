# YouTube Comment Toxicity Analyzer

## Overview
This project implements a machine learning system to analyze and classify toxic comments from YouTube videos. It uses a multi-label LSTM neural network to detect different types of toxic content, stores the results in a PostgreSQL database, and provides a web interface for analysis.

## Components

### 1. Model Development (`predict_multilabel_keras.ipynb`)
- Built using TensorFlow/Keras
- Multi-label classification for 6 toxicity categories:
  - IsToxic
  - IsAbusive
  - IsProvocative
  - IsObscene
  - IsHatespeech
  - IsRacist
- Architecture:
  - Embedding layer
  - Dual LSTM layers
  - Dropout for regularization
  - Dense output layer with sigmoid activation
- Training metrics:
  - Training Accuracy: 88.13%
  - Testing Accuracy: 92.50%
- Model export:
  - Saved in ONNX format for production
  - Tokenizer serialized using pickle

### 2. Database Management (`db_management.py`)
- PostgreSQL database with two tables:
  ```sql
  -- video table
  CREATE TABLE video (
      videoid integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
      videourl text
  );

  -- comment table
  CREATE TABLE comment (
      comment_id integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
      videoid integer,
      comment text,
      es_toxico boolean,
      is_toxic real,
      is_abusive real,
      is_provocative real,
      is_obscene real,
      is_hatespeech real,
      is_racist real
  );
