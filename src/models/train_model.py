import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import pickle
import time

# Descargar recursos de NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Cargar los datos
print("Cargando datos...")
train_df = pd.read_csv('../../data/raw/train.csv')
test_df = pd.read_csv('../../data/raw/test.csv')

# Función para limpiar texto
def clean_text(text):
    if isinstance(text, str):
        # Convertir a minúsculas
        text = text.lower()
        # Eliminar URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Eliminar etiquetas HTML
        text = re.sub(r'<.*?>', '', text)
        # Eliminar caracteres especiales y números
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Eliminar espacios múltiples
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

# Preprocesar los comentarios
print("Preprocesando texto...")
train_df['cleaned_comment'] = train_df['comment_text'].apply(clean_text)
test_df['cleaned_comment'] = test_df['comment_text'].apply(clean_text)

# Vectorización TF-IDF
print("Vectorizando texto...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=50000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2)
)

X_train = tfidf_vectorizer.fit_transform(train_df['cleaned_comment'])
X_test = tfidf_vectorizer.transform(test_df['cleaned_comment'])

# Etiquetas
label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_train = train_df[label_columns].values

# Dividir el conjunto de entrenamiento para validación
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Entrenar modelo
print("Entrenando modelo...")
start_time = time.time()

# Usamos Regresión Logística como base para MultiOutputClassifier
base_model = LogisticRegression(C=5, solver='sag', max_iter=300, n_jobs=-1, random_state=42)
model = MultiOutputClassifier(base_model, n_jobs=-1)

model.fit(X_train_split, y_train_split)

print(f"Tiempo de entrenamiento: {time.time() - start_time:.2f} segundos")

# Evaluar en conjunto de validación
print("Evaluando modelo...")
y_val_pred = model.predict_proba(X_val)

# Calcular AUC-ROC para cada etiqueta
auc_scores = []
for i, label in enumerate(label_columns):
    # Extraer las probabilidades de la clase positiva (índice 1)
    y_val_pred_proba = y_val_pred[i][:, 1]
    auc = roc_auc_score(y_val[:, i], y_val_pred_proba)
    auc_scores.append(auc)
    print(f"AUC-ROC para {label}: {auc:.4f}")

print(f"AUC-ROC promedio: {np.mean(auc_scores):.4f}")

# Predecir en conjunto de prueba
print("Generando predicciones para el conjunto de prueba...")
test_predictions = []
for i, label in enumerate(label_columns):
    # Extraer las probabilidades de la clase positiva (índice 1)
    test_pred = model.estimators_[i].predict_proba(X_test)[:, 1]
    test_predictions.append(test_pred)

# Crear DataFrame para las predicciones
submission_df = pd.DataFrame({
    'id': test_df['id']
})

for i, label in enumerate(label_columns):
    submission_df[label] = test_predictions[i]

# Guardar predicciones
submission_df.to_csv('../../data/processed/submission.csv', index=False)
print("Predicciones guardadas en submission.csv")

# Guardar modelo y vectorizador
with open('../../models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

with open('../../models/toxicity_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Modelo y vectorizador guardados correctamente")