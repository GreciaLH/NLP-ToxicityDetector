import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import gradio as gr
import os

# Configuración
MAX_LEN = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Use absolute path instead of relative path
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'bert_toxicity_model')

# Cargar tokenizador y modelo
print(f"Cargando modelo BERT desde: {MODEL_PATH}")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

# Etiquetas
label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Función para predecir toxicidad
def predict_toxicity(comment):
    # Tokenizar el comentario
    encoding = tokenizer.encode_plus(
        comment,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Mover tensores al dispositivo
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    token_type_ids = encoding['token_type_ids'].to(DEVICE)
    
    # Generar predicción
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    # Crear tabla de resultados
    results = pd.DataFrame({
        'Tipo de Toxicidad': label_columns,
        'Probabilidad (%)': [round(prob * 100, 2) for prob in probs]
    })
    
    # Determinar el nivel general de toxicidad
    max_prob = max(probs)
    if max_prob >= 0.7:
        toxicity_level = "Alto nivel de toxicidad"
    elif max_prob >= 0.4:
        toxicity_level = "Nivel medio de toxicidad"
    else:
        toxicity_level = "Bajo nivel de toxicidad"
    
    return results, toxicity_level

# Crear la interfaz de Gradio
with gr.Blocks(title="Predictor de Toxicidad en Comentarios") as app:
    gr.Markdown("# Predictor de Toxicidad en Comentarios")
    gr.Markdown("Esta aplicación analiza comentarios y predice la probabilidad de diferentes tipos de toxicidad usando un modelo BERT entrenado.")
    
    with gr.Row():
        with gr.Column():
            comment_input = gr.Textbox(
                label="Ingrese un comentario",
                placeholder="Escriba aquí el comentario que desea analizar...",
                lines=5
            )
            submit_btn = gr.Button("Analizar Comentario")
        
        with gr.Column():
            results_table = gr.Dataframe(label="Resultados")
            toxicity_level = gr.Textbox(label="Nivel de Toxicidad")
    
    submit_btn.click(
        predict_toxicity,
        inputs=[comment_input],
        outputs=[results_table, toxicity_level]
    )
    
    gr.Markdown("## Ejemplos")
    gr.Examples(
        [
            ["This is a normal comment about the topic."],
            ["You are so stupid and ignorant!"],
            ["I hate people from that country, they should all die."],
            ["The article is well written and informative."]
        ],
        inputs=[comment_input]
    )

# Lanzar la aplicación
if __name__ == "__main__":
    app.launch()