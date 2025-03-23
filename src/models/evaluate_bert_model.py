import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # Añadido para barras de progreso

# Configuración
MAX_LEN = 128
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = '../../models/bert_toxicity_model'

print(f"Usando dispositivo: {DEVICE}")

# Cargar datos de prueba
print("Cargando datos de prueba...")
test_df = pd.read_csv('../../data/raw/test.csv')
print(f"Datos cargados: {len(test_df)} comentarios")

# Definir etiquetas
label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Clase Dataset personalizada
class ToxicityDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten()
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
            
        return item

# Cargar tokenizador y modelo
print("Cargando modelo BERT entrenado...")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

# Crear dataset de prueba
test_dataset = ToxicityDataset(
    texts=test_df['comment_text'].values,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Función para generar predicciones
def generate_predictions(model, data_loader, device):
    model.eval()
    predictions = []
    comment_ids = []
    
    # Añadir barra de progreso
    progress_bar = tqdm(data_loader, desc="Generando predicciones", leave=True)
    
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()
            predictions.extend(probs)
            
            # Actualizar información en la barra de progreso
            progress_bar.set_postfix({"batch_size": input_ids.size(0)})
    
    return np.array(predictions)

# Generar predicciones
print("Generando predicciones...")
test_predictions = generate_predictions(model, test_loader, DEVICE)

# Crear DataFrame para las predicciones
submission_df = pd.DataFrame({
    'id': test_df['id']
})

for i, label in enumerate(label_columns):
    submission_df[label] = test_predictions[:, i]

# Guardar predicciones
submission_df.to_csv('../../data/processed/bert_final_submission.csv', index=False)
print("Predicciones guardadas en bert_final_submission.csv")

# Visualizar distribución de predicciones
plt.figure(figsize=(15, 10))
for i, label in enumerate(label_columns):
    plt.subplot(2, 3, i+1)
    sns.histplot(submission_df[label], bins=50, kde=True)
    plt.title(f'Distribución de predicciones: {label}')
    plt.xlabel('Probabilidad')
    plt.ylabel('Frecuencia')

plt.tight_layout()
plt.savefig('../../reports/bert_predictions_distribution.png')
print("Gráfico de distribución de predicciones guardado")