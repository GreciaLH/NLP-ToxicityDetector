import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar predicciones
lr_predictions = pd.read_csv('../../data/processed/submission.csv')
bert_predictions = pd.read_csv('../../data/processed/bert_final_submission.csv')

# Etiquetas
label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Crear gráficos de distribución de predicciones
plt.figure(figsize=(15, 10))

for i, label in enumerate(label_columns):
    plt.subplot(2, 3, i+1)
    
    # Distribución de predicciones de Regresión Logística
    sns.histplot(lr_predictions[label], color='blue', alpha=0.5, bins=50, label='Logistic Regression')
    
    # Distribución de predicciones de BERT
    sns.histplot(bert_predictions[label], color='red', alpha=0.5, bins=50, label='BERT')
    
    plt.title(f'Distribución de predicciones: {label}')
    plt.xlabel('Probabilidad')
    plt.ylabel('Frecuencia')
    plt.legend()

plt.tight_layout()
plt.savefig('../../reports/models_comparison.png')
print("Gráfico de comparación guardado")

# Crear un informe de comparación
comparison_report = pd.DataFrame({
    'Label': label_columns,
    'LR_Mean': [lr_predictions[col].mean() for col in label_columns],
    'BERT_Mean': [bert_predictions[col].mean() for col in label_columns],
    'LR_Std': [lr_predictions[col].std() for col in label_columns],
    'BERT_Std': [bert_predictions[col].std() for col in label_columns],
    'LR_Min': [lr_predictions[col].min() for col in label_columns],
    'BERT_Min': [bert_predictions[col].min() for col in label_columns],
    'LR_Max': [lr_predictions[col].max() for col in label_columns],
    'BERT_Max': [bert_predictions[col].max() for col in label_columns]
})

print("Comparación de estadísticas de predicciones:")
print(comparison_report)

# Guardar el informe
comparison_report.to_csv('../../data/processed/models_comparison_report.csv', index=False)
print("Informe de comparación guardado en models_comparison_report.csv")