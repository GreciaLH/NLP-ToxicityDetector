# Detector de Toxicidad en Comentarios

Este proyecto implementa modelos de aprendizaje automático para detectar diferentes tipos de toxicidad en comentarios, incluyendo:
- Comentarios tóxicos
- Comentarios severamente tóxicos
- Obscenidad
- Amenazas
- Insultos
- Odio basado en identidad

## Descripción

El proyecto utiliza técnicas de procesamiento de lenguaje natural (NLP) y aprendizaje automático para clasificar comentarios según su nivel de toxicidad. Se han implementado y comparado dos enfoques principales:

1. **Modelo de Regresión Logística** con vectorización TF-IDF para representar el texto
2. **Modelo BERT** (Bidirectional Encoder Representations from Transformers), un modelo de lenguaje pre-entrenado de última generación

Ambos modelos han sido entrenados y evaluados utilizando un conjunto de datos etiquetado, y sus resultados han sido comparados para determinar el enfoque más efectivo.

## Información del Dataset

Este proyecto utiliza el dataset del desafío [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) de Kaggle.

### Descripción del Conjunto de Datos

El dataset contiene una gran cantidad de comentarios de Wikipedia etiquetados por evaluadores humanos según diferentes tipos de toxicidad:

- **toxic**: Comentarios con contenido tóxico general
- **severe_toxic**: Comentarios con toxicidad severa
- **obscene**: Comentarios con contenido obsceno
- **threat**: Comentarios que contienen amenazas
- **insult**: Comentarios insultantes
- **identity_hate**: Comentarios con odio basado en identidad (raza, género, etc.)

### Archivos del Dataset

- **train.csv**: El conjunto de entrenamiento, contiene comentarios con sus etiquetas binarias para cada categoría de toxicidad.
- **test.csv**: El conjunto de prueba. Contiene comentarios para los cuales se deben predecir las probabilidades de toxicidad.

## Estructura del Proyecto

```bash
raíz/
│
├── data/                          # Datos del proyecto
│   ├── raw/                       # Datos originales sin procesar
│   │   ├── train.csv              # Conjunto de entrenamiento original
│   │   └── test.csv               # Conjunto de prueba original
│   │
│   └── processed/                 # Datos procesados
│       ├── submission.csv         # Predicciones del modelo de regresión logística
│       ├── bert_submission.csv    # Predicciones del modelo BERT
│       ├── bert_final_submission.csv # Predicciones finales del modelo BERT
│       └── models_comparison_report.csv # Informe de comparación de modelos
│
├── models/                        # Modelos entrenados
│   ├── bert_toxicity_model/       # Modelo BERT entrenado
│   │   ├── config.json            # Configuración del modelo BERT
│   │   ├── model.safetensors      # Pesos del modelo BERT
│   │   ├── special_tokens_map.json # Mapeo de tokens especiales
│   │   ├── tokenizer_config.json  # Configuración del tokenizador
│   │   └── vocab.txt              # Vocabulario del modelo
│   │
│   ├── toxicity_model.pkl         # Modelo de regresión logística serializado
│   └── tfidf_vectorizer.pkl       # Vectorizador TF-IDF serializado
│
├── notebooks/                     # Jupyter notebooks para análisis
│   └── toxicity_eda.ipynb         # Análisis exploratorio de datos
│
├── src/                           # Código fuente del proyecto
│   ├── models/                    # Scripts para modelos
│   │   ├── train_model.py         # Entrenamiento de modelo de regresión logística
│   │   ├── train_bert_model.py    # Entrenamiento de BERT
│   │   └── evaluate_bert_model.py # Evaluación de modelo BERT
│   │
│   └── visualization/             # Scripts para visualización
│       └── compare_models_performance.py # Comparación de modelos
│
├── reports/                       # Informes generados
│   ├── label_distribution.png     # Distribución de etiquetas
│   ├── comment_length_distribution.png # Distribución de longitud de comentarios
│   └── models_comparison.png      # Comparación visual de modelos
│
├── app/                           # Aplicación de predicción
│   └── bert_toxicity_app.py       # Aplicación para predicciones
│
├── requirements.txt               # Dependencias del proyecto
└── README.md                      # Este archivo
```

## Instalación

Para instalar las dependencias necesarias:

```bash
pip install -r requirements.txt
 ```

## Uso
### 1. Exploración de Datos
Para ejecutar el análisis exploratorio de datos:

```bash
python src/data/explore_data.py
 ```

Este script genera visualizaciones sobre la distribución de etiquetas, longitud de comentarios y otras características importantes del conjunto de datos.

### 2. Entrenamiento de Modelos

Modelo de Regresión Logística
```bash
python src/models/train_model.py
 ```

Este script:

- Preprocesa el texto (limpieza, tokenización)
- Vectoriza los comentarios usando TF-IDF
- Entrena un modelo de regresión logística multiclase
- Evalúa el rendimiento usando AUC-ROC
- Guarda el modelo entrenado y el vectorizador

Modelo BERT

El entrenamiento del modelo BERT se realizó en Google Colab debido a sus requisitos computacionales:

```bash
python src/models/train_bert_model.ipynb
 ```

Este script:

- Prepara los datos para el formato requerido por BERT
- Configura y entrena el modelo BERT para clasificación multi-etiqueta
- Evalúa el rendimiento usando AUC-ROC
- Guarda el modelo entrenado
### 3. Evaluación de Modelos
Para evaluar el modelo BERT en nuevos datos:

```bash
python src/models/evaluate_bert_model.py
 ```

### 4. Comparación de Modelos
Para comparar el rendimiento de ambos modelos:

```bash
python src/visualization/compare_models_performance.py
 ```

Este script genera visualizaciones comparativas y un informe detallado de las estadísticas de predicción de ambos modelos.

### 5. Aplicación de Predicción
Para ejecutar la aplicación de predicción de toxicidad:

```bash
python app/bert_toxicity_app.py
 ```

Esta aplicación permite ingresar comentarios y obtener predicciones de toxicidad en tiempo real.

## Resultados
Los resultados de la comparación entre los modelos muestran que:

- El modelo BERT generalmente proporciona predicciones más precisas, especialmente para categorías con menos ejemplos como "threat" e "identity_hate".
- El modelo de Regresión Logística ofrece un buen rendimiento con un costo computacional significativamente menor.
- Ambos modelos muestran distribuciones bimodales en sus predicciones, con concentraciones en los extremos (0 y 1).
Para más detalles, consulte el informe de comparación en reports/models_comparison_report.csv y las visualizaciones en reports/figures/ .

## Trabajo Futuro
Posibles mejoras para el proyecto:

- Implementar técnicas de aumento de datos para clases minoritarias
- Explorar modelos más ligeros basados en transformers (DistilBERT, ALBERT)
- Desarrollar una API REST para el servicio de predicción
- Implementar un sistema de retroalimentación para mejorar continuamente el modelo
## Contribuciones
Las contribuciones son bienvenidas. Por favor, siga estos pasos:

1. Fork el repositorio
2. Cree una rama para su característica ( git checkout -b feature/nueva-caracteristica )
3. Realice sus cambios y haga commit ( git commit -m 'Añadir nueva característica' )
4. Push a la rama ( git push origin feature/nueva-caracteristica )
5. Abra un Pull Request
## Licencia
Este proyecto está licenciado bajo la Licencia MIT - vea el archivo LICENSE para más detalles.
