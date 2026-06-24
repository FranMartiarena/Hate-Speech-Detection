# ============================================================
# IMPORTS
# ============================================================
from datasets import Dataset          # para crear datasets en formato HuggingFace
from transformers import (
    AutoTokenizer,                    # tokenizador compatible con el modelo elegido
    AutoModelForSequenceClassification,  # modelo base + cabeza clasificadora
    TrainingArguments,                # hiperparámetros del entrenamiento
    Trainer,                          # loop de entrenamiento listo para usar
)
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

# ============================================================
# 1. CARGAR TUS DATASETS
# ============================================================
# Asumiendo que tenés archivos CSV con columnas "text" y "label"
# Cambiá esto según el formato real de tus archivos

data = pd.read_csv("../data_set/hateval2019_en_convinado_balanceado.csv") 

data = data.rename(columns={'HS': 'labels'})

dataset_train, dataset_test = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data["labels"]
)

# Convertir a formato Dataset de HuggingFace
# (el Trainer solo acepta este formato, no DataFrames de pandas)
dataset_train = Dataset.from_pandas(data[["text", "labels"]])
dataset_test = Dataset.from_pandas(data[["text", "labels"]])


# ============================================================
# 2. TOKENIZACIÓN
# ============================================================
# El modelo no entiende texto crudo — necesita convertirlo a números.
# El tokenizador hace: texto → lista de IDs de tokens + attention mask

MODEL_NAME = "distilbert-base-uncased"  
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,    # si el texto es muy largo, lo corta
        padding="max_length",  # si es corto, lo rellena con [PAD] hasta max_length
        max_length=128,     # 128 tokens suele alcanzar para textos de redes sociales
                            # subilo a 256 si tus textos son más largos
    )

# .map() aplica la función a todo el dataset en batches (eficiente)
# batched=True significa que procesa muchos ejemplos juntos, no uno por uno
dataset_train = dataset_train.map(tokenize, batched=True)
dataset_test  = dataset_test.map(tokenize, batched=True)


# ============================================================
# 3. MODELO
# ============================================================
# Carga los pesos preentrenados del cuerpo + agrega cabeza clasificadora nueva
# num_labels=2 → salida de 2 neuronas (normal / odio)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)


# ============================================================
# 4. MÉTRICAS DE EVALUACIÓN
# ============================================================
# El Trainer llama a esta función al final de cada época para evaluar.
# eval_pred es una tupla (logits, labels) donde:
#   - logits: predicciones crudas del modelo, shape (N, 2)
#   - labels: etiquetas reales, shape (N,)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # argmax sobre las 2 columnas → elige la clase con mayor score
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        # f1 "binary" asume que 1 es la clase positiva (odio)
        # usá average="macro" si las clases están muy desbalanceadas
        "f1": f1_score(labels, preds, average="binary"),
    }


# ============================================================
# 5. HIPERPARÁMETROS DE ENTRENAMIENTO
# ============================================================
args = TrainingArguments(
    output_dir="./hate-speech-checkpoints_en",  # dónde guardar checkpoints
    num_train_epochs=3,           # pasadas completas sobre el dataset de train
    per_device_train_batch_size=16,  # ejemplos por paso de entrenamiento
    per_device_eval_batch_size=32,   # puede ser más grande, no acumula gradientes
    eval_strategy="epoch",  # evaluar al final de cada época
    save_strategy="epoch",        # guardar checkpoint al final de cada época
    load_best_model_at_end=True,  # al terminar, carga el mejor checkpoint
    metric_for_best_model="f1",   # criterio para elegir el "mejor"
    learning_rate=2e-5,           # tasa de aprendizaje; 1e-5 a 5e-5 es el rango típico
    weight_decay=0.01,            # regularización L2, evita overfitting
    fp16=torch.cuda.is_available(),  # entrenamiento en float16 si hay GPU → más rápido
)


# ============================================================
# 6. TRAINER Y ENTRENAMIENTO
# ============================================================
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset_train,  # tu dataset de train ya separado
    eval_dataset=dataset_test,    # tu dataset de test ya separado
    compute_metrics=compute_metrics,
)

# Lanza el entrenamiento
trainer.train()


# ============================================================
# 7. EVALUACIÓN FINAL Y GUARDADO
# ============================================================
# Evalúa el mejor modelo sobre el test set y muestra las métricas
results = trainer.evaluate()
print(results)

# Guarda el modelo final (pesos + configuración) y el tokenizador
# Esto te permite cargarlo después con from_pretrained("./modelo-final")
trainer.save_model("./hate-speech-model-final_en")
tokenizer.save_pretrained("./hate-speech-model-final_en")


# ============================================================
# 8. INFERENCIA (usar el modelo guardado)
# ============================================================
from transformers import pipeline

clasificador = pipeline(
    "text-classification",
    model="./hate-speech-model-final_en",
    tokenizer="./hate-speech-model-final_en",
)

print(clasificador("this is a test"))
# → [{'label': 'LABEL_1', 'score': 0.87}]  (LABEL_1 = odio si label=1)