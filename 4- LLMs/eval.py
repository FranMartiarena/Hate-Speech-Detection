import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ============================================================
# 1. CARGAR MODELO Y TOKENIZADOR
# ============================================================
MODEL_PATH = "./hate-speech-model-final_en"  # la carpeta donde guardaste el modelo

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# ============================================================
# 2. CARGAR Y TOKENIZAR EL TEST SET
# ============================================================

#data = pd.read_csv("../data_set/hateval2019_en_convinado_balanceado.csv") 
data = pd.read_csv("../data_set/youtoxic_english_1000_balanceado.csv") 
data = data.rename(columns={'IsHatespeech': 'labels'})
data['labels'] = data['labels'].astype(int) 
# Convertir a formato Dataset de HuggingFace
# (el Trainer solo acepta este formato, no DataFrames de pandas)
dataset_test = Dataset.from_pandas(data[["Text", "labels"]])


def tokenize(batch):
    return tokenizer(batch["Text"], truncation=True, padding="max_length", max_length=128)

dataset_test = dataset_test.map(tokenize, batched=True)

# ============================================================
# 3. TRAINER MÍNIMO SOLO PARA PREDECIR
# ============================================================
# No necesitás TrainingArguments completos, solo lo mínimo
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="./tmp", per_device_eval_batch_size=32),
)

# ============================================================
# 4. PREDICCIONES Y REPORTE
# ============================================================
predictions = trainer.predict(dataset_test)
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

print(classification_report(labels, preds, target_names=["no odio", "odio"]))
print(confusion_matrix(labels, preds))