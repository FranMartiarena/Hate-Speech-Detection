from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from scipy.special import softmax
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import csv
import urllib.request
import torch  # Asegúrate de importar torch
from file_paths import YOU_TOXIC, HAT_EVAL, HAT_EVAL_TEST, HAT_EVAL_DEV, HAT_EVAL_TRAIN  # Asegúrate de que esta ruta sea correcta

# Función para preprocesar el texto
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Definir la tarea y el modelo
task = 'hate'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

# Cargar el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Cargar el mapeo de etiquetas
labels = []
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]

# Cargar el dataset
df = pd.read_csv(HAT_EVAL)  
texts = df['text'].tolist()         
true_labels = df['HS'].tolist()   


true_labels = [str(label) for label in true_labels]

# Definir el tamaño máximo para la entrada
max_length = 512  

# Clasificar cada texto y almacenar predicciones
predicted_labels = []
for text in texts:
    # Preprocesar el texto
    preprocessed_text = preprocess(text)

    # Tokenizar y clasificar el texto
    encoded_input = tokenizer(preprocessed_text, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    
    with torch.no_grad():  
        output = model(**encoded_input)
    
    scores = output.logits.detach().numpy()[0]  
    scores = softmax(scores)

    # Obtener el índice de la etiqueta con mayor puntaje
    ranking = np.argmax(scores)
    predicted_labels.append(labels[ranking])  # Guardar la etiqueta predicha

# Mapear las etiquetas verdaderas a las etiquetas predichas
mapped_true_labels = ['hate' if label == '1' else 'not-hate' for label in true_labels]

# Asegúrate de que las etiquetas verdaderas y predichas están dentro del conjunto de etiquetas
unique_true_labels = set(mapped_true_labels)  # Cambiado para usar las etiquetas mapeadas
unique_predicted_labels = set(predicted_labels)


# Calcular métricas de evaluación
accuracy = accuracy_score(mapped_true_labels, predicted_labels)  
report = classification_report(
    mapped_true_labels, 
    predicted_labels, 
    target_names=['hate', 'not-hate'], 
    labels=['hate', 'not-hate'], 
    zero_division=0 
)
conf_matrix = confusion_matrix(mapped_true_labels, predicted_labels)  

# Mostrar resultados
print(f"Exactitud del modelo: {accuracy:.4f}")
print("Reporte de clasificación:")
print(report)
print("Matriz de confusión:")
print(conf_matrix)
