import pandas as pd
import os
from file_paths import YOU_TOXIC, HAT_EVAL_DEV, HAT_EVAL_TEST, HAT_EVAL_TRAIN, HAT_EVAL
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import nltk


# Lista de archivos CSV a unir

def convinacion_hateval():
    archivos_csv = [HAT_EVAL_DEV, HAT_EVAL_TEST, HAT_EVAL_TRAIN]

    # Leer y concatenar todos los archivos
    df_combinado = pd.concat([pd.read_csv(archivo) for archivo in archivos_csv])

    # Especificar la carpeta donde se guardará el archivo combinado
    carpeta_destino = 'data_set/'  # Cambia esta ruta a la carpeta deseada

    # Asegurarse de que la carpeta exista, si no, crearla
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)

    # Guardar el archivo combinado en la carpeta especificada
    ruta_archivo = os.path.join(carpeta_destino, 'hateval2019_en_convinado.csv')
    df_combinado.to_csv(ruta_archivo, index=False)

    print(f"Archivos combinados y guardados correctamente en {ruta_archivo}.")

#-----------------------------------------------------------------------------------------------
import shutil

def topic_dominante():
    nltk_data_path = '/home/andres-sadir/nltk_data'
    shutil.rmtree(nltk_data_path)  # Borra los datos de NLTK
    # Descargar el recurso necesario
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')


    # Cargar el archivo CSV
    df = pd.read_csv(HAT_EVAL)

    # Preprocesar texto (tokenización, remoción de stopwords, etc.)
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        text = re.sub(r'\W', ' ', text)  # Remover caracteres especiales
        text = text.lower()  # Convertir a minúsculas
        tokens = word_tokenize(text)  # Tokenizar
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]  # Remover stopwords
        return tokens

    df['tokens'] = df['text'].apply(preprocess_text)

    # Crear diccionario y corpus para LDA
    documents = df['tokens'].tolist()
    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(text) for text in documents]

    # Aplicar LDA para identificar temas
    lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)

    # Crear una función para asignar el tema predominante a cada comentario
    def assign_dominant_topic(lda_model, corpus):
        dominant_topics = []
        for bow in corpus:
            topics = lda_model.get_document_topics(bow)
            dominant_topic = max(topics, key=lambda x: x[1])[0]  # Elegir el tema con mayor probabilidad
            dominant_topics.append(dominant_topic)
        return dominant_topics

    # Asignar el tema dominante a cada comentario
    df['dominant_topic'] = assign_dominant_topic(lda_model, corpus)

    # Guardar el CSV con la nueva columna de temas
    df.to_csv('archivo_con_temas.csv', index=False)

    print("Archivo guardado con las etiquetas de temas.")

#-----------------------------------------------------------------------------------------------


# Cargar el archivo CSV
df = pd.read_csv(HAT_EVAL)

# Preprocesar texto (tokenización, remoción de stopwords, etc.)
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remover caracteres especiales
    text = text.lower()  # Convertir a minúsculas
    tokens = word_tokenize(text)  # Tokenizar
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]  # Remover stopwords
    return tokens

df['tokens'] = df['text'].apply(preprocess_text)

# Crear diccionario y corpus para LDA
documents = df['tokens'].tolist()
dictionary = corpora.Dictionary(documents)
corpus = [dictionary.doc2bow(text) for text in documents]

# Aplicar LDA para identificar temas
lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

# Crear una función para asignar todos los tópicos con sus respectivas probabilidades
def assign_topic_distribution(lda_model, corpus, threshold=0.2):
    topic_distributions = []
    for bow in corpus:
        topics = lda_model.get_document_topics(bow)
        topic_distribution = {f"Topic_{topic[0]}": topic[1] for topic in topics if topic[1] > threshold}  # Filtrar por umbral
        topic_distributions.append(topic_distribution)
    return topic_distributions

# Asignar las probabilidades de todos los tópicos a cada comentario
df['topic_distribution'] = assign_topic_distribution(lda_model, corpus)

# Guardar el CSV con la nueva columna de distribución de tópicos
df.to_csv('archivo_con_distribucion_de_topicos.csv', index=False)

print("Archivo guardado con las distribuciones de tópicos.")
