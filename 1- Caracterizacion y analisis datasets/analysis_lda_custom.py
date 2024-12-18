import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import pyLDAvis.gensim
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import os
from file_paths import YOU_TOXIC, HAT_EVAL_DEV, HAT_EVAL_TEST, HAT_EVAL_TRAIN, HAT_EVAL

# Definir función para preprocesar el texto
def preprocess(text):
    result = []
    # Instanciar el lematizador
    lemmatizer = WordNetLemmatizer()
    for token in simple_preprocess(text):
        if token not in stopwords.words('english') and len(token) > 3:
            # Lematizar cada palabra
            result.append(lemmatizer.lemmatize(token))
    return result

def analysis_lda_custom(archive):

    # Leer archivo CSV
    data_set = pd.read_csv(archive)

    t = ''
    if archive == YOU_TOXIC:
        t = 'Text'
    else:
        t = 'text'

    text = data_set[t].values

    # Descargar stopwords y lematizador
    nltk.download('wordnet')
    nltk.download('stopwords')

    # Preprocesar todos los documentos
    processed_text = [preprocess(doc) for doc in text]

    # Crear el diccionario
    dictionary = Dictionary(processed_text)

    # Filtrar tokens muy raros o muy comunes
    dictionary.filter_extremes(no_below=10, no_above=0.5)

    # Convertir los documentos a la representación Bag of Words
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_text]

    # Entrenar el modelo LDA con 5 temas (puedes ajustar este número)
    lda_model = LdaModel(bow_corpus, num_topics=2, id2word=dictionary, passes=15)

    # Mostrar los temas generados por el modelo
    for idx, topic in lda_model.print_topics(-1):
        print(f'Topic {idx}: {topic}')

    # Preparar los datos para visualización
    lda_display = gensimvis.prepare(lda_model, bow_corpus, dictionary)

    # Visualizar
    # Preparar los datos para visualización
    lda_display = gensimvis.prepare(lda_model, bow_corpus, dictionary)

    # Guardar la visualización como archivo HTML
    # Crear la carpeta 'visualizaciones' si no existe
    output_dir = 'graficos_html'
    os.makedirs(output_dir, exist_ok=True)

    file = ''
    if archive == YOU_TOXIC:
        file = 'you_toxic_lda_visualization.html' 
    elif archive == HAT_EVAL_TRAIN:
        file = 'hateval_train_lda_visualization.html' 
    elif archive == HAT_EVAL_TEST:
        file = 'hateval_test_lda_visualization.html' 
    elif archive == HAT_EVAL_DEV:
        file = 'hateval_dev_lda_visualization.html'
    elif archive == HAT_EVAL:
        file = 'hateval_convinado_lda_visualization.html'  

    # Guardar la visualización en un archivo HTML en la carpeta especificada
    output_file = os.path.join(output_dir, file)
    pyLDAvis.save_html(lda_display, output_file)

    print("Visualización guardada en la carpeta graficos_lda . Ábrelo en un navegador para ver el gráfico.")


text = """ingrese un valor numerico para elegir el dataset para el analisis:
1 youtoxic
2 hatEval train 
3 hatEval test
4 hatEval dev
5 hatEval
"""

archive = ''
choice = input(text)

# Convierte la entrada a un entero
if choice.isdigit():  # Verifica si la entrada es un número
    choice = int(choice)  # Convierte a entero
else:
    print("Por favor, ingrese un número válido.")
    choice = None

# Ahora compara el valor de choice
if choice == 1:
    archive = YOU_TOXIC  
elif choice == 2:
    archive = HAT_EVAL_TRAIN  
elif choice == 3:
    archive = HAT_EVAL_TEST  
elif choice == 4:
    archive = HAT_EVAL_DEV  
elif choice == 5:
    archive = HAT_EVAL  
else:
    print("Opción no válida.")

# Llama a la función si archive tiene un valor válido
if archive:
    analysis_lda_custom(archive)