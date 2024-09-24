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

# Leer archivo CSV
ds_youtoxic = pd.read_csv('data_set/youtoxic_english_1000.csv')

#print(ds_hatEval_train.head())

text_youtoxic = ds_youtoxic['Text'].values

# Descargar stopwords y lematizador
nltk.download('wordnet')
nltk.download('stopwords')

# Instanciar el lematizador
lemmatizer = WordNetLemmatizer()

# Definir función para preprocesar el texto
def preprocess(text):
    result = []
    for token in simple_preprocess(text):
        if token not in stopwords.words('english') and len(token) > 3:
            # Lematizar cada palabra
            result.append(lemmatizer.lemmatize(token))
    return result

# Preprocesar todos los documentos
processed_text = [preprocess(doc) for doc in text_youtoxic]

# Crear el diccionario
dictionary = Dictionary(processed_text)

# Filtrar tokens muy raros o muy comunes
dictionary.filter_extremes(no_below=10, no_above=0.5)

# Convertir los documentos a la representación Bag of Words
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_text]

# Entrenar el modelo LDA con 5 temas (puedes ajustar este número)
lda_model = LdaModel(bow_corpus, num_topics=5, id2word=dictionary, passes=15)

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
# Guardar la visualización en un archivo HTML en la carpeta especificada
output_file = os.path.join(output_dir, 'lda_visualization.html')
pyLDAvis.save_html(lda_display, output_file)

print("Visualización guardada en 'lda_visualization.html'. Ábrelo en un navegador para ver el gráfico.")
