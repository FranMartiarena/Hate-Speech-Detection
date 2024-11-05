import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
from nltk import ngrams

#0 para hateval, 1 para youtoxic
def longitud_comentarios(bool):
    # Leer archivo CSV
    if bool == 1:
        df = pd.read_csv('../data_set/youtoxic_english_1000.csv')
        col = "Text"
    else:
        df = pd.read_csv('../data_set/hateval2019_en_convinado.csv')
        col = "text"
    # Calcular la longitud de cada comentario (en palabras o caracteres)
    # Aquí se calcula en palabras; si prefieres caracteres, usa: df['longitud'] = df['comentario'].str.len()
    df['longitud'] = df[col].apply(lambda x: len(str(x).split()))

    # Visualización de la distribución de longitud de los comentarios
    plt.figure(figsize=(10, 6))
    sns.histplot(df['longitud'],  kde=True)
    plt.title(f'Distribución de la longitud de los comentarios de {"youtoxic" if bool ==1 else "hateval"}')
    plt.xlabel('Número de palabras')
    plt.ylabel('Frecuencia')
    plt.xlim(0, 200 if bool ==1 else 100)
    plt.savefig(f'{"youtoxic"if bool ==1 else "hateval"}_long_promedio.png')  
    plt.show()

#0 para hateval, 1 para youtoxic
def balanceo_clases(bool):
    # Leer archivo CSV
    if bool == 1:
        df = pd.read_csv('../data_set/youtoxic_english_1000.csv')
        print(f"\n{df['IsHatespeech'].value_counts()}\n")
        print(f"\n{df['IsToxic'].value_counts()}\n")
    else:
        df = pd.read_csv('../data_set/hateval2019_en_convinado.csv')
        print(f"\n{df['HS'].value_counts()}\n")

#0 para hateval, 1 para youtoxic    
def frecuencia_palabras(bool):

    # Descargar recursos necesarios de NLTK si no lo has hecho antes
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('punkt_tab')

    if bool == 1:
        df = pd.read_csv('../data_set/youtoxic_english_1000.csv')
        col = "Text"
    else:
        df = pd.read_csv('../data_set/hateval2019_en_convinado.csv')
        col = "text"

    # Inicializar herramientas de procesamiento de NLTK
    stop_words = set(stopwords.words('english'))  # Usa 'spanish' para stopwords en español
    lemmatizer = WordNetLemmatizer()  # Para lematización

    
    def preprocesar_comentario(texto):
        # Convertir a minúsculas
        texto = texto.lower()
        
        # Eliminar caracteres especiales y números
        texto = re.sub(r'[^a-zA-Z\s]', '', texto)
        
        # Tokenizar
        palabras = word_tokenize(texto)
        
        # Eliminar stopwords y aplicar lematización

        palabras = [lemmatizer.lemmatize(palabra) for palabra in palabras if palabra not in stop_words]
        
        # Reconstruir el texto limpio
        return " ".join(palabras)
    
    df['comentario_procesado'] = df[col].apply(lambda x: preprocesar_comentario(str(x)))
    # Concatenar todos los comentarios procesados en una sola cadena de texto
    todos_los_comentarios_procesados = " ".join(df['comentario_procesado'])

    # Dividir en palabras para contar su frecuencia
    palabras_procesadas = todos_los_comentarios_procesados.split()  # La tokenización ya se hizo en el preprocesamiento

    # Contar la frecuencia de cada palabra
    frecuencia_palabras_procesadas = Counter(palabras_procesadas)

    # Ver las 10 palabras más comunes
    print("\nLas 10 palabras mas comunes:\n")
    for par in frecuencia_palabras_procesadas.most_common(10):
        print(f"Palabra: {par[0]}, Frecuencia: {par[1]}")

    palabras = [item[0] for item in frecuencia_palabras_procesadas.most_common(10)]
    valores = [item[1] for item in frecuencia_palabras_procesadas.most_common(10)]

    # Crear el gráfico de barras
    plt.figure(figsize=(10, 6))  # Tamaño opcional del gráfico
    plt.bar(palabras, valores, color='skyblue')

    # Añadir etiquetas y título
    plt.xlabel('Palabras')
    plt.ylabel('Frecuencia')
    plt.title(f'Frecuencia de palabras en {"youtoxic" if bool ==1 else "hateval"}')
    plt.savefig(f'{"youtoxic"if bool ==1 else "hateval"}_frec_palabras.png')  
    # Mostrar el gráfico
    plt.show()

#bool: 0 para hateval y 1 para youtoxic
#n: para n-grama
#hs: 0 para ver n-gramas de coments sin odio, 1 para ver ngramas de coments con odio, y 2 para general
#hashtags: 0 para eliminarlos, 1 para dejarlos  
def n_grama(bool, n, hs, hashtags):

    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    
    if bool == 1:
        df = pd.read_csv('../data_set/youtoxic_english_1000.csv')
        col = "Text"
    else:
        df = pd.read_csv('../data_set/hateval2019_en_convinado.csv')
        col = "text"

    # Inicializar herramientas de procesamiento de NLTK
    stop_words = set(stopwords.words('english'))  # Usa 'spanish' para stopwords en español
    
    def eliminar_hashtags(texto):
        # Expresión regular que encuentra hashtags con o sin espacios entre ellos
        return re.sub(r'#\w+', '', texto)
    
    def preprocesar_comentario(texto):
        # Convertir a minúsculas
        texto = texto.lower()
        
        if hashtags == 0:
            texto = eliminar_hashtags(texto)

        # Eliminar caracteres especiales y números
        texto = re.sub(r'[^a-zA-Z\s]', '', texto)
        
        # Tokenizar
        palabras = word_tokenize(texto)
        
        # Eliminar stopwords

        palabras = [palabra for palabra in palabras if palabra not in stop_words]
        
        # Reconstruir el texto limpio
        return " ".join(palabras)
    
    # Función para generar n-gramas
    def generar_ngrams(comentarios, n):
        n_gramas = []
        for comentario in comentarios:
            palabras = comentario.split()
            n_gramas.extend(ngrams(palabras, n))
        return n_gramas

    if hs == 0:
        comentarios_filtrados = df[df['IsHatespeech' if bool==1 else 'HS'] == 0][col]
    elif hs == 1:
        comentarios_filtrados = df[df['IsHatespeech' if bool==1 else 'HS'] == 1][col]
    else:
        comentarios_filtrados = df[col]

    comentarios_procesados = comentarios_filtrados.apply(preprocesar_comentario)

    ngram = generar_ngrams(comentarios_procesados, n)
    
    # Contar frecuencia de bigramas y trigramas
    frecuencia = Counter(ngram)

    # Ver los 10 n-gramas más comunes
    print("\nLos 10 n-gramas más comunes:\n")
    for par in frecuencia.most_common(10):
        print(f"N-grama: {par[0]}, Frecuencia: {par[1]}")

    palabras = [" ".join(item[0]) for item in frecuencia.most_common(10)]
    valores = [item[1] for item in frecuencia.most_common(10)]

    # Crear el gráfico de barras
    plt.figure(figsize=(12, 8))  # Tamaño opcional del gráfico
    plt.bar(palabras, valores, color='skyblue')

    # Añadir etiquetas y título
    plt.xlabel(f'{n}-grama')
    plt.ylabel('Frecuencia')
    plt.title(f'Frecuencia de {n}-gramas en {"youtoxic" if bool ==1 else "hateval"}')
    plt.xticks(rotation=30)  # Rotar las etiquetas 45 grados
    plt.savefig(f'{"youtoxic"if bool ==1 else "hateval"}_frec_ngram_210.png')  
    # Mostrar el gráfico
    plt.show()


