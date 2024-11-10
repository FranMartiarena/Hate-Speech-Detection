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
from sklearn.utils import resample

def eliminar_hashtags(texto):
        # Expresión regular que encuentra hashtags con o sin espacios entre ellos
        return re.sub(r'#\w+', '', texto)
    
def preprocesar_comentario(texto, hashtags):
    stop_words = set(stopwords.words('english'))  
    lemmatizer = WordNetLemmatizer()  # Para lematización
    # Convertir a minúsculas
    texto = texto.lower()
    
    if hashtags == 0:
        texto = eliminar_hashtags(texto)

    # Eliminar caracteres especiales y números
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)
    
    # Tokenizar
    palabras = word_tokenize(texto)
    
    # Eliminar stopwords

    palabras = [lemmatizer.lemmatize(palabra) for palabra in palabras if palabra not in stop_words]
    
    # Reconstruir el texto limpio
    return " ".join(palabras)


#0 para hateval, 1 para youtoxic
#hs: 0 para ver la frecuencia en comentarios sin odio, 1 para los coment con odio, sino general
def longitud_comentarios(bool, hs):
    # Leer archivo CSV
    if bool == 1:
        df = pd.read_csv('../data_set/youtoxic_english_1000.csv')
        col = "Text"
    else:
        df = pd.read_csv('../data_set/hateval2019_en_convinado.csv')
        col = "text"

    if hs == 0:
        comentarios_filtrados = df[df['IsHatespeech' if bool==1 else 'HS'] == 0][col]
    elif hs == 1:
        comentarios_filtrados = df[df['IsHatespeech' if bool==1 else 'HS'] == 1][col]
    else:
        comentarios_filtrados = df[col]


    # Calcular la longitud de cada comentario (en palabras o caracteres)
    # Aquí se calcula en palabras; 
    long_comentarios_filtrados = comentarios_filtrados.apply(lambda x: len(str(x).split()))

    # Visualización de la distribución de longitud de los comentarios
    plt.figure(figsize=(10, 6))
    sns.histplot(long_comentarios_filtrados, kde=True)
    plt.title(f'Distribución de la longitud de los comentarios de {"youtoxic" if bool ==1 else "hateval"}')
    plt.xlabel('Número de palabras')
    plt.ylabel('Frecuencia')
    #plt.xticks([0,20,40,60,80,100,200])  # Agrega 20 específicamente aquí
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
        print("Hateval combinado:")
        df = pd.read_csv('../data_set/hateval2019_en_convinado.csv')
        print(f"\n{df['HS'].value_counts()}\n")
        print("Hateval train:")
        df = pd.read_csv('../data_set/hateval2019_en_train.csv')
        print(f"\n{df['HS'].value_counts()}\n")
        print("Hateval test:")
        df = pd.read_csv('../data_set/hateval2019_en_test.csv')
        print(f"\n{df['HS'].value_counts()}\n")
        print("Hateval dev:")
        df = pd.read_csv('../data_set/hateval2019_en_dev.csv')
        print(f"\n{df['HS'].value_counts()}\n")

#bool: 0 para hateval, 1 para youtoxic
#hs: 0 para ver la frecuencia en comentarios sin odio, 1 para los coment con odio, sino general
#hashtags: 0 para eliminarlos, 1 para dejarlos 
def frecuencia_palabras(bool, hs, hashtags):

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
    
    if hs == 0:
        comentarios_filtrados = df[df['IsHatespeech' if bool==1 else 'HS'] == 0][col]
    elif hs == 1:
        comentarios_filtrados = df[df['IsHatespeech' if bool==1 else 'HS'] == 1][col]
    else:
        comentarios_filtrados = df[col]

    comentario_procesado = comentarios_filtrados.apply(lambda x: preprocesar_comentario(str(x), hashtags))

    # Concatenar todos los comentarios procesados en una sola cadena de texto
    todos_los_comentarios_procesados = " ".join(comentario_procesado)

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
    plt.savefig(f'{"youtoxic"if bool ==1 else "hateval"}_frec_palabras1.png')  
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

    comentarios_procesados = comentarios_filtrados.apply(preprocesar_comentario, hashtags=hashtags)

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
    plt.savefig(f'{"youtoxic"if bool ==1 else "hateval"}_frec_ngram_300.png')  
    # Mostrar el gráfico
    plt.show()

def balance_hateval():
    df = pd.read_csv('../data_set/hateval2019_en_convinado.csv')
    # Separar las clases
    df_majority = df[df['HS'] == 0]  # Clase mayoritaria
    df_minority = df[df['HS'] == 1]  # Clase minoritaria
    print(f"\n{df['HS'].value_counts()}\n")
    # Submuestrear la clase mayoritaria
    df_majority_downsampled = resample(df_majority,
                                    replace=False,
                                    n_samples=len(df_minority),
                                    random_state=42)

    # Combinar las dos clases
    df_balanced = pd.concat([df_majority_downsampled, df_minority])

    print(f"\n{df_balanced['HS'].value_counts()}\n")

    df_balanced.to_csv('hateval2019_en_convinado_balanceado.csv', index=False)

def balance_youtoxic():
    df = pd.read_csv('../data_set/youtoxic_english_1000.csv')
    # Separar las clases
    df_majority = df[df['IsHatespeech'] == 0]  # Clase mayoritaria
    df_minority = df[df['IsHatespeech'] == 1]  # Clase minoritaria
    print(f"\n{df['IsHatespeech'].value_counts()}\n")
    # Submuestrear la clase mayoritaria
    df_majority_downsampled = resample(df_majority,
                                    replace=False,
                                    n_samples=len(df_minority),
                                    random_state=42)

    # Combinar las dos clases
    df_balanced = pd.concat([df_majority_downsampled, df_minority])

    print(f"\n{df_balanced['IsHatespeech'].value_counts()}\n")

    df_balanced.to_csv('youtoxic_english_1000_balanceado.csv', index=False)


