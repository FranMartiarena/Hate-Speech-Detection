#Este es el primer modelo de deteccion de odio usando support vector machines.

#Por el Separating hyperplane theorem, si tenemos dos conjuntos convexos que son disjuntos, entonces existe un hiperplano que separa ambos conjuntos.
#La idea es encontrar el hiperplano con mas margen entre los 2 conjuntos. (Como sabemos si existe dicho hiperplano??)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from file_paths import YOU_TOXIC, HAT_EVAL_DEV, HAT_EVAL_TEST, HAT_EVAL_TRAIN, HAT_EVAL, HAT_EVAL_BAL, YOU_TOXIC_BAL
import matplotlib as mpl 
import matplotlib.cm as cm 
import matplotlib.pyplot as plt 
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import GridSearchCV
import re 
from sklearn.feature_extraction.text import CountVectorizer
import fasttext
import fasttext.util
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
import gensim.downloader as api
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()  # Para lematización

#hashtags: 0 para eliminarlos, 1 para dejarlos (luego se saca el #)
def limpiar_texto(texto, hashtags):
        # Sacamos hashtags
        if hashtags == 0: texto = re.sub(r'#\w+', '', texto)
        # Eliminamos caracteres especiales y numeros.
        texto = re.sub(r'[^a-zA-Z\s]', '', texto)
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(texto)
        texto = [lemmatizer.lemmatize(w) for w in word_tokens if not w.lower() in stop_words]
        return " ".join(texto)

def preprocess(archive, bool, opcion=1):
    
    ds = pd.read_csv(archive)
    if bool:
        if opcion == 1:
            ds = ds.rename(columns={'IsToxic': 'HS'})
        else:
            ds = ds.rename(columns={'IsHatespeech': 'HS'})
        ds = ds.rename(columns={'Text': 'text'})
        ds['HS'] = ds['HS'].astype(int)
    # Dividir el dataset en entrenamiento y prueba
    train_ds, test_ds = train_test_split(ds, test_size=0.2, random_state=41, stratify=ds['HS'])
    print(f"\n{train_ds['HS'].value_counts()}\n")
    print(f"\n{test_ds['HS'].value_counts()}\n")
    # Reiniciar los índices en ambos conjuntos
    train_ds = train_ds.reset_index(drop=True)
    test_ds = test_ds.reset_index(drop=True)
    train = train_ds
    test = test_ds
    svm_model_custom(train,test)

def svm_model_custom(train,test, htgs=1):

    #Tendremos que transformar el texto en una representación numérica adecuada para SVM.
    #Una técnica común es usar TF-IDF (Term Frequency-Inverse Document Frequency) para convertir el texto en vectores de características.

    # Separar las características (Text) y las etiquetas, ademas limpio los comentarios.
    x_train = train['text'].apply(limpiar_texto, hashtags=htgs)  
    y_train = train['HS']
    x_test = test['text'].apply(limpiar_texto, hashtags=htgs)  
    y_test = test['HS']  

    
    # Convertir el texto a representaciones numéricas usando TF-IDF
    #vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
    #x_train_tfidf = vectorizer.fit_transform(x_train)
    #x_test_tfidf = vectorizer.transform(x_test)
    
    # Convertir el texto a representaciones numéricas usando bag of words
    # Inicializar el CountVectorizer para Bag-of-Words
    #bow_vectorizer = CountVectorizer()
    #matriz dispersa (sparse matrix), donde cada fila representa un comentario y cada columna representa una palabra del vocabulario
    #con el valor de cada celda indicando la cantidad de veces que aparece esa palabra en ese comentario.
    #x_train_bow = bow_vectorizer.fit_transform(x_train)
    #x_test_bow = bow_vectorizer.transform(x_test)
    
    # Convertir el texto a representaciones numericas usando bag of characters
    # Configurar el CountVectorizer para Bag-of-Characters
    #ngram_vectorizer = CountVectorizer(ngram_range=(1, 3))
    #x_train_ngram = ngram_vectorizer.fit_transform(x_train)
    #x_test_ngram = ngram_vectorizer.transform(x_test)
    
    
    fasttext.util.download_model('en', if_exists='ignore')  # English
    ft = fasttext.load_model('cc.en.300.bin')

    # Función para obtener el embedding promedio de un comentario
    def obtener_embedding_promedio(comentario):
        palabras = comentario.split()
        embeddings = [ft.get_word_vector(palabra) for palabra in palabras]
        return np.sum(embeddings, axis=0) if embeddings else np.zeros((ft.get_dimension(),))
    

    x_train_ft = np.vstack(x_train.apply(obtener_embedding_promedio))
    x_test_ft = np.vstack(x_test.apply(obtener_embedding_promedio))
    

    
    # Crear el modelo SVM
    #svm_model_tfidf = SVC(kernel='linear')  # El kernel 'linear' es común en estos casos
    #svm_model_bow = SVC(kernel='linear')
    #svm_model_ngram = SVC(kernel='linear')
    svm_model_ft = SVC(kernel='linear')

    #svm_model_tfidf.fit(x_train_tfidf, y_train)
    #svm_model_bow.fit(x_train_bow, y_train)
    #svm_model_ngram.fit(x_train_ngram, y_train)
    svm_model_ft.fit(x_train_ft, y_train)

    # Predecir en el conjunto de prueba
    #y_pred = svm_model_tfidf.predict(x_test_tfidf)
    #y_pred = svm_model_bow.predict(x_test_bow)
    #y_pred = svm_model_ngram.predict(x_test_ngram)
    y_pred = svm_model_ft.predict(x_test_ft)

    # Calcular las métricas
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Evaluar el rendimiento
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    #print(classification_report(y_test, y_pred_tfidf))
    # Imprimir los resultados
    print("Precision:", precision)
    print("Recall:", recall)
    print("Macro-averaged F1-score:", f1)


    cnf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    plot_confusion_matrix(cnf_matrix, classes=['Not Hate','Hate'],normalize=True,
                        title='Confusion matrix with all features')
    plt.savefig('grafico.png')


#---------------------------------------------------
def cross_evaluation(archive_train,archive_test,bool, hashtags=0):   
    ds = pd.read_csv(archive_train)
    if bool:
        ds = ds.rename(columns={'IsToxic': 'HS'})
        ds = ds.rename(columns={'Text': 'text'})
        ds['HS'] = ds['HS'].astype(int)
    # Dividir el dataset en entrenamiento y prueba
    train_ds, test_ds = train_test_split(ds, test_size=0.2, random_state=41)

    # Reiniciar los índices en ambos conjuntos
    train_ds = train_ds.reset_index(drop=True)
    test_ds = test_ds.reset_index(drop=True)
    train = train_ds
    test = pd.read_csv(archive_test)
    if not bool:
        test = test.rename(columns={'Text': 'text'})
        test = test.rename(columns={'IsToxic': 'HS'})
        test['HS'] = test['HS'].astype(int)
    svm_model_custom(train,test)

#---------------------------------------------------
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',fontsize=15)
    plt.xlabel('Predicted label',fontsize=15)
#---------------------------------------------------



text = """ingrese un valor numerico para elegir el dataset para el entrenamiento del modelo:
1 youtoxic - toxicidad
2 youtoxic - hatespeech
3 youtoxic - hatespeech balanceado
4 hatEval
5 hatEval combinado
6 hatEval combinado balanceado
7 evaluacion cruzada train Hateval
8 evaluacion cruzada train YOUTOXIC
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
    preprocess(YOU_TOXIC,True, 1)
elif choice == 2:
    preprocess(YOU_TOXIC,True, 2) 
elif choice == 3:
    preprocess(YOU_TOXIC_BAL,True, 3) 
elif choice == 4:
    train = pd.read_csv(HAT_EVAL_TRAIN)
    test = pd.read_csv(HAT_EVAL_TEST)
    svm_model_custom(train,test)
elif choice == 5:
    preprocess(HAT_EVAL,False)
elif choice == 6:
    preprocess(HAT_EVAL_BAL,False)
elif choice == 7:
    cross_evaluation(HAT_EVAL,YOU_TOXIC,False)
elif choice == 8:
    cross_evaluation(YOU_TOXIC,HAT_EVAL,True)
else:
    print("Opción no válida.")
