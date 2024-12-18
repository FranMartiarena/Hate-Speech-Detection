#En este archivo entreno los modelos de deteccion de un tema especifico de odio y lo aplico a un tema del otro dataset.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt 
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix 
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re 

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


#Si true entonces entreno con hateval en coments de racismo y lo pruebo en ytxic
#Si false entonces entreno con youtoxic y lo pruebo en los comentarios racistas de hateval
def topico(entrenar_hateval):
    # Cargar hateval con topicos y ytxic(que lo uso entero porq tiene un solo tipo de hate)
    hateval = pd.read_csv('../data_set/hateval_topics.csv')
    youtoxic = pd.read_csv('../data_set/data_set/youtoxic_english_1000.csv')

    ngram_vectorizer = CountVectorizer(ngram_range=(1, 3))

    if entrenar_hateval:
        hateval = hateval[~((hateval['HS'] == 1) & (hateval['topic_distribution'] == "Topic_0"))]#Le saco los comentarios con topic_0 (contra mujeres) que tengan HS.
        x_train = hateval['text'].apply(limpiar_texto, hashtags=1)
        y_train = hateval['HS']
    else:
        hateval = hateval[(hateval['topic_distribution'] != "Topic_0")]#Le saco todos comentarios con topic_0
        x_train = youtoxic['Text'].apply(limpiar_texto, hashtags=1)  
        y_train = youtoxic['IsHatespeech']  

    x_train = ngram_vectorizer.fit_transform(x_train)
    
    if entrenar_hateval:
        x_test_topic = ngram_vectorizer.transform(youtoxic["Text"].apply(limpiar_texto, hashtags=1))
        y_test_topic = youtoxic['IsHatespeech']
    else:
        x_test_topic = ngram_vectorizer.transform(hateval["text"].apply(limpiar_texto, hashtags=1))
        y_test_topic = youtoxic['HS']
        
    # Crear el modelo SVM
    svm_model = SVC(kernel='linear')  # El kernel 'linear' es común en estos casos

    # Entrenar el modelo
    svm_model.fit(x_train, y_train)

    # Predecir en el dataset opuesto
    y_pred_topic = svm_model.predict(x_test_topic)

    print(f"Accuracy sobre conjunto opuesto: {accuracy_score(y_test_topic, y_pred_topic)}")
    print(f"precision sobre conjunto opuesto: {precision_score(y_test_topic, y_pred_topic)}")
    print(f"recall sobre conjunto opuesto: {recall_score(y_test_topic, y_pred_topic)}")
    print(f"f1_score sobre conjunto opuesto: {f1_score(y_test_topic, y_pred_topic, average='macro')}")

    cnf_matrix = confusion_matrix(y_test_topic, y_pred_topic)
    plt.figure(figsize=(8,6))
    plot_confusion_matrix(cnf_matrix, classes=['Not Hate','Hate'],normalize=True,
                        title='Confusion matrix with all features')
    plt.savefig('grafico.png')

def voto_mayoritario_hateval():

    hateval = pd.read_csv('../data_set/hateval_topics.csv')

    train, test = train_test_split(hateval, test_size=0.2, random_state=41, stratify=hateval['HS'])

    y_test = test['HS']

    #Modelo Inmigrantes
    hateval_inmigrantes = train[~((train['HS'] == 1) & (train['topic_distribution'] == "Topic_0"))]#Le saco los comentarios con topic_0 (contra mujeres) que tengan HS.
    ngram_vectorizer_inmigrantes = CountVectorizer(ngram_range=(1, 3))
    x_train_inmigrantes = ngram_vectorizer_inmigrantes.fit_transform(hateval_inmigrantes['text'].apply(limpiar_texto, hashtags=1))
    y_train_inmigrantes = hateval_inmigrantes['HS']    
    x_test_inmigrantes = ngram_vectorizer_inmigrantes.transform(test['text'].apply(limpiar_texto, hashtags=1))

    svm_model_ngram_inmigrantes = SVC(kernel='linear')
    svm_model_ngram_inmigrantes.fit(x_train_inmigrantes, y_train_inmigrantes)

    y_pred_inmigrantes = svm_model_ngram_inmigrantes.predict(x_test_inmigrantes)

    #Modelo Mujeres
    hateval_mujeres = train[~((train['HS'] == 1) & (train['topic_distribution'] == "Topic_1"))]#Le saco los comentarios con topic_1 (contra inmigrantes) que tengan HS.
    ngram_vectorizer_mujeres = CountVectorizer(ngram_range=(1, 3))
    x_train_mujeres = ngram_vectorizer_mujeres.fit_transform(hateval_mujeres['text'].apply(limpiar_texto, hashtags=1))
    y_train_mujeres = hateval_mujeres['HS']  
    x_test_mujeres = ngram_vectorizer_mujeres.transform(test['text'].apply(limpiar_texto, hashtags=1))
    
    svm_model_ngram_mujeres = SVC(kernel='linear')
    svm_model_ngram_mujeres.fit(x_train_mujeres, y_train_mujeres)

    y_pred_mujeres = svm_model_ngram_mujeres.predict(x_test_mujeres)

    #Modelo General
    ngram_vectorizer = CountVectorizer(ngram_range=(1, 3))
    x_train = ngram_vectorizer.fit_transform(train['text'].apply(limpiar_texto, hashtags=1))
    y_train = train['HS']
    x_test = ngram_vectorizer.transform(test['text'].apply(limpiar_texto, hashtags=1))

    svm_model_ngram = SVC(kernel='linear')
    svm_model_ngram.fit(x_train, y_train)

    y_pred = svm_model_ngram.predict(x_test)

    #Ahora junto las predicciones y hago voto mayoritario 
    assert len(y_pred_mujeres) == len(y_pred_inmigrantes)
    y_pred_final = np.where((y_pred_mujeres == 1) | (y_pred_inmigrantes == 1), 1, 0)


    #Analisis modelo voto mayoritario
    print(f"Accuracy voto mayoritario: {accuracy_score(y_test, y_pred_final)}")
    print("Precision voto mayoritario:", precision_score(y_test, y_pred_final))
    print("Recall voto mayoritario:", recall_score(y_test, y_pred_final))
    print("Macro-averaged F1-score voto mayoritario:", f1_score(y_test, y_pred_final, average='macro'))

    cnf_matrix = confusion_matrix(y_test, y_pred_final)
    plt.figure(figsize=(8,6))
    plot_confusion_matrix(cnf_matrix, classes=['Not Hate','Hate'],normalize=True,
                        title='Confusion matrix with all features')
    plt.savefig('matriz_voto_mayoritario.png')

    #Analisis modelo general
    print(f"Accuracy modelo general: {accuracy_score(y_test, y_pred)}")
    print("Precision modelo general:", precision_score(y_test, y_pred))
    print("Recall modelo general:", recall_score(y_test, y_pred))
    print("Macro-averaged F1-score modelo general:", f1_score(y_test, y_pred, average='macro'))

    cnf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    plot_confusion_matrix(cnf_matrix, classes=['Not Hate','Hate'],normalize=True,
                        title='Confusion matrix with all features')
    plt.savefig('matriz_modelo_general.png')

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

text = """
ingrese un valor numerico para elegir el dataset sobre el que se entrenara el modelo de un topico:
1 hateval
2 youtoxic
3 voto mayoritario vs modelo general
"""

choice = input(text)

# Convierte la entrada a un entero
if choice.isdigit():  # Verifica si la entrada es un número
    choice = int(choice)  # Convierte a entero
else:
    print("Por favor, ingrese un número válido.")
    choice = None

# Ahora compara el valor de choice
if choice == 1:
    topico(True) 
elif choice == 2:
    topico(False)
elif choice == 3:
    voto_mayoritario_hateval()
else:
    print("opcion no valida")
