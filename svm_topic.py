#En este archivo entreno los modelos de deteccion de un tema especifico de odio y lo aplico a un tema del otro dataset.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

#Si true entonces entreno con hateval en coments de racismo y lo pruebo en ytxic
#Si false entonces entreno con youtoxic y lo pruebo en los comentarios racistas de hateval
def topico(entrenar_hateval):
    # Cargar hateval con topicos y ytxic(que lo uso entero porq tiene un solo tipo de hate)
    hateval = pd.read_csv('hateval_topics.csv')
    youtoxic = pd.read_csv('data_set/youtoxic_english_1000.csv')

    if entrenar_hateval:
        hateval = hateval[~((hateval['HS'] == 1) & (hateval['topic_distribution'] == "Topic_0"))]#Le saco los comentarios con topic_0 (contra mujeres) que tengan HS.
        #Divido hateval en train y test 20, 80
        train, test = train_test_split(hateval, test_size=0.2, random_state=23)
        # Separar las características (Text) y las etiquetas
        x_train = train['text']
        y_train = train['HS']

        x_test = test['text']
        y_test = test['HS']

    else:
        hateval = hateval[(hateval['topic_distribution'] != "Topic_0")]#Le saco todos comentarios con topic_0
        #Divido youtoxic en train y test 20, 80
        train, test = train_test_split(youtoxic, test_size=0.2, random_state=23)
        # Separar las características (Text) y las etiquetas
        x_train = train['Text']
        y_train = train['IsHatespeech']

        x_test = test['Text']
        y_test = test['IsHatespeech']
        

    # Convertir el texto a representaciones numéricas usando TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # puedes ajustar 'max_features'
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    if entrenar_hateval:
        #Es importante queuse el mismo vectorizador
        x_test_topic = vectorizer.transform(youtoxic["Text"])
    else:
        x_test_topic = vectorizer.transform(hateval["text"])
        
    # Crear el modelo SVM
    svm_model = SVC(kernel='linear')  # El kernel 'linear' es común en estos casos

    # Entrenar el modelo
    svm_model.fit(x_train_tfidf, y_train)

    # Predecir en el conjunto de prueba
    y_pred = svm_model.predict(x_test_tfidf)

    # Evaluar el rendimiento en el conjunto de prueba
    print(f"Accuracy sobre conjunto test: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

    # Predecir en el dataset opuesto
    y_pred_topic = svm_model.predict(x_test_topic)

    if entrenar_hateval:
        # Evaluar el rendimiento en yotoxic
        print(f"Accuracy sobre conjunto youtoxic: {accuracy_score(youtoxic['IsHatespeech'], y_pred_topic)}")
        print(classification_report(youtoxic['IsHatespeech'], y_pred_topic))
    else:
        # Evaluar el rendimiento en hateval
        print(f"Accuracy sobre conjunto hateval: {accuracy_score(hateval['HS'], y_pred_topic)}")
        print(classification_report(hateval['HS'], y_pred_topic))

    
text = """
ingrese un valor numerico para elegir el dataset sobre el que se entrenara el modelo de un topico:
1 hateval
2 youtoxic

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
else:
    print("opcion no valida")
