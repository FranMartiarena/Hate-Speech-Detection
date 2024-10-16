#En este archivo entreno los modelos de deteccion de un tema especifico de odio y lo aplico a un tema del otro dataset.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Cargar hateval con topicos y ytxic(que lo uso entero porq tiene un solo tipo de hate)

hateval = pd.read_csv('hateval_topics.csv')
hateval = hateval[~((hateval['HS'] == 1) & (hateval['topic_distribution'] == "Topic_0"))]#Le saco los comentarios con topic_0, que son contra mujeres, que ademas tengan HS.
youtoxic = pd.read_csv('data_set/youtoxic_english_1000.csv')

#Divido hateval en train y test 20, 80
train, test = train_test_split(hateval, test_size=0.2, random_state=23)

#Tendremos que transformar el texto en una representación numérica adecuada para SVM.
#Una técnica común es usar TF-IDF (Term Frequency-Inverse Document Frequency) para convertir el texto en vectores de características.

# Separar las características (Text) y las etiquetas
x_train = train['text']
y_train = train['HS']

x_test = test['text']
y_test = test['HS']

# Convertir el texto a representaciones numéricas usando TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # puedes ajustar 'max_features'
x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)

#Es importante queuse el mismo vectorizador
x_test_yt = vectorizer.transform(youtoxic["Text"])

# Crear el modelo SVM
svm_model = SVC(kernel='linear')  # El kernel 'linear' es común en estos casos

# Entrenar el modelo
svm_model.fit(x_train_tfidf, y_train)

# Predecir en el conjunto de prueba
y_pred = svm_model.predict(x_test_tfidf)

# Evaluar el rendimiento en el conjunto de prueba
print(f"Accuracy sobre conjunto test: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Predecir en ytoxic
y_pred_yt = svm_model.predict(x_test_yt)

# Evaluar el rendimiento en yotoxic
print(f"Accuracy sobre conjunto youtoxic: {accuracy_score(youtoxic['IsHatespeech'], y_pred_yt)}")
print(classification_report(youtoxic['IsHatespeech'], y_pred_yt))