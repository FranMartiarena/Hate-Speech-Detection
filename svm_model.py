#Este es el primer modelo de deteccion de odio usando support vector machines.

#Por el Separating hyperplane theorem, si tenemos dos conjuntos convexos que son disjuntos, entonces existe un hiperplano que separa ambos conjuntos.
#La idea es encontrar el hiperplano con mas margen entre los 2 conjuntos. (Como sabemos si existe dicho hiperplano??)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Cargar los datos de test hateval
train = pd.read_csv('data_set/hateval2019_en_train.csv')
test = pd.read_csv('data_set/hateval2019_en_test.csv')

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

# Crear el modelo SVM
svm_model = SVC(kernel='linear')  # El kernel 'linear' es común en estos casos

# Entrenar el modelo
svm_model.fit(x_train_tfidf, y_train)

# Predecir en el conjunto de prueba
y_pred = svm_model.predict(x_test_tfidf)

# Evaluar el rendimiento
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))



