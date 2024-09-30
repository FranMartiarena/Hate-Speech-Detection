#Este es el primer modelo de deteccion de odio usando support vector machines.

#Por el Separating hyperplane theorem, si tenemos dos conjuntos convexos que son disjuntos, entonces existe un hiperplano que separa ambos conjuntos.
#La idea es encontrar el hiperplano con mas margen entre los 2 conjuntos. (Como sabemos si existe dicho hiperplano??)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from file_paths import YOU_TOXIC, HAT_EVAL_DEV, HAT_EVAL_TEST, HAT_EVAL_TRAIN, HAT_EVAL


def preprocess(archive,bool):

    ds = pd.read_csv(archive)
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
    test = test_ds
    svm_model_custom(train,test)


def svm_model_custom(train,test):

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



#---------------------------------------------------

text = """ingrese un valor numerico para elegir el dataset para el entrenamiento del modelo:
1 youtoxic
2 hatEval
3 hatEval convinado
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
    preprocess(YOU_TOXIC,True) 
elif choice == 2:
    train = pd.read_csv(HAT_EVAL_TRAIN)
    test = pd.read_csv(HAT_EVAL_TEST)
    svm_model_custom(train,test)
elif choice == 3:
    preprocess(HAT_EVAL,False) 
else:
    print("Opción no válida.")
