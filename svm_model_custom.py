#Este es el primer modelo de deteccion de odio usando support vector machines.

#Por el Separating hyperplane theorem, si tenemos dos conjuntos convexos que son disjuntos, entonces existe un hiperplano que separa ambos conjuntos.
#La idea es encontrar el hiperplano con mas margen entre los 2 conjuntos. (Como sabemos si existe dicho hiperplano??)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from file_paths import YOU_TOXIC, HAT_EVAL_DEV, HAT_EVAL_TEST, HAT_EVAL_TRAIN, HAT_EVAL
import matplotlib as mpl 
import matplotlib.cm as cm 
import matplotlib.pyplot as plt 
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import GridSearchCV

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
    #svm_model = SVC(kernel='poly', degree=1) 
    #svm_model = SVC(kernel='rbf', gamma='scale') 
    #svm_model = SVC(kernel='sigmoid')
    # Entrenar el modelo

    # Definir los hiperparámetros que quieres probar
    #param_grid = {
    #    'C': [0.1, 1, 10],
    #    'kernel': ['linear', 'poly', 'rbf'],
    #    'degree': [1, 3, 5],
    #    'gamma': ['scale', 'auto']
    #}

    # Crear un modelo SVM
    #svm_model = SVC()

    # Hacer una búsqueda de grilla
    #grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy')
    #grid_search.fit(x_train_tfidf, y_train)

    # Los mejores parámetros encontrados
    #print(grid_search.best_params_)

    svm_model.fit(x_train_tfidf, y_train)

    # Predecir en el conjunto de prueba
    y_pred = svm_model.predict(x_test_tfidf)

    # Evaluar el rendimiento
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

    cnf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    plot_confusion_matrix(cnf_matrix, classes=['Not Hate','Hate'],normalize=True,
                        title='Confusion matrix with all features')
    plt.savefig('grafico.png')
    return svm_model




#---------------------------------------------------
def cross_evaluation(archive_train,archive_test,bool):   
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
1 youtoxic
2 hatEval
3 hatEval convinado
4 evaluacion cruzada train Hateval
5 evaluacion cruzada train YOUTOXIC
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
elif choice == 4:
    cross_evaluation(HAT_EVAL,YOU_TOXIC,False)
elif choice == 5:
    cross_evaluation(YOU_TOXIC,HAT_EVAL,True)
else:
    print("Opción no válida.")
