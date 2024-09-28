#Este es el primer modelo de deteccion de odio usando support vector machines.

#Por el Separating hyperplane theorem, si tenemos dos conjuntos convexos que son disjuntos, entonces existe un hiperplano que separa ambos conjuntos.
#La idea es encontrar el hiperplano con mas margen entre los 2 conjuntos. (Como sabemos si existe dicho hiperplano??)

import pandas as pd

# Cargar los datos
df = pd.read_csv('youtoxic_english.csv')
