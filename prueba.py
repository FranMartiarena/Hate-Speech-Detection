import pandas as pd
import os
from file_paths import YOU_TOXIC, HAT_EVAL_DEV, HAT_EVAL_TEST, HAT_EVAL_TRAIN
# Lista de archivos CSV a unir
archivos_csv = [HAT_EVAL_DEV, HAT_EVAL_TEST, HAT_EVAL_TRAIN]

# Leer y concatenar todos los archivos
df_combinado = pd.concat([pd.read_csv(archivo) for archivo in archivos_csv])

# Especificar la carpeta donde se guardará el archivo combinado
carpeta_destino = 'data_set/'  # Cambia esta ruta a la carpeta deseada

# Asegurarse de que la carpeta exista, si no, crearla
if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)

# Guardar el archivo combinado en la carpeta especificada
ruta_archivo = os.path.join(carpeta_destino, 'hateval2019_en_convinado.csv')
df_combinado.to_csv(ruta_archivo, index=False)

print(f"Archivos combinados y guardados correctamente en {ruta_archivo}.")
