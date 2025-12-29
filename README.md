# Detección de odio en texto generado por usuarios

El discurso de odio puede definirse como discurso que expresa o incita a dañar a un individuo o a un grupo de personas en función de una o más de sus características personales, como el género, la raza, la religión, la sexualidad, etc.
En este proyecto exploramos varias estrategias de deteccion de discurso de odio en redes, viendolo como un problema de clasificacion sobre conjuntos de datos ya existentes.

## ¿Porque detectar discurso de odio?

Creemos que la deteccion del discurso de odio ayuda a mitigar las siguientes problematicas generadas por el discurso de odio en redes:

- **Impacto Social Negativo**: Se normaliza la violencia simbolica virtual, que puede escalar a violencia real. Tiene efectos medibles en salud mental, autocensura y exclusion en un espacio publico. Afecta desproporcionadamente a minorías
- **Escala del problema**: Hay millones de mensajes de odio por dia, lo que es dificil de moderar por humanos a mano.
- **Sesgos**: El discurso de odio en redes puede sesgar a las personas cambiando lo que parece aceptable.

## Algunas Ideas

(Explicar aplicaciones para resolver problematicas)

---
# Que falta hacer?

- Analisis de deteccion usando LLM's abiertos
- Creacion ó busqueda de datasets mas abarcativos
- 

---
## Creación entorno virtual

```
$ virtualenv venv
$ source ./venv/bin/activate
```

## Instalación packages

```
$ pip install -r requirements.txt
```

## Ejecucion del analisis lda

El Analisis LDA se encuentra en el directorio de caracterizacion y analisis de datasets.
Al ejecutar el archivo analysis_lda_custom.py este le pedira elegir alguno de los dataset disponibles para realizar el analisis. 
Luego, para una muestra de los resultados, se guarda en la carpeta grafico_html un archivo html el cual se puede abrir en el navegador para ver el grafico.

 
