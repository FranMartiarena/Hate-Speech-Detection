# Detección de odio en texto generado por usuarios

El discurso de odio puede definirse como discurso que expresa o incita a dañar a un individuo o a un grupo de personas en función de una o más de sus características personales, como el género, la raza, la religión, la sexualidad, etc.

En este proyecto exploramos varias estrategias de deteccion de discurso de odio en redes, viendolo como un problema de clasificacion y usando conjuntos de datos ya existentes.


## ¿Porque detectar discurso de odio?

La deteccion del discurso de odio ayuda a mitigar las siguientes problematicas (Entre muchas otras):

- **Impacto Social Negativo**: El discurso de odio normaliza la violencia simbolica virtual, que puede escalar a violencia real. Tiene efectos medibles en salud mental, autocensura y exclusion en un espacio publico. Afecta desproporcionadamente a minorías
- **Escala del problema**: Hay millones de mensajes de odio por dia, lo que es dificil de moderar por humanos a mano.
- **Sesgos**: El discurso de odio en redes puede sesgar a las personas cambiando lo que parece aceptable y alterando su realidad.

Se pueden crear modelos que filtren o reduzcan estos daños.
Es fundamental para proteger a los usuarios en línea del abuso y para permitir que los proveedores de servicios ofrezcan un entorno seguro y confiable para sus usuarios.


## Tareas por hacer

- Hacer un alisis de deteccion usando LLM's abiertos
- Creacion ó busqueda de datasets mas abarcativos
- Ponerse al dia con las tendencias del area
- Mejorar metricas y liberar modelos

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

 
