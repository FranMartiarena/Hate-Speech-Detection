# Indice
{Crear indice} 

# Detección de odio en texto generado por usuarios

El discurso de odio puede definirse como discurso que expresa o incita a dañar a un individuo o a un grupo de personas en función de una o más de sus características personales, como el género, la raza, la religión, la sexualidad, etc.

En este proyecto exploramos varias estrategias de deteccion de discurso de odio en redes, viendolo como un problema de clasificacion y usando conjuntos de datos ya existentes. Inicialmente se realizó un analisis y curacion de los distintos datasets, donde se encontraron caracteristicas relevantes que permitieron obtener una mejor prediccion. Posteriormente, utilizamos modelos convencionales de clasificacion (SVM) y modelos basados en transformers para predecir si un comentario es discurso de odio o no, concluyendo que los modelos de lenguaje basados en transformers, pre-entrenados y finetuneados para la deteccion de odio superan ampliamente a los modelos convencionales.

El objetivo de este trabajo es, por un lado, explorar distintas tecnicas de deteccion de discurso de odio y, por el otro, investigar formas en las que pueden ser solucionadas distintas problematicas utilizando estas tecnicas, siempre manteniendo un esquema abierto al publico.


## ¿Porque detectar discurso de odio?

La deteccion del discurso de odio ayuda a mitigar las siguientes problematicas (Entre muchas otras):

- **Impacto Social Negativo**: El discurso de odio normaliza la violencia simbolica virtual, que puede escalar a violencia real. Tiene efectos medibles en salud mental, autocensura y exclusion en un espacio publico. Afecta desproporcionadamente a minorías.

- **Escala del alcance**: Hay millones de mensajes de odio por dia, lo que es dificil de moderar por humanos a mano y puede tener un impacto negativo en su salud.

- **Sesgos**: El discurso de odio en redes puede sesgar a las personas cambiando lo que parece aceptable y alterando su realidad.

- **Polarizacion afectiva**: El discurso de odio impulsa la polarizacion afectiva, aumentando desconfianza y haciendo cada vez mas dificil alcanzar acuerdos en la sociedad.

Se pueden crear modelos que filtren o reduzcan estos daños.
Es fundamental para proteger a los usuarios en línea del abuso y para permitir que los proveedores de servicios ofrezcan un entorno seguro y confiable para sus usuarios.

## Problemas Fundamentales

- **Contexto**: Existen distintas comunidades y contextos, y lo que puede ser Hate Speech en una capaz no lo es en otra. Supongamos que estamos en un partido de futbol y una persona dice "estos tipos son unos animales" refiriendose al equipo contrario. Esta frase si bien tiene odio no es Hate Speech, ya que no ataca un colectivo, mientras que si el contexto fuese distinto, como por ejemplo un politico conocido por su racismo que está dando un discurso, entonces si es Hate Speech. Este ejemplo es algo extremo, pero cuando se piensa en clasificar texto en linea el contexto y comunidad influye. Hay claras diferencias entre Chats de distintos streamer(i.e. Twitch), videos de youtube, hilos o hashtags de Twitter, etc. Cada plataforma tiene su forma distinta de dar contexto.

- **Lenguaje dinámico(Tendencias, Jerga, Evolucion)**: Siempre surgen nuevas palabras o nuevos usos para palabras ya existentes. El odio (y el lenguaje en general) se adapta y cambia constantemente, por lo que distintos datasets quedan rapidamente desactualizados. Es importante encontrar alguna forma de mantener el clasificador de Hate Speech al tanto de estos cambios, que se adapte a nuevos patrones linguisticos que adopta la comunidad.  

- **Lenguaje implícito y figurado**: Hay fenomenos asociados al lenguaje que puede hacer muy dificil detectar Hate Speech, en particular cuando no se menciona explicitamente el grupo objetivo, por ejemplo “Ya sabemos cómo son…”. En estos casos el contexto pareciera ser fundamental para realizar una correcta clasificacion.

    - **Ironia**: Se dice lo contrario de lo que se quiere expresar, un ejemplo podria ser el comentario “Claro, porque ellos siempre son tan trabajadores 🙄”. El texto literal parece positivo, pero el tono es negativo y en el contexto adecuado puede ser Hate Speech.
        
    - **Metaforas y Analogias**: Se realiza un comentario de odio explicando conceptos en terminos de otros, por ejemplo "Son una plaga", "Hay que limpiar esto". Las palabras plaga y limpiar pueden estar haciendo referencia a cierto grupo, convirtiendolo en Hate Speech.   
    
    - **eufemismos**: Formas “suavizadas” de discurso de odio, "Gente de bien" puede ser una expresion ambigua pero que segrega e incita odio a minorias.

Una de las soluciones propuestas a los problemas de lenguaje de odio implicito es extender los dataset, generarando por cada dato un parafraseo que tenga (o no) ironia, metafora, y eufemismo usando un LLM, de esta forma el modelo aprende a detectar cada una de las formas.

- **Insulto vs odio**: Si bien el insulto a un individuo especifico es odio y hay que mitigarlo (i.e. “Juan, sos un idiota”), esto no se considera Hate Speech ya que no hace referencia a un colectivo (como por ejemplo inmigrantes). Las categorias de odio son varias y pueden solaparse, entre ellas se encuentran:
    
    - Hate Speech: Abuso dirigido a un grupo protegido (o individuo como miembro de ese grupo).
    - Lenguaje Toxico: Cualquier contenido que puede ser percibido como ofensivo, dañino o disruptivo.
    - Lenguaje Abusivo: Lenguaje que ataca o denigra a alguien (individuo o grupo).
    - Harassment: Abuso dirigido y repetido hacia un target.

La division de odio en estas categorias es util ya que los diferentes tipos de daño requieren distintas respuestas o soluciones, i.e. Hate Speech tiene implicaciones sociales mas amplias que un comentario toxico a Juan, y se resuelven de distintas maneras. 
Este trabajo se centra en Hate Speech, pero a futuro se buscará desarrollar un sistema que abarque todo tipo de odio.

- **Idiomas**: El Hate Speech surge en todos los idiomas, y es importarte dar soporte a cada uno de ellos. En este trabajo nos centramos en comentarios en Ingles y Español, pero estaria bueno generalizar a todos los idiomas.

## Data
Para la etapa inicial de deteccion se realizaron pruebas sobre 2 datasets que consideramos relevantes. Si bien un sistema de deteccion de odio necesita datos especificos a su contexto para una mejor deteccion (por ejemplo comentarios de un chat de streaming especifico), la idea es usar estos datasets para tratar de definir un baseline y entender el problema fundamental. A futuro se buscaran soluciones al problema de deteccion de Hate Speech global independiente de contexto.

- **[HatEval](https://huggingface.co/datasets/valeriobasile/HatEval)**: Dataset de comentarios en español e ingles, sacado de Twitter y centrado en la deteccion de odio hacia inmigrantes y mujeres. Este dataset fue parte del Workshop SemEval-2019, Task 5: Multilingual Detection of Hate Speech Against Immigrants and Women in Twitter.

- **[Youtoxic](https://www.kaggle.com/datasets/reihanenamdari/youtube-toxicity-data)**: Dataset de comentarios de youtube sobre el Ferguson unrest, centrado en la deteccion de racismo. El dataset esta puramente en ingles y ademas cuenta con etiquetas de toxicidad, Lenguaje abusivo, amenaza, provocativo, entre otras. Permite detecetar de manera mas fina las categorias de odio.

## Hipotesis de trabajo

Las hipotesis iniciales del trabajo fueron:

- La detección de discurso de odio entrenada en un conjunto de datos no generaliza bien a otro conjunto de datos diferente, como en este caso entre los datasets  "Hateval" y "YouTube Toxicity".

- El uso de técnicas de modelado temático (LDA) permitirá identificar con mayor precisión el discurso de odio asociado a temas específicos dentro de los datasets.

- El uso de arquitecturas basadas en transformers superará a modelos convencionales como Support vector machines.

## Resultados

Luego de entrenar modelos convencionales sobre cada dataset, se corroboró que hacer evaluacion cruzada no dá resultados muy prometedores y  que dichos datasets no permiten que el modelo generalice el contexto. 

Ademas, utilizar tecnicas de modelado tematico fue util para encontrar el colectivo objetivo en caso de haber HateSpeech, pero unicamente dentro del contexto sobre el que fue entrenado. 

Por ultimo, se entrenó y realizo fine tuning a una arquitectura basada en Transformers (DistilBERT) utilizando el dataset HatEval y superando ampliamente a los modelos convencionales en la tarea de clasificacion.   

## Tareas por hacer

- Maneras de detectar el target group si hay Hate Speech
- Hacer un alisis de deteccion usando LLM's abiertos
- Creacion ó busqueda de datasets mas abarcativos, buscar generalizar la tarea y adaptar en tiempo real al contexto y cambios de lenguaje
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

## Bibliografia
{Agregar links de trabajos parecidos}
 
