# Informe Final - Text Mining

## Detección de Discurso de Odio en Hateval y YouTube Toxicity

------------------------------------------------------------------------

# 1. Resumen

El presente trabajo aborda el desarrollo, análisis y comparación de
distintos enfoques para la detección automática de discurso de odio
utilizando dos conjuntos de datos: Hateval (comentarios de Twitter) y
YouTube Toxicity (comentarios de YouTube).

El proyecto se estructuró en cuatro ejes principales:

1.  Análisis exploratorio mediante modelado temático (LDA).
2.  Entrenamiento de modelos clásicos de clasificación (SVM).
3.  Evaluación cruzada para analizar la transferencia entre dominios.
4.  Comparación con un modelo de lenguaje basado en Transformers.

Los resultados obtenidos permiten concluir que la detección de odio es
altamente dependiente del dominio y que los modelos clásicos basados en
n-gramas presentan limitaciones significativas de generalización. Los
modelos contextuales muestran mejoras sustanciales, aunque tampoco
resuelven completamente el problema.

------------------------------------------------------------------------

# 2. Hipótesis de Trabajo

1.  La detección de discurso de odio entrenada en un conjunto de datos
    no generaliza adecuadamente a otro dominio diferente.
2.  El uso de técnicas de modelado temático (LDA) permitirá identificar
    con mayor precisión el discurso de odio asociado a temas específicos
    dentro de los datasets.

------------------------------------------------------------------------

# 3. Objetivos

-   Entrenar un modelo capaz de detectar discurso de odio.
-   Analizar su capacidad de generalización entre plataformas.
-   Identificar temáticas comunes y diferenciales entre los datasets.
-   Evaluar y contrastar modelos clásicos con modelos de lenguaje
    preentrenados.
-   Analizar errores y comportamientos extremos en escenarios de
    transferencia.

------------------------------------------------------------------------

# 4. Técnicas Relevantes

## 4.1 Modelado Temático (LDA)

Se utilizó Latent Dirichlet Allocation (LDA) para identificar
estructuras latentes en los textos.\
En Hateval emergieron principalmente dos ejes temáticos: odio contra
inmigrantes y misoginia.\
En YouToxic predominó el racismo vinculado a conflictos sociales.

El LDA fue utilizado como herramienta exploratoria y de segmentación
temática, no como clasificador final.

## 4.2 Representación Léxica

Se utilizó Bag of Words y n-gramas ponderados con TF-IDF para
representar los textos en forma vectorial.

## 4.3 Clasificación Supervisada

Se implementaron modelos Support Vector Machines (SVM) por su:

-   Simplicidad.
-   Buen rendimiento en alta dimensionalidad.
-   Facilidad de implementación.

## 4.4 Modelos de Lenguaje (LLMs)

Se evaluó un modelo preentrenado especializado en clasificación de odio
utilizando la librería Transformers.\
El modelo fue aplicado directamente a los comentarios para inferencia
supervisada sin reentrenamiento completo.

------------------------------------------------------------------------

# 5. Desarrollo Experimental (Punto 3 y 4 del Proyecto)

## 5.1 Modelos Temáticos y Transferibilidad

Se entrenaron:

-   Un modelo general por dataset.
-   Modelos especializados por tópico identificado mediante LDA.

### Evaluación cruzada

Se aplicaron los modelos entrenados en un dataset sobre el otro.

Resultados observados:

-   HatEval → YouToxic: alto recall, baja precisión (modelo
    "paranoico").
-   YouToxic → Hateval: muy bajo recall (modelo "ciego").

Estos resultados confirman que los modelos basados en frecuencia
aprenden correlaciones léxicas superficiales y no estructuras semánticas
transferibles.

------------------------------------------------------------------------

## 5.2 Evaluación con Modelo de Lenguaje

El modelo basado en Transformers mostró:

-   Mejor equilibrio entre precisión y recall.
-   Menor dependencia de palabras gatillo.
-   Mejor tratamiento de frases ambiguas.

No obstante, cuando la distribución temática difiere considerablemente,
el rendimiento también se degrada.

------------------------------------------------------------------------

# 6. Problemas Detectados

Durante el análisis se identificaron limitaciones importantes:

-   Diferencias de distribución entre plataformas.
-   Diferencias en longitud promedio de los comentarios.
-   Diferencias en proporción de clases.
-   Falta de anotación explícita de humor o ironía.
-   Poca diversidad temática dentro de cada dataset.

Se detectó además la necesidad de cuidado al unificar fuentes de datos,
ya que cada plataforma presenta dinámicas discursivas distintas.

------------------------------------------------------------------------

# 7. Evaluación

Para evaluar los modelos se utilizó:

-   Matriz de confusión.
-   Accuracy.
-   Precision.
-   Recall.
-   F1-score.

Se definió "generalizar bien" como mantener métricas similares al
desempeño intra-dominio al aplicarse en otro dataset.

La degradación observada confirma la dificultad estructural de la
transferencia entre dominios en tareas de NLP.

------------------------------------------------------------------------

# 8. Relación entre Hipótesis, Objetivos y Estado Final

La primera hipótesis fue confirmada: la generalización entre dominios
fue limitada y dependiente del léxico y contexto específico.

La segunda hipótesis fue parcialmente validada: LDA permitió identificar
temas coherentes, pero la especialización temática no compensó la falta
de comprensión contextual profunda.

Los objetivos fueron alcanzados, aunque el análisis reveló que la mejora
real proviene de modelos contextuales más que de la segmentación
temática.

------------------------------------------------------------------------

# 9. Relación entre Planificación y Ejecución

La planificación fue seguida en gran medida:

-   Recolección y caracterización de datos.
-   Preprocesamiento y análisis exploratorio.
-   Modelado temático.
-   Entrenamiento y evaluación cruzada.

La etapa final (evaluación de múltiples LLMs) debió reducirse por
limitaciones temporales, optándose por evaluar un único modelo
representativo.

Esto no invalida los resultados, pero limita la amplitud comparativa.

------------------------------------------------------------------------

# 10. Discusión sobre Devoluciones de Otros Grupos

Las devoluciones recibidas aportaron claridad y permitieron fortalecer
el proyecto:

-   Se aclaró el concepto de generalización.
-   Se explicó que LDA cumple un rol exploratorio.
-   Se detalló cómo se aplicó el modelo de lenguaje.
-   Se discutió la problemática de humor y ambigüedad.

Respecto a la detección de humor, se concluyó que depende del criterio
de anotación humana y del contexto externo no disponible en los
datasets.

------------------------------------------------------------------------

# 11. Relación con Trabajo Previo (Bibliografía)

El proyecto se apoyó en:

-   Documentación del dataset Hateval.
-   Estudios sobre métricas de evaluación en clasificación binaria.
-   Literatura sobre domain adaptation en NLP.
-   Investigaciones sobre embeddings contextuales y Transformers.

Los resultados obtenidos son consistentes con investigaciones previas
que señalan limitaciones de modelos basados en frecuencia y ventajas de
representaciones contextuales.

------------------------------------------------------------------------

# 12. Librerías y Justificación Técnica

## Gensim (LDA)

Elegida por eficiencia y compatibilidad con visualización mediante
PyLDAvis.

## Scikit-learn (SVM y métricas)

Seleccionada por: - Facilidad de implementación. - Integración de
métricas. - Simplicidad para prototipado rápido.

## Transformers (Hugging Face)

Elegida por: - Acceso a modelos preentrenados. - Comunidad activa. -
Documentación extensa.

## Librerías Complementarias

-   NLTK (stopwords).
-   Pandas y NumPy (procesamiento).
-   Matplotlib (visualización).

------------------------------------------------------------------------

# 13. Implicaciones y Planificación Futura

Con un equipo de cinco personas durante un año, el proyecto podría
ampliarse mediante:

-   Construcción de un dataset multi-dominio balanceado.
-   Inclusión de anotaciones explícitas de ironía y ambigüedad.
-   Implementación de técnicas de domain adaptation.
-   Comparación sistemática de múltiples modelos Transformer.
-   Análisis de sesgos algorítmicos.
-   Evaluación en múltiples idiomas.

Esto permitiría redefinir el punto de partida hacia un enfoque
multi-dominio desde el inicio.

------------------------------------------------------------------------

# 14. Conclusión General

La detección de discurso de odio es un problema complejo que trasciende
la simple identificación de palabras ofensivas.

Los modelos clásicos muestran fuertes limitaciones de transferencia.\
Los modelos contextuales representan una mejora significativa, aunque el
desafío persiste debido a la variabilidad cultural, contextual y
temática del lenguaje humano.

El proyecto no solo implementó modelos, sino que permitió comprender
críticamente las dificultades estructurales del problema.
