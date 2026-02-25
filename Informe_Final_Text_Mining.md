# Informe Final - Text Mining

## Detección de Discurso de Odio en Hateval y YouTube Toxicity

------------------------------------------------------------------------

# 1. Introducción

La detección automática de discurso de odio constituye uno de los
desafíos más complejos dentro del procesamiento del lenguaje natural
(NLP). A diferencia de otras tareas de clasificación textual, el hate
speech no depende únicamente de la presencia de palabras ofensivas, sino
de factores contextuales, culturales y pragmáticos.

En este trabajo se desarrolló un análisis comparativo entre modelos
clásicos basados en representaciones léxicas (n-gramas + SVM) y modelos
contextuales basados en Transformers, utilizando los datasets Hateval
(Twitter) y YouTube Toxicity (YouToxic). El eje central del proyecto fue
analizar la capacidad de generalización entre dominios y evaluar si la
especialización temática mejora el rendimiento.

------------------------------------------------------------------------

# 2. Resumen Ejecutivo

El proyecto se estructuró en cuatro etapas:

1.  Modelado temático con LDA para identificar subdominios de odio.
2.  Entrenamiento de modelos SVM generales y especializados por tópico.
3.  Evaluación cruzada para analizar transferencia entre dominios.
4.  Comparación con un modelo de lenguaje preentrenado basado en
    Transformers.

Los resultados evidencian que:

-   Los modelos clásicos presentan fuerte dependencia del dominio.
-   La especialización temática no resuelve el problema estructural de
    transferencia.
-   Los modelos contextuales mejoran significativamente el balance entre
    precisión y recall.
-   Persisten desafíos asociados a diferencias de distribución entre
    plataformas.

------------------------------------------------------------------------

# 3. Hipótesis de Trabajo

1.  Un modelo entrenado en un dominio específico no generaliza
    adecuadamente a otro dominio con distinta distribución temática y
    léxica.
2.  El modelado temático permitirá mejorar la especialización y,
    potencialmente, el rendimiento intra-dominio.

------------------------------------------------------------------------

# 4. Desarrollo Experimental

## 4.1 Modelado Temático (LDA)

Se aplicó LDA para identificar estructuras latentes en los datasets. El
análisis reveló:

-   Hateval: fuerte presencia de discurso contra inmigrantes y
    misoginia.
-   YouToxic: predominancia de racismo asociado a conflictos policiales.

El uso de LDA fue fundamental como herramienta exploratoria para
segmentar los datos y entrenar modelos especializados.

Sin embargo, el modelado temático no incorpora semántica contextual
profunda, sino co-ocurrencia estadística de términos.

------------------------------------------------------------------------

## 4.2 Modelos Clásicos (SVM + n-gramas)

Los modelos fueron entrenados utilizando representación TF-IDF basada en
n-gramas.

### Resultados intra-dominio:

Buen desempeño dentro del mismo dataset.

### Evaluación cruzada:

Se observaron dos comportamientos extremos:

-   Modelo "paranoico": alto recall y baja precisión.
-   Modelo "ciego": baja capacidad de detección en dominio nuevo.

Esto demuestra que los modelos aprenden correlaciones léxicas
superficiales y no estructuras semánticas generalizables.

------------------------------------------------------------------------

## 4.3 Evaluación con Modelo Transformer

Se utilizó un modelo preentrenado especializado en clasificación de
discurso de odio.

Resultados observados:

-   Mejor balance entre precisión y recall.
-   Reducción significativa de falsos positivos asociados a palabras
    gatillo.
-   Mayor robustez frente a frases ambiguas.

Sin embargo, cuando la distribución temática cambia drásticamente,
incluso el modelo contextual experimenta degradación de rendimiento.

------------------------------------------------------------------------

# 5. Análisis Profundo de Transferencia de Dominio

La transferencia de dominio falló principalmente por:

1.  Diferencias en longitud promedio de comentarios.
2.  Diferencias en estilo discursivo (Twitter vs YouTube).
3.  Diferencias en proporción de clases.
4.  Diferencias en vocabulario específico de cada conflicto social.

Los modelos clásicos basados en frecuencia no capturan intención
comunicativa ni relaciones sintácticas complejas, lo que explica su baja
capacidad de adaptación.

------------------------------------------------------------------------

# 6. Relación entre Hipótesis, Objetivos y Estado Final

La primera hipótesis fue confirmada empíricamente. La degradación del
F1-score en evaluación cruzada demuestra limitaciones estructurales de
generalización.

La segunda hipótesis fue parcialmente validada: LDA permitió aislar
temáticas coherentes, pero la especialización no compensó la falta de
comprensión contextual.

Los objetivos fueron alcanzados, logrando no solo implementar modelos,
sino analizar críticamente sus limitaciones.

------------------------------------------------------------------------

# 7. Relación entre Planificación y Ejecución

La planificación fue seguida en su mayoría:

-   Recolección y limpieza de datos.
-   Modelado temático.
-   Entrenamiento y evaluación cruzada.

La etapa de evaluación de múltiples LLMs debió reducirse por
limitaciones temporales, optándose por un modelo representativo.

Esta adaptación no comprometió la validez experimental, pero limitó la
comparación amplia entre arquitecturas.

------------------------------------------------------------------------

# 8. Librerías y Justificación Técnica

## Gensim

Elegida para LDA por eficiencia y compatibilidad con PyLDAvis.

## Scikit-learn

Utilizada para SVM y métricas por: - Facilidad de implementación. -
Pipeline integrado. - Amplia documentación.

## Transformers (Hugging Face)

Seleccionada por: - Acceso a modelos preentrenados. - Simplicidad en
inferencia. - Comunidad activa.

## Librerías complementarias

Pandas, NumPy, Matplotlib y NLTK fueron utilizadas para manipulación,
análisis y visualización.

------------------------------------------------------------------------

# 9. Discusión sobre Devoluciones Recibidas

Las devoluciones destacaron:

-   Necesidad de clarificar el concepto de generalización.
-   Explicar el rol exploratorio del LDA.
-   Aclarar cómo se aplicaba el LLM.
-   Problematizar la ambigüedad y el humor.

Se integraron estas sugerencias incorporando definiciones formales y
ampliando la discusión sobre limitaciones semánticas.

No fue posible incorporar etiquetado explícito de humor debido a
restricciones del dataset.

------------------------------------------------------------------------

# 10. Relación con Trabajo Previo (Bibliografía)

El trabajo se alinea con investigaciones previas que indican:

-   Limitaciones de modelos basados en frecuencia.
-   Importancia de embeddings contextuales.
-   Dificultad estructural de la transferencia entre dominios en NLP.

Los resultados obtenidos son consistentes con la literatura sobre domain
adaptation y hate speech detection.

------------------------------------------------------------------------

# 11. Implicaciones y Continuación del Proyecto

Con un equipo de cinco personas durante un año se podría:

-   Construir dataset multi-dominio balanceado.
-   Incorporar anotaciones de ironía.
-   Implementar técnicas de domain adaptation.
-   Analizar sesgos algorítmicos.
-   Comparar múltiples arquitecturas Transformer.

------------------------------------------------------------------------

# 12. Conclusión Final

El discurso de odio no puede reducirse a una lista de palabras
prohibidas. Es un fenómeno contextual, dinámico y dependiente del
dominio.

Los modelos clásicos muestran limitaciones severas de transferencia. Los
modelos contextuales representan un avance significativo, aunque el
problema sigue abierto.

El proyecto demuestra que la verdadera dificultad no es detectar
palabras ofensivas, sino comprender intención, contexto y variabilidad
cultural.
