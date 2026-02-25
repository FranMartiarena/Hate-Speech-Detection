# Punto 4: Análisis de Performance de LLMs vs. Modelos Clásicos

## 🚀 Guía de Instalación y Ejecución (Punto 4)

Este módulo permite evaluar cómo un modelo de lenguaje basado en **Transformers (RoBERTa)** aborda la detección de odio en comparación con las técnicas de Machine Learning tradicional (SVM).

### 1. Requisitos Previos
* Python 3.9 o superior.
* Entorno virtual activado.

### 2. Instalación de librerías necesarias
Ejecute el siguiente comando para instalar las dependencias de Deep Learning:
```powershell
pip install torch transformers scipy pandas scikit-learn
```
### 3. Ejecución del Modelo
Para iniciar la inferencia con RoBERTa y obtener las métricas de evaluación:
```
python roberta_model_evaluation.py
```

## 🎯 Objetivo
Determinar si los Modelos de Lenguaje (LLMs) logran superar las limitaciones de "ceguera léxica" y "paranoia" observadas en los modelos SVM del Punto 3, utilizando su capacidad de comprensión contextual.
## 1. El Modelo: 
Twitter-RoBERTa-BaseSe utilizó el modelo cardiffnlp/twitter-roberta-base-hate. A diferencia de la SVM (basada en n-gramas), RoBERTa utiliza Self-Attention para entender la relación entre palabras.
## 2. Resultados de Performance:
 RoBERTa Tras procesar los datasets de HatEval y YouToxic, los resultados obtenidos por el modelo de lenguaje son los siguientes:

| Métrica | RoBERTa (HatEval) | RoBERTa (YouToxic) |
| :--- | :---: | :---: |
| **Accuracy** | **0.82** | **0.86** |
| **Precision (Hate)** | 0.73 | 0.35 |
| **Recall (Hate)** | **0.90** | 0.11 |
| **F1-Score (Macro)** | **0.82** | **0.55** |

### Observación clave:
En HatEval, RoBERTa logró un Recall del 90%, superando significativamente la capacidad de detección de los modelos anteriores sin sacrificar la precisión.


## 3. Comparativa: SVM vs. LLM (Análisis Exploratorio)
### ¿Resolvió el LLM los errores del Punto 3?
Sometimos al LLM a los casos donde la SVM fallaba por sesgo léxico (identificar odio solo por palabras clave). Los resultados confirman la superioridad del análisis contextual:

|Texto del Comentario|Resultado SVM|Resultado RoBERTa|Análisis de Mejora
| :--- | :---: | :---: | :--- | 
|The Arab dude is absolutely right, he should not have been shot|Hate (Error)|NOT-HATE (Acierto)|El modelo entiende que es una frase de apoyo.|
|Refugees welcome, tourists go home|Hate (Error)|NOT-HATE (Acierto)|Identifica la intención positiva de "welcome".|