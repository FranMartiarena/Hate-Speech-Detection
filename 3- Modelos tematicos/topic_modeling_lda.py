import pandas as pd
import gensim
from gensim import corpora
from file_paths import HAT_EVAL
from text_preprocessing import TextPreprocessor

def convinacion_hateval():
    # Asumimos que esta función ya la tenés para unir los csv de HatEval
    # Si ya tenés el archivo hateval2019_en_convinado.csv, podés dejarla vacía
    pass

def topic_dominante():
    print("Iniciando Modelado de Tópicos (LDA)...")
    df = pd.read_csv(HAT_EVAL)
    preprocessor = TextPreprocessor()
    
    # Preprocesamiento para LDA
    textos_limpios = [preprocessor.clean(t).split() for t in df['text']]
    
    # Crear Diccionario y Corpus
    dictionary = corpora.Dictionary(textos_limpios)
    corpus = [dictionary.doc2bow(text) for text in textos_limpios]
    
    # Entrenar LDA (2 tópicos: Inmigrantes y Mujeres/Otros)
    lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, 
                                       num_topics=2, passes=10, random_state=42)
    
    # Asignar el tópico dominante a cada fila
    def get_topic(bow):
        topics = lda_model.get_document_topics(bow)
        return sorted(topics, key=lambda x: x[1], reverse=True)[0][0]

    df['topic'] = [f"Topic_{get_topic(b)}" for b in corpus]
    
    # GUARDADO: Sobrescribimos para que la columna 'topic' exista
    df.to_csv(HAT_EVAL, index=False)
    print(f"✔ Tópicos guardados en {HAT_EVAL}")

if __name__ == "__main__":
    topic_dominante()