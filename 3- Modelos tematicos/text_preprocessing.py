import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self, keep_negations=True):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        if keep_negations:
            # Quitamos palabras de negación de las stopwords para no perder sentido
            negations = {'not', 'no', 'never', "n't", 'neither', 'nor'}
            self.stop_words = self.stop_words - negations

    def clean(self, text, remove_hashtags=True):
        if remove_hashtags:
            text = re.sub(r'#\w+', '', text)
        # Solo letras y espacios
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text.lower())
        # Lematización filtrando stopwords
        cleaned = [self.lemmatizer.lemmatize(w) for w in tokens if w not in self.stop_words]
        return " ".join(cleaned)