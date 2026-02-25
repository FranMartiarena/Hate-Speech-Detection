import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
from sklearn.metrics import classification_report, accuracy_score
from file_paths import HAT_EVAL, YOU_TOXIC

class RobertaEvaluator:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-hate"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.labels_mapping = {0: 'not-hate', 1: 'hate'}

    def preprocess(self, text):
        new_text = []
        for t in str(text).split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def predict(self, texts):
        self.model.eval()
        processed_texts = [self.preprocess(t) for t in texts]
        encoded_input = self.tokenizer(processed_texts, return_tensors='pt', 
                                     padding=True, truncation=True, max_length=128).to(self.device)
        
        with torch.no_grad():
            output = self.model(**encoded_input)
        
        logits = output.logits.cpu().numpy()
        probs = softmax(logits, axis=1)
        predictions = np.argmax(probs, axis=1)
        return predictions, probs

    def run_evaluation(self, file_path, text_col, label_col, title):
        print(f"\n Evaluando RoBERTa en: {title} ".center(60, "="))
        df_full = pd.read_csv(file_path)
        
        # Si el dataset tiene menos de 500 filas, usa todas. Si tiene más, toma 500.
        n_to_sample = min(len(df_full), 500)
        df = df_full.sample(n_to_sample, random_state=42)
        
        texts = df[text_col].tolist()
        # Asegurar que las etiquetas sean 0 y 1
        y_true = df[label_col].apply(lambda x: 1 if str(x) in ['1', 'True', '1.0'] else 0).tolist()
        
        predictions, _ = self.predict(texts)
        
        print("\n" + classification_report(y_true, predictions, target_names=['not-hate', 'hate']))
        return accuracy_score(y_true, predictions)