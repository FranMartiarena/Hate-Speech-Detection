import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from text_preprocessing import TextPreprocessor
from file_paths import HAT_EVAL, YOU_TOXIC

class CrossDomainEvaluator:
    def __init__(self):
        self.preprocessor = TextPreprocessor(keep_negations=True)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        self.model = None

    def train_with_optimization(self, X, y):
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear']}
        grid = GridSearchCV(SVC(class_weight='balanced', probability=True), 
                            param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
        grid.fit(X, y)
        return grid.best_estimator_

    def get_error_analysis(self, df_test, y_true, y_pred, n=5):
        df_errors = df_test.copy()
        df_errors['true_label'] = y_true
        df_errors['pred_label'] = y_pred
        return df_errors[df_errors['true_label'] != df_errors['pred_label']].head(n)

if __name__ == "__main__":
    print("Este archivo es un módulo. Ejecute main.py para iniciar el pipeline.")