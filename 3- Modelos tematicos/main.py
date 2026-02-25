import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from text_preprocessing import TextPreprocessor
from file_paths import HAT_EVAL, YOU_TOXIC

class CrossDomainEvaluator:
    def __init__(self):
        self.preprocessor = TextPreprocessor(keep_negations=True)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        self.model = SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True)

    def prepare_data(self, df, text_col, label_col):
        print(f"-> Preprocesando {len(df)} filas...")
        # Limpieza de textos
        df['clean_text'] = df[text_col].apply(self.preprocessor.clean)
        return df['clean_text'], df[label_col], df[text_col]

    def show_errors(self, df_results, n=5):
        """Muestra ejemplos de Falsos Positivos y Falsos Negativos."""
        print("\n" + " ANALISIS DE ERRORES ".center(50, "="))
        
        # Falsos Positivos: Predijo Odio (1) pero era Neutral (0)
        fps = df_results[(df_results['y_true'] == 0) & (df_results['y_pred'] == 1)]
        print(f"\n[FALSOS POSITIVOS] (Predijo Odio - Era Neutral) - Total: {len(fps)}")
        for i, row in fps.head(n).iterrows():
            print(f"- Texto: {row['original_text'][:100]}...")

        # Falsos Negativos: Predijo Neutral (0) pero era Odio (1)
        fns = df_results[(df_results['y_true'] == 1) & (df_results['y_pred'] == 0)]
        print(f"\n[FALSOS NEGATIVOS] (Predijo Neutral - Era Odio) - Total: {len(fns)}")
        for i, row in fns.head(n).iterrows():
            print(f"- Texto: {row['original_text'][:100]}...")

    def run_experiment(self, train_df, test_df, train_cols, test_cols, title):
        print(f"\n{'#'*60}")
        print(f"{title.center(60)}")
        print(f"{'#'*60}")
        
        # Preparar Entrenamiento
        X_train_raw, y_train, _ = self.prepare_data(train_df, train_cols[0], train_cols[1])
        X_train = self.vectorizer.fit_transform(X_train_raw)
        
        # Entrenar
        print("-> Entrenando SVM...")
        self.model.fit(X_train, y_train)
        
        # Preparar Test
        X_test_raw, y_test, original_text = self.prepare_data(test_df, test_cols[0], test_cols[1])
        X_test = self.vectorizer.transform(X_test_raw)
        
        # Predecir
        y_pred = self.model.predict(X_test)
        
        # Reporte de métricas
        print("\n" + " REPORTE DE CLASIFICACIÓN ".center(50, "-"))
        print(classification_report(y_test, y_pred))
        
        # Guardar resultados para análisis de error
        results_df = pd.DataFrame({
            'original_text': original_text,
            'y_true': y_test.values if isinstance(y_test, pd.Series) else y_test,
            'y_pred': y_pred
        })
        
        self.show_errors(results_df)

def main():
    evaluator = CrossDomainEvaluator()
    
    # Carga de datos
    try:
        df_hateval = pd.read_csv(HAT_EVAL)
        df_youtoxic = pd.read_csv(YOU_TO_TOXIC) if 'YOU_TO_TOXIC' in locals() else pd.read_csv(YOU_TOXIC)
    except Exception as e:
        print(f"Error al cargar archivos: {e}")
        return

    menu = """
    ======================================================
    PUNTO 3: MENÚ DE EXPERIMENTOS
    ======================================================
    1. Entrenar HatEval (Inmigrantes) -> Probar en YouToxic
    2. Entrenar YouToxic (Racismo) -> Probar en HatEval (Inmigrantes)
    3. Salir
    Seleccione opción: """

    while True:
        choice = input(menu)
        if choice == '1':
            df_imm = df_hateval[df_hateval['topic'] == 'Topic_0'].copy()
            evaluator.run_experiment(df_imm, df_youtoxic, ('text', 'HS'), ('Text', 'IsHatespeech'), "HAT-EVAL (INM) -> YOUTOXIC")
        elif choice == '2':
            df_imm = df_hateval[df_hateval['topic'] == 'Topic_0'].copy()
            evaluator.run_experiment(df_youtoxic, df_imm, ('Text', 'IsHatespeech'), ('text', 'HS'), "YOUTOXIC -> HAT-EVAL (INM)")
        elif choice == '3':
            break
        else:
            print("Opción inválida.")

if __name__ == "__main__":
    main()