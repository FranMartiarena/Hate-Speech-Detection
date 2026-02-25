from roberta_model_evaluation import RobertaEvaluator
from file_paths import HAT_EVAL, YOU_TOXIC # Importamos los 

def main():
    print("\n" + "="*60)
    print("EJECUCIÓN FINAL: RoBERTa")
    print("="*60)

    evaluator = RobertaEvaluator()

    # 1. Evaluar en HatEval  (Combinado de train/test/dev)
    # Según tu file_paths.py, HAT_EVAL apunta a '../data_set/hateval2019_en_convinado.csv'
    evaluator.run_evaluation(HAT_EVAL, 'text', 'HS', "HAT-EVAL")

    # 2. Evaluar en YouToxic 
    # Según tu file_paths.py, YOU_TOXIC apunta a '../data_set/youtoxic_english_1000.csv'
    evaluator.run_evaluation(YOU_TOXIC, 'Text', 'IsHatespeech', "YOU-TOXIC")

    # 3. Prueba de fuego (Casos donde SVM fallaba)
    print("\n" + "="*60)
    print("COMPROBACIÓN DE CASOS CRÍTICOS")
    print("="*60)
    
    test_cases = [
        "The Arab dude is absolutely right", # SVM solía marcarlo como odio por 'Arab'
        "Refugees welcome, tourists go home",  # SVM se confundía con 'Refugees'
        "I hate how people treat others poorly" # Usa 'hate' pero no es discurso de odio
    ]
    
    preds, _ = evaluator.predict(test_cases)
    
    for txt, p in zip(test_cases, preds):
        print(f"Texto: {txt}\nPredicción: {'HATE' if p==1 else 'NOT-HATE'}\n")

if __name__ == "__main__":
    main()