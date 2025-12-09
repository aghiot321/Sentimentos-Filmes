import sys
sys.path.append('.')

from src.preprocessing import limpar_texto
import joblib
import os

if not os.path.exists('models/svm_model.pkl'):
    print("Erro: modelos não encontrados. Execute: python src/train.py")
    sys.exit()

if not os.path.exists('models/tfidf_vectorizer.pkl'):
    print("Erro: vetorizador não encontrado. Execute: python src/train.py")
    sys.exit()

modelo = joblib.load('models/svm_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

print("\nModelo pronto. Digite avaliações em inglês:")
print("Exemplos: 'This movie was amazing!' ou 'Terrible film, waste of time.'\n")

while True:
    texto = input("Avaliação (ou 'sair'): ").strip()
    
    if texto.lower() == 'sair':
        break
    
    if not texto:
        print("Digite algo válido!")
        continue
    
    texto_limpo = limpar_texto(texto)
    X = vectorizer.transform([texto_limpo])
    previsao = modelo.predict(X)[0]
    probabilidade = modelo.decision_function(X)[0]
    
    if previsao == 1:
        print(f"POSITIVO (confiança: {abs(probabilidade):.2f})")
    else:
        print(f"NEGATIVO (confiança: {abs(probabilidade):.2f})")