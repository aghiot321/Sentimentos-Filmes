import sys
sys.path.append('.')

from sklearn.model_selection import train_test_split
from src.preprocessing import carregar_dados, preprocessar, vetorizar_textos
from src.model import ModeloSentimentos
import joblib

print("Carregando dados...")
try:
    df = carregar_dados('data/raw/IMDB Dataset.csv')
except:
    print("Erro: arquivo não encontrado")
    sys.exit()

print("Preprocessando...")
df = preprocessar(df)

print("Dividindo dados...")
X_treino, X_teste, y_treino, y_teste = train_test_split(
    df['review_clean'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)
print(f"Treino: {len(X_treino)} | Teste: {len(X_teste)}")

print("Vetorizando textos...")
X_treino_vec, X_teste_vec, vectorizer = vetorizar_textos(X_treino, X_teste)
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

print("\nTreinando modelos...")
modelos = {
    'logistic': ModeloSentimentos('logistic'),
    'svm': ModeloSentimentos('svm'),
    'random_forest': ModeloSentimentos('random_forest')
}

resultados = {}

for nome, modelo in modelos.items():
    print(f"\n{nome.upper()}:")
    modelo.treinar(X_treino_vec, y_treino)
    y_pred = modelo.avaliar(X_teste_vec, y_teste)
    resultados[nome] = modelo.metricas
    modelo.salvar(f'models/{nome}_model.pkl')

print("\n" + "="*50)
print("RESULTADOS:")
print("="*50)

melhor_modelo = None
melhor_score = 0

for nome, metricas in resultados.items():
    print(f"\n{nome}:")
    print(f"  Acurácia:  {metricas['accuracy']:.4f}")
    print(f"  Precisão:  {metricas['precision']:.4f}")
    print(f"  Recall:    {metricas['recall']:.4f}")
    print(f"  F1-Score:  {metricas['f1']:.4f}")
    
    if metricas['f1'] > melhor_score:
        melhor_score = metricas['f1']
        melhor_modelo = nome

print(f"\nMelhor modelo: {melhor_modelo} (F1: {melhor_score:.4f})")
print("\nTreinamento concluído.")
print("   2. Teste suas próprias avaliações!")
print("\n")