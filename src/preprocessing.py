import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def limpar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'[^a-z\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    texto = texto.strip()
    return texto

def carregar_dados(caminho_arquivo):
    try:
        df = pd.read_csv(caminho_arquivo)
        print(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas.")
        return df
    except FileNotFoundError:
        print(f"Erro: arquivo '{caminho_arquivo}' não encontrado.")
        return None
    except Exception as e:
        print(f"Erro ao carregar CSV: {e}")
        return None

def preprocessar_dataframe(df):
    if df is None:
        print("Erro: DataFrame é None.")
        return None
    
    if 'review' not in df.columns or 'sentiment' not in df.columns:
        print("Erro: faltam colunas 'review' e 'sentiment'.")
        return None

    df_processado = df.dropna(subset=['review', 'sentiment']).copy()
    df_processado['review_clean'] = df_processado['review'].apply(limpar_texto)
    df_processado['label'] = (df_processado['sentiment'] == 'positive').astype(int)
    
    print(f"Linhas processadas: {df_processado.shape[0]}")
    print("\nDistribuição:")
    print(df_processado['label'].value_counts())

    return df_processado

def vetorizar_textos(textos_treino, textos_teste, max_features=5000, stop_words='english', ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=stop_words,
        ngram_range=ngram_range
    )

    X_treino_vec = vectorizer.fit_transform(textos_treino)
    X_teste_vec = vectorizer.transform(textos_teste)
    
    print(f"Vetorização: treino {X_treino_vec.shape}, teste {X_teste_vec.shape}")

    return X_treino_vec, X_teste_vec, vectorizer

preprocessar = preprocessar_dataframe
