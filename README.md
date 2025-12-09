# Análise de Sentimentos em Avaliações de Filmes

Projeto para classificar avaliações de filmes do IMDB como positivas ou negativas usando machine learning.

## Dataset

Usa o [IMDB Dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) com 50 mil avaliações em inglês. O arquivo CSV deve estar em `data/raw/IMDB Dataset.csv`.

## Estrutura

```
sentimentos-filmes/
├── data/raw/           # dados brutos
├── models/             # modelos treinados
├── src/
│   ├── preprocessing.py    # limpeza e vetorização
│   ├── model.py           # classe do modelo
│   ├── train.py           # script de treinamento
│   └── prever.py          # interface para predições
└── README.md
```

## Como usar

### 1. Preparar ambiente

```bash
python -m venv venv
venv\Scripts\activate
pip install pandas scikit-learn joblib
```

### 2. Treinar modelo

```bash
python src/train.py
```

Vai testar três algoritmos (Regressão Logística, SVM e Random Forest) e salvar o melhor em `models/`.

### 3. Fazer predições

```bash
python src/prever.py
```

Digite avaliações em inglês para ver se são classificadas como positivas ou negativas.

## Pré-processamento

- Remove caracteres especiais e números
- Converte tudo para minúsculas
- Vetorização com TF-IDF (5000 features)
- Stop words em inglês removidas

## Modelos

Três algoritmos são comparados:

- **Logistic Regression** (baseline rápido)
- **Linear SVM** (geralmente o melhor desempenho)
- **Random Forest** (testa ensemble)

O melhor modelo é salvo automaticamente.

## Métricas

O script mostra accuracy, precision, recall e F1-score para cada modelo. Geralmente o SVM alcança ~89% de accuracy no conjunto de teste.

## Dependências

- pandas
- scikit-learn
- joblib

## Observações

O modelo só funciona com texto em inglês, já que foi treinado no dataset do IMDB.
