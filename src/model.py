from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib

class ModeloSentimentos:
    def __init__(self, tipo='logistic'):
        if tipo == 'logistic':
            self.modelo = LogisticRegression(max_iter=1000, random_state=42)
        elif tipo == 'svm':
            self.modelo = LinearSVC(random_state=42, max_iter=2000)
        elif tipo == 'random_forest':
            self.modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("Tipo inválido")
        
        self.tipo = tipo
        self.metricas = {}
    
    def treinar(self, X_treino, y_treino):
        self.modelo.fit(X_treino, y_treino)
    
    def avaliar(self, X_teste, y_teste):
        y_pred = self.modelo.predict(X_teste)
        
        self.metricas = {
            'accuracy': accuracy_score(y_teste, y_pred),
            'precision': precision_score(y_teste, y_pred),
            'recall': recall_score(y_teste, y_pred),
            'f1': f1_score(y_teste, y_pred)
        }
        
        print(f"Acurácia: {self.metricas['accuracy']:.4f}")
        print(classification_report(y_teste, y_pred, target_names=['Negativo', 'Positivo']))
        
        return y_pred
    
    def salvar(self, caminho):
        joblib.dump(self.modelo, caminho)
    
    def carregar(self, caminho):
        self.modelo = joblib.load(caminho)
    
    def prever(self, X_novo):
        return self.modelo.predict(X_novo)