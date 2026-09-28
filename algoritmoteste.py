# Importar bibliotecas necessárias
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Carregar dados
iris = load_iris()

# Features (variáveis de entrada)
X = iris.data

# Target (classe da flor)
y = iris.target

print("Shape dos dados:", X.shape)
print("Primeiras linhas:")
print(X[:5])

# 2. Dividir treino/teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Treino:", X_treino.shape)
print("Teste:", X_teste.shape)

# 3. Treinar modelo
modelo = DecisionTreeClassifier(max_depth=3)

modelo.fit(X_treino, y_treino)

print("Modelo treinado com sucesso!")

# 4. Fazer previsões
y_pred = modelo.predict(X_teste)

print("Previsões:")
print(y_pred[:10])

# Avaliar desempenho
acuracia = accuracy_score(y_teste, y_pred)

print(f"Acurácia do modelo: {acuracia:.2%}")