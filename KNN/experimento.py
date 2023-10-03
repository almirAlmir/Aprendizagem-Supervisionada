# Importe as bibliotecas necessárias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

dataset = pd.read_csv('heart_cleveland_upload.csv')

#Separo o rótulo condition dos recursos do dataset
X = dataset.drop('condition', axis=1)
y = dataset['condition']

# Dividir o conjunto de dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizando os recursos X
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Escolher hiperparâmetros usando Grid Search
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Obtendo os melhores hiperparâmetros
best_k = grid_search.best_params_['n_neighbors']


knn_classifier = KNeighborsClassifier(n_neighbors=best_k)
knn_classifier.fit(X_train, y_train)

# Previsões no conjunto y de teste
y_pred = knn_classifier.predict(X_test)

# Avaliando acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)

# Calculando o F1 Score e a Precisão para diferentes valores de "k"
f1_scores = []
precision_scores = []
for k in param_grid['n_neighbors']:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_k = knn.predict(X_test)
    f1_k = f1_score(y_test, y_pred_k)
    precision_k = precision_score(y_test, y_pred_k)
    f1_scores.append(f1_k)
    precision_scores.append(precision_k)


# Gráfico para visualizar o desempenho de diferentes valores de "k" na acurácia
results = grid_search.cv_results_
plt.figure(figsize=(10, 6))
plt.plot(param_grid['n_neighbors'], results['mean_test_score'], marker='o', label='Acurácia')
plt.title('Desempenho do kNN com diferentes valores de "k" (Acurácia)')
plt.xlabel('Número de Vizinhos (k)')
plt.ylabel('Acurácia Média (Validação Cruzada)')
plt.grid(True)
plt.legend()

# Gráfico para visualizar o desempenho do F1 Score e da Precisão
plt.figure(figsize=(10, 6))
plt.plot(param_grid['n_neighbors'], f1_scores, marker='o', label='F1 Score')
plt.plot(param_grid['n_neighbors'], precision_scores, marker='o', label='Precisão')
plt.title('Desempenho do kNN com diferentes valores de "k" (F1 Score e Precisão)')
plt.xlabel('Número de Vizinhos (k)')
plt.ylabel('Média (Validação Cruzada)')
plt.legend()
plt.grid(True)

plt.show()