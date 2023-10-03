import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, roc_auc_score



dataset = pd.read_csv('heart_cleveland_upload.csv')

# Separar os recursos (X) e o rótulo (y)
X = dataset.drop('condition', axis=1)
y = dataset['condition']


# Dividir o conjunto de dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizando os recursos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# GRID SEARCH usado para esolher os melhores hiperparametros depois
param_grid = {
    'max_depth': [None, 5, 10, 15, 20],  # Profundidade máxima da árvore
    'min_samples_split': [2, 5, 10],     # Número mínimo de amostras necessárias para dividir um nó
    'min_samples_leaf': [1, 2, 4]        # Número mínimo de amostras em uma folha
}
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)


best_max_depth = grid_search.best_params_['max_depth']
best_min_samples_split = grid_search.best_params_['min_samples_split']
best_min_samples_leaf = grid_search.best_params_['min_samples_leaf']

# Treinando o classificador de Árvore de Decisão com os melhores hiperparâmetros
tree_classifier = DecisionTreeClassifier(max_depth=best_max_depth, min_samples_split=best_min_samples_split, min_samples_leaf=best_min_samples_leaf)
tree_classifier.fit(X_train, y_train)

# precisao no conjunto y de testes
y_pred = tree_classifier.predict(X_test)

# Avaliando o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)

# Calculando o coeficiente de Gini
y_prob = tree_classifier.predict_proba(X_test)[:, 1]
gini_coefficient = 2 * roc_auc_score(y_test, y_prob) - 1

# Exibir métricas
print(f'Acurácia: {accuracy:.2f}')
print(f'Coeficiente de Gini: {gini_coefficient:.2f}')

# Gráfico da Acurácia e Coeficiente de Gini
plt.figure(figsize=(10, 6))
plt.plot(['Acurácia', 'Coeficiente de Gini'], [accuracy, gini_coefficient], marker='o')
plt.title('Métricas de Desempenho')
plt.xlabel('Métrica')
plt.ylabel('Valor')
plt.grid(True)
plt.show()

# Criar um gráfico de barras dos melhores hiperparâmetros
plt.figure(figsize=(10, 6))
plt.bar(['max_depth', 'min_samples_split', 'min_samples_leaf'], [best_max_depth, best_min_samples_split, best_min_samples_leaf])
plt.title('Melhores Hiperparâmetros Encontrados pelo Grid Search')
plt.xlabel('Hiperparâmetro')
plt.ylabel('Valor')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(15, 10))
plot_tree(tree_classifier, filled=True, feature_names=X.columns.tolist(), class_names=['0', '1'])
plt.show()
