import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Carregando o conjunto de dados 'heart_cleveland_upload.csv' em um DataFrame do Pandas
dataset = pd.read_csv('heart_cleveland_upload.csv')

# Etapa 1: Preparação dos Dados
# Separar as features (X) e o target (y).
X = dataset.drop('condition', axis=1)
y = dataset['condition']

# Dividir o conjunto de dados em conjuntos de treinamento e teste.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Etapa 2: Ajuste de Hiperparâmetros (n_estimators)
# Definir os valores a serem testados para n_estimators
param_grid = {
    # Altere os valores conforme necessário
    'n_estimators': [50, 100, 150, 200, 250]
}

# Criar o modelo Random Forest
rf_classifier = RandomForestClassifier(random_state=42)

# Configurar a busca em grade com validação cruzada
grid_search = GridSearchCV(
    estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)

# Executar a busca em grade no conjunto de treinamento
grid_search.fit(X_train, y_train)

# Obter os resultados da busca em grade
results = grid_search.cv_results_

# Extrair os valores de n_estimators e a acurácia média associada
n_estimators_values = results['param_n_estimators'].data
accuracy_values = results['mean_test_score']

# Etapa 3: Geração do Gráfico
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_values, accuracy_values, marker='o')
plt.xlabel('Número de Árvores (n_estimators)')
plt.ylabel('Acurácia Média')
plt.title('Acurácia Média vs. Número de Árvores (n_estimators)')
plt.grid(True)
plt.show()

# Obter o número ideal de árvores (n_estimators)
best_n_estimators = grid_search.best_params_['n_estimators']

# Etapa 4: Treinamento do Modelo com o Melhor n_estimators
# Criar e treinar o modelo Random Forest com o número ideal de árvores
rf_classifier_best = RandomForestClassifier(
    n_estimators=best_n_estimators, random_state=42)
rf_classifier_best.fit(X_train, y_train)

# Etapa 5: Avaliação do Modelo
# Fazer previsões no conjunto de teste.
y_pred = rf_classifier_best.predict(X_test)

# Avaliar o desempenho do modelo com o melhor n_estimators.
accuracy = accuracy_score(y_test, y_pred)
print(f'Melhor n_estimators: {best_n_estimators}')
print(f'Acurácia do modelo: {accuracy:.2f}')

# Exibir a matriz de confusão para uma análise mais detalhada.
conf_matrix = confusion_matrix(y_test, y_pred)
print('Matriz de Confusão:')
print(conf_matrix)

# Gerar um relatório de classificação com métricas adicionais.
class_report = classification_report(y_test, y_pred)
print('Relatório de Classificação:')
print(class_report)


# Etapa 6: Visualização da Importância das Features
# Obter a importância das features do modelo treinado
feature_importances = rf_classifier_best.feature_importances_

# Criar um DataFrame para facilitar a visualização
importance_df = pd.DataFrame(
    {'Feature': X.columns, 'Importance': feature_importances})

# Ordenar as features pela importância
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plotar um gráfico de barras com as importâncias das features
plt.figure(figsize=(10, 6))
plt.bar(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Features')
plt.ylabel('Importância')
plt.title('Importância das Features')
plt.xticks(rotation=45)
plt.show()
