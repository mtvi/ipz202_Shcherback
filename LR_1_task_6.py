import numpy as np
import pandas as pd
from sklearn import svm, naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Зчитуємо дані з файлу
data = pd.read_csv('data_multivar_nb.txt', header=None, names=['feature1', 'feature2', 'label'])

# Розділяємо дані на ознаки і цільову змінну
X = data[['feature1', 'feature2']]
y = data['label']

# Розділяємо дані на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Навчаємо SVM модель
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Навчаємо наївний байєсівський класифікатор
naive_bayes_classifier = naive_bayes.GaussianNB()
naive_bayes_classifier.fit(X_train, y_train)

# Прогнозуємо класи для тестових даних
svm_predictions = svm_classifier.predict(X_test)
naive_bayes_predictions = naive_bayes_classifier.predict(X_test)

# Оцінюємо якість SVM моделі
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions, average='weighted')
svm_recall = recall_score(y_test, svm_predictions, average='weighted')
svm_f1 = f1_score(y_test, svm_predictions, average='weighted')

# Оцінюємо якість наївного байєсівського класифікатора
naive_bayes_accuracy = accuracy_score(y_test, naive_bayes_predictions)
naive_bayes_precision = precision_score(y_test, naive_bayes_predictions, average='weighted')
naive_bayes_recall = recall_score(y_test, naive_bayes_predictions, average='weighted')
naive_bayes_f1 = f1_score(y_test, naive_bayes_predictions, average='weighted')

# Виводимо результати
print("SVM Метрики:")
print(f"Акуратність: {svm_accuracy}")
print(f"Точність: {svm_precision}")
print(f"Чутливість: {svm_recall}")
print(f"F1-показник: {svm_f1}")

print("\nНаївний байєсівський класифікатор Метрики:")
print(f"Акуратність: {naive_bayes_accuracy}")
print(f"Точність: {naive_bayes_precision}")
print(f"Чутливість: {naive_bayes_recall}")
print(f"F1-показник: {naive_bayes_f1}")
