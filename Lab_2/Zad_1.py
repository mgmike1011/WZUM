'''
Otwórz swój kod z TODO 2 z Lab01 i wykorzystując wiedzę i kod z poprzednich laboratoriów:
    Stwórz klasyfikator SVC i wytrenuj (metoda fit) go na pięciu zdjęciach z bazy danych.
    Przetestuj klasyfikator na jednym z wybranych zdjęć (metoda predict), wyświetl wynik.
'''
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
# print(digits['DESCR'])

X, y = digits['data'], digits['target'] #można tez digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

clf = SVC()
clf.fit(X_train, y_train)

# pred = clf.predict(X_test)

train_score = clf.score(X_train, y_train)
print(f'train_score: {train_score}')

test_score = clf.score(X_test, y_test) # dokona predykcji automatycznie i porówna z wynikami, zwróci % skuteczności klasyfikatora
print(f'test_score: {test_score}')
