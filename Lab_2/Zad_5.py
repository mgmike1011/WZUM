'''
Otwórz swój kod z TODO 8 z poprzednich zajęć i:
    Wytrenuj model umożliwiający predykcję czasu pracy na baterii. Tym razem będzie to zadanie regresji.
    Na wykresie zaprezentuj wynik - rzeczywisty czas pracy dla danych testowych vs wynik predykcji.
'''
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('training_data.txt', header=None, names=["charging", "watching"])
# print(df)
X, y = df['charging'], df['watching']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train = np.array(X_train).reshape(-1, 1)
X_test = np.array(X_test).reshape(-1, 1)

plt.scatter(df['charging'], df['watching'])


pf = PolynomialFeatures(degree=10)
pf.fit(X_train)
X_train = pf.transform(X_train)

mdl = LinearRegression()
# mdl = DecisionTreeRegressor()
mdl.fit(X_train, y_train)

print(mdl.score(pf.transform(X_test), y_test))

plt.scatter(X_test, mdl.predict(pf.transform(X_test)))
plt.show()