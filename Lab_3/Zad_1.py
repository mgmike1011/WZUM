'''
Wczytajmy kolejny, tym razem większy, i wbudowany w bibliotekę scikit-learn zbiór danych – Iris dataset,
które jest zbiorem pomiarów pewnych charakterystycznych wielkości kwiatów kosaćca (więcej informacji: https://en.wikipedia.org/wiki/Iris_flower_data_set).
'''
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import json

df_data, df_target = load_iris(as_frame=True, return_X_y=True)
# X - sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# y - 'setosa', 'versicolor', 'virginica'

# df_data = df_data[['petal length (cm)', 'petal width (cm)']] #Tylko do mlxtend plot regions

X_train, X_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.2, random_state=42, stratify=df_target)

'''
Następnie sprawdź, jakie jest procentowe rozłożenie poszczególnych klas w zbiorze treningowym i testowym. Dobrze było by,
 aby dystrybucje klas próbek w tych zbiorach były identyczne – zmodyfikuj poprzedni kod tak, żeby dane po podziale 
 spełniały ten warunek (wskazówka: słówko stratify).
'''
print(f'Rozkład w procentach :{y_train.value_counts() / len(y_train)*100}')

# skaler = MinMaxScaler()
# skaler.fit(X_train)
# X_train = skaler.transform(X_train) #Normalizujemy wartości od 0 do 1

# standard_skaler = StandardScaler()
# standard_skaler.fit(X_train)
# X_train = standard_skaler.transform(X_train) #Standaryzacja


# Pipeline
clsf = [
    LinearSVC,
    SVC,
    RandomForestClassifier,
    DecisionTreeClassifier
]

results = dict()

for clf in clsf:
    mdl = Pipeline([
        ('standard_scaler', StandardScaler()),
        ('min_max_scaler', MinMaxScaler()),
        ('classifier', clf())
    ])
    mdl.fit(X_train, y_train)
    results[str(clf.__name__)] = mdl.score(X_test, y_test)
# X_train, y_train = mdl.transform(X_train, y_train) #Po dodaniu klasyfikatora nie ma już metody transform
print(results)
with open('wyniki.json', 'w') as file:
    json.dump(results, file, indent=True)

# from mlxtend.plotting import plot_decision_regions
# plot_decision_regions(np.array(X_train), np.array(y_train), clf=mdl, legend=1)
# plt.show()

# Wizualizujemy tylko dwie pierwsze cechy – aby móc je przedstawić bez problemu w 2D.
# plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1])
# plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1])
# plt.scatter(X_train[y_train == 2, 0], X_train[y_train == 2, 1])
# plt.legend(['0', '1', '2'])
# plt.axvline(x=0)
# plt.axhline(y=0)
# plt.title('Iris sepal features')
# plt.xlabel('sepal length (cm)')
# plt.ylabel('sepal width (cm)')
# plt.show()

