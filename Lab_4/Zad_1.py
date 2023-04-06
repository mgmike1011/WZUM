import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.io.arff import loadarff

# Wczytanie danych
raw_data = loadarff('dataset_37_diabetes.arff')
diabetes = pd.DataFrame(raw_data[0])
# print(diabetes.describe())

# Podział danych
X = diabetes.iloc[:, :-1]
y = diabetes.iloc[:, -1]
y = [1 if i == b'tested_positive' else 0 for i in y]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
# print(y)

import seaborn as sns
# sns.histplot(diabetes['mass'])
sns.boxplot(X_train['mass'])
plt.show()
plt.scatter(X_train['mass'], X_train['plas']) # Sprawdzenie korelacji danych
plt.show()

zscore = abs((X_train - X_train.mean())/ X_train.std()) #Jak bardzo nasze dane odstają od datasetu
print(f'ZScore: {zscore}')
# # Chcemy odrzucić wartości, odstające za bardzo
# y_train = y_train[(zscore <3).all(axis=1)]
# X_train = X_train.loc[(zscore <3).all(axis=1)]


# Próba dowolnym klasyfikatorem - dane dalej z brakami:
classifier1 = SVC()
classifier2 = RandomForestClassifier()
classifier3 = DecisionTreeClassifier()

classifier1.fit(X_train, y_train)
classifier2.fit(X_train, y_train)
classifier3.fit(X_train, y_train)

print(f'SVC: {classifier1.score(X_test, y_test)}')
print(f'Random Forest: {classifier2.score(X_test, y_test)}')
print(f'Decision Tree: {classifier3.score(X_test, y_test)}')

# # Usuwanie wierszów z zerami
# print(diabetes.columns)
for col in ['plas', 'pres', 'skin', 'insu', 'mass', 'pedi']: # Zamieniamy wwszytskie zera w tych kolumnach na Nan
    X_train[col].loc[X_train[col] == 0] = np.NaN
#
y_train = np.array(y_train)[X_train.notna().all(axis=1)] # Usuwamy wszytskie wiersze gdzie w X_train jest Nan
X_train = X_train.dropna() # Usuwamy wszytskie wiersze z Nan
#
# # Ponowna próba z klasyfikatorami na wartościach usuniętych
# classifier1_ = SVC()
# classifier2_ = RandomForestClassifier()
# classifier3_ = DecisionTreeClassifier()
#
# classifier1_.fit(X_train, y_train)
# classifier2_.fit(X_train, y_train)
# classifier3_.fit(X_train, y_train)
#
# print(f'SVC _: {classifier1_.score(X_test, y_test)}')
# print(f'Random Forest _: {classifier2_.score(X_test, y_test)}')
# print(f'Decision Tree _: {classifier3_.score(X_test, y_test)}')
# # Wynik spadł w stosunku do poprzedniego, bo mamy mniej danych

# Zamiana Nan z pomocą inputera
from sklearn.impute import SimpleImputer, KNNImputer
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer.fit(X_train)
# X_train = imputer.transform(X_train)
# print(X_train)

mdl = Pipeline([
        ('Simple_imputer', KNNImputer()),
        ('standard_scaler', StandardScaler()),
        ('min_max_scaler', MinMaxScaler()),
        ('classifier', RandomForestClassifier())
    ])
mdl.fit(X_train, y_train)
print(mdl.score(X_test, y_test))


parameters = {
    'kernel': ('linear', 'rbf'),
    'C': [1, 10]
}
# clf = GridSearchCV(SVC(), parameters, cv=10)
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
# clf = VotingClassifier([
#     ('SVC', SVC()),
#     ('RF', RandomForestClassifier()),
#     ('KNN', KNeighborsClassifier())
# ])
clf = StackingClassifier([
    ('SVC', SVC()),
    ('RF', RandomForestClassifier()),
    ('KNN', KNeighborsClassifier())
], SVC())
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
exit()

pvt = pd.pivot_table(
    pd.DataFrame(clf.cv_results_),
    values='mean_test_score',
    index='param_kernel',
    columns='param_C'
)

ax = sns.heatmap(pvt)
plt.show()

import pickle
with open('best_clf.pkl', 'wb') as file:
    pickle.dump(clf.best_estimator_, file)
    # clf = pickle.load(file) # DO wczytania

