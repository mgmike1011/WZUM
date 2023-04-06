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
import arff
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer

'''
Zad_1
Załaduj bazę danych Titanic, zapoznaj się z informacjami jakie zawiera. Wykorzystaj ładowanie jako pandas.DataFrame 
i np. meotdę info oraz describe. Sprawdź jakie wartości przyjmują poszczególne cechy.
'''
print('########### TODO 1 #############')
data = pd.read_csv('titanic.csv')
print(data.describe())
print(data.info())
# pclass - w jakiej klasie płynęli +
# survived - target - to czego szukamy
# name - imie nazwisko - do usunięcia
# sex - płeć +
# age - wiek +
# sibsp - liczba rodzeństwa i
# parch - numer parceli
# ticket - numer biletu
# fare - ile zapłaciła za bilet
# cabin - numer kabiny
# embarked - skąd wypłynęła
# boat - szalupa ratunkowa - do usunięcia
# body - numer zwłok wyłowionych z tytanika - do usunięcia
# home destination - skąd pochodziła

X = data.drop(['survived', 'boat', 'body', 'home.dest', 'cabin'], axis=1) #Usuwamy niepotrzebne kolumny, reszta zostaje
y = data['survived']
X = X.replace('?', np.NaN)
X['age'] = pd.to_numeric(X['age'], downcast='float') #Zamiana age na wartości typu float
'''
TODO 3
Podziel dane na zbiór treningowy i testowy. Ustalmy wartość random_state na 42, użyjmy stratyfikację, test_size na poziomie 0.1.
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

'''
TODO 4
Opracuj metodę, która dla danych testowych wyznacza szansę przeżycia (np. w sposób losowy). 
Jaką wartość uzyskałeś przy losowaniu szansy przeżycia z równym prawdopodobieństwem? Czy da się to poprawić?
'''
print('########### TODO 4 #############')
print(f'Szanse na przezycie: {sum(y_train)/len(y_train)*100}')
def basic_solution(person):
    return np.random.random() < 0.38

print(f'Przykład: {basic_solution(X_train.iloc[0])}')

results = []
for person in X_test.iterrows():
    results.append(basic_solution(person))
print(f'Skuteczność dla losowego: {sum(results == y_test)/len(results)*100} %')

'''
TODO 5
Przeszkuja bazę danych pod kątem brakujących wartości. Wyznacz ile dokładnie brakuje.
'''
print('########### TODO 5 #############')
import missingno as msno
# Biblioteka domyślne szuka NaN u nas one są jako '?', musimy je zastąpić
X = X.replace('?', np.NaN)
msno.matrix(X)
plt.show()
# Brak wartości w Cabin i Age, odrzucamy Cabin
# Uzupełniamy wiek


'''
TODO 6
Bazując na wiedzy z poprzednich zajęć spróbuj uzupełnić brakujące wartości
'''
print('########### TODO 6 #############')
# Tworzymy średnie grupy wiekowe dla pasaera i jego płci
ages = X_train.groupby(['sex', 'pclass'])['age'].median() #chcemy pogrupować według płci oraz klasy i kondensujemy za pomocą mediamy
print(ages)
for row, passenger in X_train.loc[np.isnan(X_train['age'])].iterrows():
    X_train['age'].iloc[row] = ages[passenger['sex']][passenger['pclass']]
for row, passenger in X_test.loc[np.isnan(X_test['age'])].iterrows():
    X_test['age'].iloc[row] = ages[passenger['sex']][passenger['pclass']]
