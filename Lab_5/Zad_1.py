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
    X_train.loc[row, 'age'] = ages[passenger['sex']][passenger['pclass']]
for row, passenger in X_test.loc[np.isnan(X_test['age'])].iterrows():
    X_test.loc[row, 'age'] = ages[passenger['sex']][passenger['pclass']]

y_train = y_train[X_train['fare'].notna()]
X_train = X_train.dropna()


'''
TODO 7
Wykorzystaj dostępne w sklearn elementy (np. LabelEncoder) lub przygotuj własny i zamień wszystkie cechy na liczbowe.
'''
print('########### TODO 7 #############')
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_train['sex'] = le.fit_transform(X_train['sex']) #female 0, male 1
X_test['sex'] = le.transform(X_test['sex'])
le_embarked = LabelEncoder()
X_train['embarked'] = le_embarked.fit_transform(X_train['embarked'])
X_test['embarked'] = le_embarked.transform(X_test['embarked'])

X_train['pclass'].loc[X_train['pclass'] == 3] = 0 #Dodatkowa zamiana na potrzeby ML, żeby dodac 0
X_test['pclass'].loc[X_test['pclass'] == 3] = 0
X_train['pclass'].loc[X_train['pclass'] == 2] = 0.5 #Dodatkowa zamiana na potrzeby ML, żeby dodac 0
X_test['pclass'].loc[X_test['pclass'] == 2] = 0.5

X_train['family_size'] = X_train['sibsp'] + X_train['parch'] # łączymy w jedną kolumnę ile ma członków rodziny
X_train.drop(['name', 'ticket', 'sibsp', 'parch'], axis=1, inplace=True) #odrzucamy kolumny name i ticket
X_test['family_size'] = X_test['sibsp'] + X_test['parch']
X_test.drop(['name', 'ticket', 'sibsp', 'parch'], axis=1, inplace=True) #odrzucamy kolumny name i ticket i sibsp i parch

# Standaryzacja
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
# print(X_train)

'''
TODO 11
Przygotuj wykres przedstawiający korelację poszczególnych cech. Wymaga to połączenia cech z wartością oczekiwaną, np.:
'''
print('########### TODO 11 #############')
# X_combined = pd.concat([X_train, y_train.astype(float)], axis=1)
# sns.pairplot(X_combined.astype(float), vars=['pclass', 'age', 'sex', 'fare'], hue='survived')
# sns.heatmap(X_combined.corr(), annot=True, cmap="coolwarm")
plt.show()

'''
TODO 8
Przygotowane dane są gotowe do wykorzystania - wytrenuj wybrany klasyfikator i 
oceń go względem przygotowanego wcześniej rozwiązania bazowego.
'''
print('########### TODO 8 #############')
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
predicted = dtc.predict(X_test)
expected = y_test
cm = confusion_matrix(expected, predicted) # Wygenerowanie tablicy pomyłek
print(f'confusion_matrix: \n{cm}') # Wyświetlenie tablicy pomyłek
disp_cm = ConfusionMatrixDisplay(cm) # Graficzne wyświetlenie tablicy pomyłek
disp_cm.plot()
plt.show()
train_score = dtc.score(X_train, y_train)
print(f'train_score: {train_score}')