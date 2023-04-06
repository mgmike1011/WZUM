'''
Otwórz swój kod z TODO 5 z Lab01 i spróbuj wytrenować model. To samo zrób dla TODO 6 i TODO 7. Spróbuj różnych modeli.
'''
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt

dataset = datasets.fetch_olivetti_faces()
# print(dataset['DESCR'])

X, y = dataset['data'], dataset['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# clf = LinearSVC(verbose=True) #verbose - mówi co się dzieje podczas uczenia
clf = SVC()
clf.fit(X_train, y_train)

train_score = clf.score(X_train, y_train)
print(f'train_score: {train_score}')

test_score = clf.score(X_test, y_test)
print(f'test_score: {test_score}')

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
predicted = clf.predict(X_test)
expected = y_test
cm = confusion_matrix(expected, predicted)
print(f'confusion_matrix: \n{cm}') #Tablica pomyłek
disp_cm = ConfusionMatrixDisplay(cm, display_labels=np.unique(dataset.target)) #Graficzne wyświetlenie tablicy pomyłek
disp_cm.plot()
plt.show()