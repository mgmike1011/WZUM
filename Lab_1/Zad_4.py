from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset = datasets.fetch_olivetti_faces()
print(dataset['DESCR'])
#40 klas, 400 próbek, Dimensionality 4096 Features real, between 0 and 1


X, y = dataset['data'], dataset['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Stratify - podzieli tak, eby y były równo reprezentowane
for i in range(10):
    print(i, ': ', sum(y_test == i))
