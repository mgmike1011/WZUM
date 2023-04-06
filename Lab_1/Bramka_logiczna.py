import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# AND
X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]
y = [0, 0, 0, 1]

clf = DecisionTreeClassifier()
clf.fit(X, y)

print(clf.predict([[1, 1]]))   # Sprawdź sam(a) jakie będą wyniki dla innych danych wejściowych.

plot_tree(clf)
plt.show()