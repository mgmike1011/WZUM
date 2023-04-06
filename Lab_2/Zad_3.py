from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

X_train =[[15, 12, "brak"],
          [20, 15, "brak"],
          [10, 3, "małe"],
          [15, 8, "brak"],
          [1, 9, "średnie"],
          [23, 3, "brak"],
          [18, 12, "duże"],
          [17, 11, "małe"],
          [19, 19, "małe"],
          [25, 10, "średnie"]] #
y_train = [1, 1, 1, 0, 0, 0, 0, 1, 1, 0]

opady_d = {"brak": 0, "małe": 1, "średnie": 2, "duże": 3} # Ostatnia kolumne trzeba zmienic ze str na float

for i, m in enumerate(X_train):
    m[2] = opady_d[m[2]] #zamiana z wartości str do float
print(X_train)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

print(clf.predict([[-5, 10, 3]]))

colors = ['r', 'g'] #czerwony - nie idziemy, 1 - idziemy
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for data, label in zip(X_train, y_train):
    ax.scatter(data[0], data[1], data[2], marker='o', c=colors[label])
ax.set_xlabel('temperatura')
ax.set_ylabel('godzina')
ax.set_zlabel('opady')
plt.show()
