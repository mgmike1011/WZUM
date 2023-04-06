from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

digits = datasets.load_digits()
print(digits['data'])
# target - to jakie są prawdziwe wyjścia
# Images a data - data dane spłaszczone, images dane prawdziwe, images do naszej interpretacji, dane

# Wyświetlenie danego obrazka
plt.imshow(digits['images'][0])
print(digits['target'][0])
plt.show()

# Wyświetlić po kilka elementów dla każdej z dostępnyc klas
fig, axs = plt.subplots(len(digits['target_names']), 5) #Po 5 instancji każdej klasy
for class_n in digits['target_names']:
    for col in range(5):
        axs[class_n][col].imshow(digits['images'][digits['target'] == class_n][col], cmap='gray_r') #Wyswietlamy tylko te których klasa jest zgodna, cmap = zamienia kolory na szare _r odwrócone
        axs[class_n][col].axis('off')
plt.show()

# Podział dataesu na zbiory
X, y = digits['data'], digits['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) # test_size - ile %na zbiór testowy, random_state -
print(len(X))
print(len(X_train))
print(len(X_test))

