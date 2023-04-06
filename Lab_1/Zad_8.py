from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('training_data.txt', header=None, names=["charging", "watching"])
print(df)


plt.scatter(df['charging'], df['watching'])
plt.show()