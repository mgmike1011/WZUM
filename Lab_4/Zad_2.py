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
import seaborn as sns

parameters = {
    'kernel': ('linear', 'rbf'),
    'C':[1, 10]
}
clf = GridSearchCV(svm.SVC(), parameters, cv=10)
clf.fit(X_train, y_train)

pvt = pd.pivot_table(
    pd.DataFrame(clf.cv_results_),
    values='mean_test_score',
    index='param_kernel',
    columns='param_C'
)

ax = sns.heatmap(pvt)
