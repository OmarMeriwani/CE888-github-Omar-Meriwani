import numpy as np
import sklearn
import pandas as pd
print(sklearn.__version__)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
bank = pd.read_csv('bank-additional-full.csv', sep=';',header=0)
print(bank.keys())
print(bank.shape)
bank.dropna()
le = LabelEncoder()
bank.pop('duration')
bank['job'] = le.fit_transform(bank['job'])
bank['marital'] = le.fit_transform(bank['marital'])
bank['education'] = le.fit_transform(bank['education'])
bank['default'] = le.fit_transform(bank['default'])
bank['housing'] = le.fit_transform(bank['housing'])
bank['loan'] = le.fit_transform(bank['loan'])
bank['contact'] = le.fit_transform(bank['contact'])
bank['month'] = le.fit_transform(bank['month'])
bank['day_of_week'] = le.fit_transform(bank['day_of_week'])
bank['poutcome'] = le.fit_transform(bank['poutcome'])
X = bank.values[:,0:19]
print(X)
Y = bank.values[:, 19]
print(Y)

Y2 = [0 if a == 'no' else 1 for a in Y]

x_train,x_test,y_train,y_test = train_test_split(X,Y2, test_size=0.3, random_state=100)
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
fig2 = plt.boxplot(X)
