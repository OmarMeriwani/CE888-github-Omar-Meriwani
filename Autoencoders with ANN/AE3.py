from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import winsound
import keras.initializers
from sklearn.preprocessing import LabelEncoder
import pickle
import seaborn
from matplotlib import pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
np.random.seed(1337)

test = 'autism'
tobestored = False
if test == 'diabetes':
    '''
    ===================== DIABETES ======================'''
    df = pd.read_csv('diabetes.csv',header=0)
    X = df.iloc[:, 0:7].values
    y = df.iloc[:, 8].values

if test == 'heartattack':
    '''
    ===================== HEART-ATTACK ======================
    '''
    df = pd.read_csv('heart-attack.csv',header=0)
    X = df.iloc[:, [0,1,2,3,4,5,6,7,8,9,13]].values
    X = [k for k in X if '?' not in [m for m in k]]
    X = [[float(j) for j in i] for i in X]
    X = np.array(X)
    df = pd.DataFrame(X)
    y = X[:, 10]
    X = X[:, [0,1,2,3,4,5,6,8,9]]
if test == 'autism':

    '''
    ===================== AUTISM ======================
    Case_No,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,Age_Mons,Qchat-10-Score,Sex,Ethnicity,Jaundice,Family_mem_with_ASD,Who completed the test,"Class/ASD Traits "
    1,0,0,0,0,0,0,1,1,0,1,28,3,f,middle eastern,yes,no,family member,No
    '''
    df = pd.read_csv('Autism.csv',header=0)
    le = LabelEncoder()
    df.iloc[:,18] = le.fit_transform(df.iloc[:,18])
    df.iloc[:,17] = le.fit_transform(df.iloc[:,17])
    df.iloc[:,16] = le.fit_transform(df.iloc[:,16])
    df.iloc[:,15] = le.fit_transform(df.iloc[:,15])
    df.iloc[:,14] = le.fit_transform(df.iloc[:,14])
    df.iloc[:,13] = le.fit_transform(df.iloc[:,13])

    y = df.iloc[:, 18].values
    X = df.iloc[:, [1,2,3,4,5,6,15,14,13,11]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
dim = len(X_train[0])
from keras.models import Sequential
from keras.layers import Dense, Dropout

input_lyr = Input(shape=(dim,))
encoded = Dense(10, name='enc1', activation='relu')(input_lyr)
encoded = Dense(7, name='enc2',activation='relu')(encoded)
encoded = Dense(5, name='enc3',activation='relu')(encoded)

decoded = Dense(5, name='dec1',activation='relu')(encoded)
decoded = Dense(7, name='dec2',activation='relu')(decoded)
decoded = Dense(dim, name='dec3',activation='relu')(decoded)
AE = Model(input_lyr, decoded)
Encoder = Model(input_lyr,encoded)

AE.compile(optimizer='adam', loss='mean_squared_error')
AE.fit(X_train,X_train,epochs=1000,batch_size=100,shuffle=False,verbose=True)

print('Before',X_test)
predicted = AE.predict(X_test)
for u in range(0,len(X_test)):
    print(X_test[u])
    print( predicted[u])

frequency = 2000
duration = 1000
winsound.Beep(frequency, duration)
