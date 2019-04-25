import pandas as pd
import numpy as np
import winsound
import keras.initializers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import StratifiedKFold
from keras.layers import BatchNormalization
from sklearn.preprocessing import LabelEncoder
import seaborn
from matplotlib import pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
np.random.seed(1337)

test = 'heartattack'
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
    X = df.iloc[:, [1,2,3,4,15,14,13,11]].values


s=seaborn.heatmap(df.corr(),cmap='coolwarm', linewidths=.5)
s.set_yticklabels(s.get_yticklabels(),rotation=45,fontsize=5)
s.set_xticklabels(s.get_xticklabels(),rotation=45,fontsize=5)
if tobestored == True:
    plt.savefig('heatmap-'+ test +'.png')

WithAE = True
AEdim = 5
dim = len(X[0])

if WithAE == True:
    input_lyr = Input(shape=(dim,))
    encoded = Dense(15, name='enc1', activation='relu')(input_lyr)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(rate=0.1)(encoded)
    encoded = Dense(15, name='enc2',activation='relu')(encoded)
    encoded = Dense(13, name='enc3',activation='relu')(encoded)
    encoded = Dense(13, name='enc4',activation='relu')(encoded)
    encoded = Dense(AEdim, name='enc5',activation='relu')(encoded)

    decoded = Dense(AEdim, name='dec1',activation='relu')(encoded)
    decoded = Dense(13, name='dec2',activation='relu')(decoded)
    decoded = Dense(13, name='dec3',activation='relu')(decoded)
    decoded = Dense(15, name='dec4',activation='relu')(decoded)
    decoded = Dropout(rate=0.1)(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dense(15, name='dec5',activation='relu')(decoded)
    decoded = Dense(dim, name='dec6',activation='relu')(decoded)
    AE = Model(input_lyr, decoded)
    Encoder = Model(input_lyr,encoded)

    AE.compile(optimizer='adam', loss='mean_squared_error')
    AE.fit(X,X,epochs=2000,batch_size=100,shuffle=True,verbose=True)

model = Sequential()
model2 = Sequential()

sd_value = 0.01
print(y.shape, X.shape)

print(y.shape, X.shape)
#model.add(GaussianNoise(stddev=sd_value, input_shape=(dim,)))
#model.add(GaussianDropout(rate = 0.9))
model.add(Dense(14, activation='relu', input_shape=(dim,)))
model.add(Dense(2, name='dense5', activation='softmax'))

#model.add(GaussianNoise(stddev=sd_value))
#model.add(GaussianDropout(rate = 0.9))
model2.add(Dense(14, activation='relu', input_shape=(AEdim,)))
model2.add(Dense(2, name='dense5', activation='softmax'))

folds = 5
j = 0
score= []
score2= []
X2 = X
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model2.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

if WithAE == True:
    X2 = Encoder.predict(X)
for k in range(0,2):
    kf= StratifiedKFold(n_splits=folds, random_state=121, shuffle=True)
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        X_train2, X_test2 = X2[train_index], X2[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_train = keras.utils.np_utils.to_categorical(y_train)
        y_test = keras.utils.np_utils.to_categorical(y_test)
        model.fit(X_train,y_train,epochs=1000,batch_size=100,shuffle=False,verbose=0)
        model2.fit(X_train2,y_train,epochs=1000,batch_size=100,shuffle=False,verbose=0)

        score.append( model.evaluate(X_test, y_test))
        score2.append( model2.evaluate(X_test2, y_test))

avgacc = [acc for loss, acc in score]
avgloss = [loss for loss, acc in score]

avgacc2 = [acc for loss, acc in score2]
avgloss2 = [loss for loss, acc in score2]

avgacc =sum(avgacc) / len(avgacc)
avgloss = sum(avgloss) / len(avgloss)

avgacc2 =sum(avgacc2) / len(avgacc2)
avgloss2 = sum(avgloss2) / len(avgloss2)

print('Results', (avgacc2 * 100).__round__(2),'\t', (avgloss2 *100).__round__(2),'\t',(avgacc * 100).__round__(2), '\t',(avgloss*100).__round__(2))
