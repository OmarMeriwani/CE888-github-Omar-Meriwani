from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import winsound
import keras.initializers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder
import pickle
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

model = MLPClassifier(hidden_layer_sizes=(14, 14, 14, 14, 14, 14, 14), activation='relu', solver='lbfgs',learning_rate='adaptive')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
'''sc = RobustScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
print(len(X))
max = 0
s = 0
for i in range(0,200):
    model.fit(X_train,y_train)
    s =  model.score(X_test,y_test)
    if s > max:
        if tobestored == True:
            pickle.dump(model,open('mlp'+test+'.sav','wb'))
        max = s
        print(s)'''

'''visible = Input(shape=(len(X_train[0]),))
hidden1 = Dense(14, activation='relu')(visible)
hidden2 = Dense(14, activation='relu')(hidden1)
hidden3 = Dense(14, activation='relu')(hidden2)
hidden4 = Dense(14, activation='relu')(hidden3)
hidden5 = Dense(14, activation='relu')(hidden4)
hidden6 = Dense(14, activation='relu')(hidden5)
output = Dense(2, activation='sigmoid')(hidden6)
model = Model(inputs=visible, outputs=output)'''
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import StratifiedKFold
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import GaussianNoise
from keras.layers import GaussianDropout
WithAE = True
AEdim = 5
dim = len(X[0])

#sc = MinMaxScaler()
#X = sc.fit_transform(X)

if WithAE == True:
    input_lyr = Input(shape=(dim,))
    encoded = Dense(15, name='enc1', activation='relu')(input_lyr)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(rate=0.1)(encoded)
    #encoded = GaussianNoise(stddev=0.05)(encoded)
    encoded = Dense(15, name='enc2',activation='relu')(encoded)
    #encoded = GaussianDropout(rate=0.5)(encoded)
    encoded = Dense(13, name='enc3',activation='relu')(encoded)
    encoded = Dense(13, name='enc4',activation='relu')(encoded)
    #encoded = Dense(15, name='enc41',activation='relu')(encoded)
    #encoded = Dropout(rate=0.1)(encoded)
    encoded = Dense(AEdim, name='enc5',activation='relu')(encoded)
    #encoded = Activation('softmax')(encoded)

    decoded = Dense(AEdim, name='dec1',activation='relu')(encoded)
    #encoded = Dropout(rate=0.1)(encoded)
    #decoded = Dense(15, name='dec21',activation='relu')(decoded)
    decoded = Dense(13, name='dec2',activation='relu')(decoded)
    decoded = Dense(13, name='dec3',activation='relu')(decoded)
    #decoded = GaussianDropout(rate=0.5)(decoded)
    decoded = Dense(15, name='dec4',activation='relu')(decoded)
    #decoded = GaussianNoise(stddev=0.05)(decoded)
    decoded = Dropout(rate=0.1)(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dense(15, name='dec5',activation='relu')(decoded)
    decoded = Dense(dim, name='dec6',activation='relu')(decoded)
    AE = Model(input_lyr, decoded)
    Encoder = Model(input_lyr,encoded)

    AE.compile(optimizer='adam', loss='mean_squared_error')
    AE.fit(X,X,epochs=2000,batch_size=100,shuffle=True,verbose=True)

#encoding_dim = 8
#if WithAE == True:
#    dim = AEdim

model = Sequential()
model2 = Sequential()
'''if test == 'diabetes':
    model.add(Dense(14, name='dense1', activation='relu', input_shape=(dim,)))
    model.add(Dense(2, name='dense2', activation='softmax'))

    model2.add(Dense(14, name='dense1', activation='relu', input_shape=(AEdim,)))
    model2.add(Dense(2, name='dense2', activation='softmax'))
if test == 'heartattack':
    model.add(Dense(14, name='dense1',activation='relu', input_shape=(dim,)))
    model.add(Dense(2, name='dense5', activation='softmax'))

    model2.add(Dense(14, name='dense1',activation='relu', input_shape=(AEdim,)))
    model2.add(Dense(2, name='dense5', activation='softmax'))

if test == 'autism':'''
#Threshold: 1.1001
sd_value = 0.01
samples_reduction = 0.5
print(y.shape, X.shape)
#X = X[:int(len(X) * samples_reduction)]
#y = y[:int(len(y) * samples_reduction)]
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

print('Without AE: ', (avgacc2 * 100).__round__(2),'\t', (avgloss2 *100).__round__(2),'\t',(avgacc * 100).__round__(2), '\t',(avgloss*100).__round__(2))
print('With AE: ',avgacc2, avgloss2)
#model = pickle.load(open('mlp-diabetes2.sav','rb'))
#score = model.score(X_test,y_test)
#print(score)
frequency = 2000
duration = 500
winsound.Beep(frequency, duration)
