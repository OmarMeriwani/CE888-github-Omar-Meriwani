import keras
from keras import layers
from keras import backend as K
from keras.models import Model
import keras_preprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense


df = pd.read_csv('diabetes.csv',header=0)
X = df.iloc[:, 0:7].values
y = df.iloc[:, 8].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model = Sequential([Dense(4, input_dim=7), Dense(4)])
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train,y_train,epochs=400)

score = model.evaluate(X_test, y_test)
print(score)