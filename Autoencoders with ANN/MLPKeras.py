from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
from keras import optimizers
import logging
from keras.layers import Activation
import os
visible = Input(shape=(7,))
hidden1 = Dense(10, activation='relu')(visible)
hidden2 = Dense(5, activation='relu')(hidden1)
hidden3 = Dense(10, activation='relu')(hidden2)
output = Dense(2, activation='sigmoid')(hidden3)

model = Model(inputs=visible, outputs=output)

# summarize layers
print(model.summary())
model.compile(optimizer='RMSprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# plot graph
df = pd.read_csv('diabetes.csv',header=0)
X = df.iloc[:, 0:7].values
y = df.iloc[:, 8].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
model.fit(X_train,y_train,epochs=2000)

score = model.evaluate(X_test, y_test)
print(score)