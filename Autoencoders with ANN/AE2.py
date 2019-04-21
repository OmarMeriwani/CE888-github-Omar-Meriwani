from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
import pandas as pd

encoding_dim = 8
input_lyr = Input(shape=(7,))

'''========== 1 =========='''
encoded = Dense(encoding_dim, activation='relu')(input_lyr)
decoded = Dense(7, activation='sigmoid')(encoded)
autoencoder = Model(input_lyr, decoded)

'''========== 2 =========='''
encoder = Model(input_lyr, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

'''========== data =========='''
df = pd.read_csv('diabetes.csv',header=0)
X = df.iloc[:, 0:7].values
y = df.iloc[:, 8].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

'''========== Model =========='''
autoencoder.fit(x_train, x_train,
                epochs=50,
                shuffle=True,
                validation_data=(x_test, x_test))
encoded_lyr = encoder.predict(x_test)
decoded_lyr = decoder.predict(encoded_lyr)
'''
for i in range(0,len(x_test)):
    print('BEFORE:',x_test[i])
    print('ENCODED:',encoded_lyr[i])
    print('DECODED:',decoded_lyr[i])
'''
visible = Input(shape=(7,))
hidden1 = Dense(10, activation='relu')(visible)
hidden2 = Dense(5, activation='relu')(hidden1)
hidden3 = Dense(10, activation='relu')(hidden2)
output = Dense(2, activation='sigmoid')(hidden3)
model = Model(inputs=visible, outputs=output)
model.compile(optimizer='RMSprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train,y_train,epochs=2000)
score = model.evaluate(decoded_lyr, y_test)
print(score)

