# first neural network with keras tutorial
import random

import matplotlib.pyplot as plt
import numpy
from keras import regularizers

from keras.layers import Dense, Dropout, LSTM, Embedding, GRU, Flatten
from keras.models import Sequential
from numpy import loadtxt


# load the dataset
dataset = loadtxt('data/Random_patients.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:, 0:7]
X = X.reshape(-1, 1, 7)

y = dataset[:, 8]
# define the keras model
model = Sequential()
model.add(GRU(100, input_shape=(1, 7), return_sequences=True))
model.add(LSTM(50, input_shape=(1, 7), return_sequences=True))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(20, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(16, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(16, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01)))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['mse', 'acc'])
# fit the keras model on the dataset
history = model.fit(X, y, epochs=10, validation_split=0.1, shuffle=True, batch_size=1000, verbose=1)

print(model.summary())

# evaluate the keras model
_, accuracy, _ = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy * 100))
# make class predictions with the model
predictions = model.predict(X)
# summarize the first 5 case
for i in range(100):
    print('%s => %f (expected %f)' % (X[i].tolist(), predictions[i], y[i]))

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
