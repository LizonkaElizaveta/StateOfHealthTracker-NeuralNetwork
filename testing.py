# first neural network with keras tutorial
import random

import matplotlib.pyplot as plt
import numpy

from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.models import Sequential
from numpy import loadtxt


# load the dataset
dataset = loadtxt('data/Random_patients.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:, 0:7]
numpy.random.shuffle(X)

y = dataset[:, 8]
# define the keras model
model = Sequential()
model.add(Dense(random.randint(3,64), input_dim=7, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['mse', 'acc'])
# fit the keras model on the dataset
history = model.fit(X, y, epochs=30, validation_split=0.1, batch_size=5000, verbose=1)

print(model.summary())

# evaluate the keras model
_, accuracy, _ = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy * 100))
# make class predictions with the model
predictions = model.predict_classes(X)
# summarize the first 5 case
for i in range(100):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))

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
