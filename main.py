# first neural network with keras tutorial
import matplotlib.pyplot as plt
import numpy

import tensorflow as tf

Dense = tf.keras.layers.Dense
Sequential = tf.keras.models.Sequential
from numpy import loadtxt

# load the dataset
dataset1 = loadtxt('data/patient_health.csv', delimiter=',')
dataset2 = loadtxt('data/patient_sick.csv', delimiter=',')
# split into input (X) and output (y) variables
# X = dataset[:1998, 0:3]
X = numpy.stack([dataset1, dataset2])
# numpy.random.shuffle(X)

y = numpy.array([0., 1.])
# y = dataset[:, 9]
# define the keras model
model = Sequential()
model.add(tf.keras.layers.GRU(128))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['mse', 'acc'])
# fit the keras model on the dataset
history = model.fit(X, y, epochs=10, batch_size=100, verbose=0)

print(model.summary())

# evaluate the keras model
# _, accuracy, _ = model.evaluate(X, y)
# print('Accuracy: %.2f' % (accuracy * 100))
# make class predictions with the model
predictions = model.predict(X)
print('Predictions:', predictions)

# summarize the first 5 case
for i in range(100):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], i / 27000))

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

# angles_input = tf.keras.layers.InputLayer(input_shape=(None, None, 3))
# pills_received = tf.keras.layers.InputLayer(input_shape=(None, 1))
#
# rnn_layer = tf.keras.layers.GRU(128)(angles_input)
# concat_layer = tf.keras.layers.Concatenate([rnn_layer, pills_received])
#
# dense = tf.keras.layers.Dense(16, 'relu')
# output = tf.keras.layers.Dense(1, 'sigmoid')
#
# model_ = tf.keras.models.Model(inputs=[angles_input, pills_received], outputs=[output])
