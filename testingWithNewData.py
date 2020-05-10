import matplotlib.pyplot as plt
from keras import regularizers

from keras.layers import Dense, Dropout, LSTM, GRU, Flatten
from keras.models import Sequential
from numpy import loadtxt

# load the dataset
dataset = loadtxt('data/Testing_patients.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:, 0:7]
# X = X.reshape(-1, 1, 7)

y = dataset[:, 8]
# define the keras model
model = Sequential()
# model.add(LSTM(150, input_shape=(1, 7), return_sequences=True, dropout=0.1))
# model.add(Flatten())
# model.add(Dense(256))
# model.add(Dropout(0.2))
# model.add(Dense(128))
# model.add(Dropout(0.2))
# model.add(Dense(64))
# model.add(Dense(16))
# model.add(Dense(1, activation="sigmoid"))
# model.compile(loss='mean_squared_error',
#               optimizer='rmsprop',
#               metrics=['mse', 'acc'])
# model.add(Flatten())
model.add(Dense(199))
model.add(Dropout(0.13617411274661234))
model.add(Dense(711))
model.add(Dense(1, activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'acc'])
history = model.fit(X, y, epochs=5, validation_split=0.25, shuffle=True, batch_size=5000, verbose=0)

print(model.summary())

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

# evaluate the keras model
# _, accuracy, _ = model.evaluate(X, y, verbose=0)
# print('Accuracy: %.2f' % (accuracy * 100))
# # make predictions with the model
# predictions = model.predict(X[:100])
# # summarize the first 100 case
# for i in range(100):
#     print('%s => %f (expected %f)' % (X[i].tolist(), predictions[i], y[i]))

dataset = loadtxt('data/Testing_validation_patients.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:, 0:7]
# X = X.reshape(-1, 1, 7)

y = dataset[:, 8]
# evaluate the keras model
_, accuracy, _ = model.evaluate(X, y, verbose=0)
print('Accuracy: %.2f' % (accuracy * 100))
# make predictions with the model
predictions = model.predict(X)
# summarize the first 100 case
for i in range(100):
    print('%s => %f (expected %f)' % (X[i].tolist(), predictions[i], y[i]))
