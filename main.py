# first neural network with keras tutorial
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from numpy import loadtxt

# load the dataset
dataset = loadtxt('data/data.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:, 0:17]
y = dataset[:, 4]
# define the keras model
model = Sequential()
model.add(Dense(64, input_dim=17, activation='relu'))
model.add(Dense(32, input_dim=14, activation='relu'))
model.add(Dense(16, input_dim=12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'acc'])
# fit the keras model on the dataset
history = model.fit(X, y, epochs=150, validation_split=0.25, batch_size=10, verbose=0)

print(model.summary())

# evaluate the keras model
_, accuracy, _ = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy * 100))
# make class predictions with the model
predictions = model.predict_classes(X)
# summarize the first 5 case
for i in range(12):
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
