# # first neural network with keras tutorial
# from numpy import loadtxt
# from keras.models import Sequential
# from keras.layers import Dense,Dropout
# import tensorflow as tf
#
#
# # load the dataset
# dataset = loadtxt('data/Random2.csv', delimiter=',')
# # split into input (X) and output (y) variables
# X = dataset[:,0:8]
# y = dataset[:,8]
#
# model_ = tf.keras.models.Sequential([
#                                      tf.keras.layers.LSTM(128,input_shape=(1, 3), return_sequences=True),
#                                      tf.keras.layers.LSTM(128, input_shape=(1, 3),  return_sequences=True),
#                                      # tf.keras.layers.Dense(64, activation='relu'),
#                                      # tf.keras.layers.Dense(32, activation='relu'),
#                                      tf.keras.layers.Dense(1, activation='sigmoid'),
# ])
# model_.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['mse', 'acc'])
# history = model_.fit(X, y, epochs=20, batch_size=100, verbose=2, validation_split=0.2)
#
# predictions = model_.predict(X)
# print('Predictions:', predictions)
# #
# # # define the keras model
# # model = Sequential()
# # # model.add(Dense(12, input_dim=8, activation='relu'))
# # # model.add(Dense(8, activation='relu'))
# # # model.add(Dense(1, activation='sigmoid'))
# #
# # model.add(Dense(128, input_dim=8, activation='relu'))
# # model.add(Dense(30, activation='relu'))
# # model.add(Dense(10, activation='relu'))
# # model.add(Dense(1, activation='sigmoid'))
# #
# # model.compile(optimizer='adam',
# #               loss='binary_crossentropy',
# #               metrics=['mse', 'acc'])
# # # compile the keras model
# # # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# #
# # # fit the keras model on the dataset
# # model.fit(X, y, epochs=20, batch_size=100, verbose=1, validation_split=0.2)
# #
# #
# # # evaluate the keras model
# # _, accuracy = model.evaluate(X, y)
# # print('Accuracy: %.2f' % (accuracy*100))
# # predictions = model.predict(X)
# # print('Predictions:', predictions)
# # for i in range(100):
# #     print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], i / 27000))
# #
# #
# # # import numpy as np
# # # from keras.utils import to_categorical
# # # from keras import models
# # # from keras import layers
# # # from keras.datasets import imdb
# # #
# # # (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
# # # data = np.concatenate((training_data, testing_data), axis=0)
# # targets = np.concatenate((training_targets, testing_targets), axis=0)
# #
# #
# # def vectorize(sequences, dimension=10000):
# #     results = np.zeros((len(sequences), dimension))
# #     for i, sequence in enumerate(sequences):
# #         results[i, sequence] = 1
# #     return results
# #
# #
# # data = vectorize(data)
# # targets = np.array(targets).astype("float32")
# # test_x = data[:10000]
# # test_y = targets[:10000]
# # train_x = data[10000:]
# # train_y = targets[10000:]
# #
# # model = models.Sequential()
# # # Input - Layer
# # model.add(layers.Dense(50, activation='relu', input_shape=(10000, )))
# # # Hidden - Layers
# # model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
# # model.add(layers.Dense(50, activation='relu'))
# # model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
# # model.add(layers.Dense(50, activation="relu"))
# # # Output- Layer
# # model.add(layers.Dense(1, activation="sigmoid"))
# # model.summary()
# # # compiling the model
# # model.compile(
# #  optimizer = "adam",
# #  loss = "binary_crossentropy",
# #  metrics = ["accuracy"]
# # )
# # results = model.fit(
# #  train_x, train_y,
# #  epochs= 2,
# #  batch_size = 500,
# #  validation_data = (test_x, test_y)
# # )
# # print("Test-Accuracy:", np.mean(results.history["val_acc"]))
# #
#
# #
# # # load the dataset
# # dataset1 = loadtxt('data/patient_health.csv', delimiter=',')
# # dataset2 = loadtxt('data/patient_sick.csv', delimiter=',')
# # # split into input (X) and output (y) variables
# # # X = dataset[:1998, 0:3]
# # X = numpy.stack([dataset1, dataset2])
# # # numpy.random.shuffle(X)
# #
# # y = numpy.array([0., 1.])
# # # y = dataset[:, 9]
# # # define the keras model
# # model = Sequential()
# # model.add(tf.keras.layers.GRU(128))
# # model.add(Dense(12, activation='relu'))
# # model.add(Dense(8, activation='relu'))
# # model.add(Dense(1, activation='sigmoid'))
# # # compile the keras model
# # model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['mse', 'acc'])
# # # fit the keras model on the dataset
# # history = model.fit(X, y, epochs=10, batch_size=100, verbose=0)
# #
# # print(model.summary())
# #
# # # evaluate the keras model
# # # _, accuracy, _ = model.evaluate(X, y)
# # # print('Accuracy: %.2f' % (accuracy * 100))
# # # make class predictions with the model
# # predictions = model.predict(X)
# # print('Predictions:', predictions)
# #
# # # summarize the first 5 case
# # for i in range(100):
# #     print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], i / 27000))
# #
# # # Plot training & validation accuracy values
# # plt.plot(history.history['acc'])
# # plt.plot(history.history['val_acc'])
# # plt.title('Model accuracy')
# # plt.ylabel('Accuracy')
# # plt.xlabel('Epoch')
# # plt.legend(['Train', 'Test'], loc='upper left')
# # plt.show()
# #
# # # Plot training & validation loss values
# # plt.plot(history.history['loss'])
# # plt.plot(history.history['val_loss'])
# # plt.title('Model loss')
# # plt.ylabel('Loss')
# # plt.xlabel('Epoch')
# # plt.legend(['Train', 'Test'], loc='upper left')
# # plt.show()
#
# # angles_input = tf.keras.layers.InputLayer(input_shape=(None, None, 3))
# # pills_received = tf.keras.layers.InputLayer(input_shape=(None, 1))
# #
# # rnn_layer = tf.keras.layers.GRU(128)(angles_input)
# # concat_layer = tf.keras.layers.Concatenate([rnn_layer, pills_received])
# #
# # dense = tf.keras.layers.Dense(16, 'relu')
# # output = tf.keras.layers.Dense(1, 'sigmoid')
# #
# # model_ = tf.keras.models.Model(inputs=[angles_input, pills_received], outputs=[output])
