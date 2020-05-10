import random

import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, LSTM, GRU, Flatten
from keras.models import Sequential
from numpy import loadtxt

bestAccuracy = 0.0
bestModel = list()
# load the dataset
dataset = loadtxt('data/Testing_patients.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:, 0:7]
X = X.reshape(-1, 1, 7)

y = dataset[:, 8]

ValidationDataSet = loadtxt('data/Testing_validation_patients.csv', delimiter=',')
# split into input (X) and output (y) variables
XVal = ValidationDataSet[:, 0:7]
XVal = XVal.reshape(-1, 1, 7)

yVal = ValidationDataSet[:, 8]

while True:
    # define the keras model
    model = Sequential()
    currentModel = list()
    maxLayers = 5
    i = 0
    while random.randint(0, 2) != 0:
        count = random.randint(20, 70)
        dropout = random.uniform(0, 0.5)
        if random.randint(0, 3) != 0:
            model.add(LSTM(count, input_shape=(1, 7), return_sequences=True, dropout=dropout))
            currentModel.append(
                "model.add(LSTM(" + str(count) + ", input_shape=(1, 7), return_sequences=True, dropout=" + str(
                    dropout) + "))")
        else:
            model.add(GRU(count, input_shape=(1, 7), return_sequences=True, dropout=dropout))
            currentModel.append(
                "model.add(GRU(" + str(count) + ", input_shape=(1, 7), return_sequences=True, dropout=" + str(
                    dropout) + "))")
        i += 1
        if i > maxLayers:
            break
    model.add(Flatten())
    i = 0
    maxLayers = 15
    while random.randint(0, 7) != 0:
        count = random.randint(2, 1024)
        dropout = random.uniform(0, 0.5)
        if random.randint(0, 3) != 0:
            model.add(Dense(count))

            currentModel.append("model.add(Dense(" + str(count) + "))")
        else:
            model.add(Dropout(dropout))
            currentModel.append("model.add(Dropout(" + str(dropout) + "))")
        i += 1
        if i > maxLayers:
            break
    model.add(Dense(1, activation='sigmoid'))
    currentModel.append("Dense(1, activation=\"sigmoid\")")
    if random.randint(0, 1) == 0:
        loss = 'mean_squared_error'
    else:
        loss = 'binary_crossentropy'

    if random.randint(0, 1) == 0:
        optimizer = 'rmsprop'
    else:
        optimizer = 'adam'

    currentModel.append("model.compile(loss='" + loss + "', optimizer='" + optimizer + "', metrics=['mse', 'acc'])")
    for st in currentModel:
        print(st)

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['mse', 'acc'])


    epoch = random.randint(5, 15)
    # fit the keras model on the dataset
    history = model.fit(X, y, epochs=epoch, validation_split=0.25, shuffle=True, batch_size=5000, verbose=0)

    currentModel.append(
        "model.fit(X, y, epochs=" + str(epoch) + ", validation_split=0.25, shuffle=True, batch_size=5000, verbose=0)")

    # print(model.summary())

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
    score = model.evaluate(XVal, yVal, verbose=0)
    print('Test Loss: %f' % score[0])
    print('Test Accuracy: %f' % score[1])
    currentAccuracy = score[1]
    # make predictions with the model
    predictions = model.predict(XVal)
    if (currentAccuracy > 0.55 or currentAccuracy < 0.45) and currentAccuracy > bestAccuracy:
        bestModel = list(currentModel)
        bestAccuracy = currentAccuracy
        afile = open("data/best.txt", "a")
        print("\n\n==========================")
        afile.write("\n\n==========================")
        afile.write("Best Model Accuracy %f" % currentAccuracy+"\n")
        print("Best Model Accuracy %f" % currentAccuracy)
        for st in bestModel:
            print(st)
            afile.write(st+'\n')
        afile.write("==========================\n\n")
        afile.close()
        print("==========================\n\n")