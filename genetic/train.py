from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping


def compile_model(network, nb_classes, input_shape):
    layers_count = network['layers_count']
    neurons_count = network['neurons_count']
    activation = network['activation']
    optimizer = network['optimizer']
    dropout_layers_values = network['dropout_layers_values']
    loss = network['loss']
    output_activation = network['output_activation']

    model = Sequential()

    model.add(Dense(neurons_count, activation=activation, input_shape=input_shape))

    for i in range(layers_count - 1):
        model.add(Dense(neurons_count, activation=activation))

        model.add(Dropout(dropout_layers_values))

    model.add(Dense(nb_classes, activation=output_activation))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    print(model.summary())

    return model


def train_and_score(network, dataset, common_settings):
    """Train the model, return test loss.

        :param network: network settings
        :param common_settings: common settings
        :param dataset: (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

    """
    nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test = dataset

    model = compile_model(network, nb_classes, input_shape)

    # model.fit(x_train, y_train,
    #           batch_size=batch_size,
    #           epochs=common_settings['epochs'],  # using early stopping, so no real limit
    #           verbose=common_settings['verbose'],
    #           validation_data=(x_test, y_test),
    #           callbacks=[EarlyStopping(patience=common_settings['early_stopping_patience'])])
    #
    # score = model.evaluate(x_test, y_test, verbose=common_settings['verbose'])

    # return score[1]  # return accuracy
    return random.uniform()
