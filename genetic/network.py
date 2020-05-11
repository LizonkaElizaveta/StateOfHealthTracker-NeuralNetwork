import random

from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU, Flatten, LSTM
from keras.callbacks import EarlyStopping


class NeuralNetwork:

    def __init__(self, settings_config=None):
        """Initialize neural network.

        :param settings_config - settings for create neural network
        """
        self.accuracy = 0.0
        self.settings = settings_config
        self.network_params = {}
        self.model = []

    def create_random(self):
        for key in self.settings:
            self.network_params[key] = random.choice(self.settings[key])

    def setup_network(self, network):
        """Setup network properties.

        :param network: The network settings

        """
        self.network_params = network

    def model_append(self, line):
        """Append neural network model for print

        :param line: line of model

        """
        self.model.append(line)

    def train(self, dataset, common_settings):
        """Train the network and record the accuracy.
        """
        if self.accuracy == 0.0:
            self.accuracy = train_and_score(self, dataset, common_settings)

    def print_network(self):
        log_file = open('models', 'a')
        log_file.write('\n')
        log_file.write('=' * 80)
        log_file.write('\n')
        log_file.write("Accuracy: %.2f%%" % (self.accuracy * 100) + '\n')
        print("Accuracy: %.2f%%" % (self.accuracy * 100))
        for st in self.model:
            log_file.write(st + '\n')
            print(st)


def train_and_score(network, dataset, common_settings):
    """Train the model, return test loss.

        :param network: network settings
        :param common_settings: common settings
        :param dataset: (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

    """
    nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test = dataset

    recurrent_layers_count = network.network_params['recurrent_layers_count']
    if recurrent_layers_count > 0:
        x_train = x_train.reshape(-1, 1, input_shape[0])
        x_test = x_test.reshape(-1, 1, input_shape[0])
        print("Reshape")

    model = compile_model(network, nb_classes, input_shape)
    network.model_append('model.fit(x_train, y_train,\n\
               batch_size=' + str(batch_size) + ',\n\
               epochs=' + str(common_settings['epochs']) + ',\n\
               verbose=' + str(common_settings['verbose']) + ',\n\
               validation_data=(x_test, y_test),\n\
               callbacks=[EarlyStopping(patience=' + str(common_settings['early_stopping_patience']) + ')])')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=common_settings['epochs'],
              verbose=common_settings['verbose'],
              validation_data=(x_test, y_test),
              callbacks=[EarlyStopping(patience=common_settings['early_stopping_patience'])])

    score = model.evaluate(x_test, y_test, verbose=common_settings['verbose'])

    network.model_append('model.evaluate(x_test, y_test, verbose=' + str(common_settings['verbose']) + ')')
    return score[1]  # return accuracy


def compile_model(network, nb_classes, input_shape):
    layers_count = network.network_params['layers_count']
    neurons_count = network.network_params['neurons_count']
    activation = network.network_params['activation']
    optimizer = network.network_params['optimizer']
    dropout_layers_values = network.network_params['dropout_layers_values']
    dropout_layers_probability = network.network_params['dropout_layers_probability']
    loss = network.network_params['loss']
    output_activation = network.network_params['output_activation']
    recurrent_layers_count = network.network_params['recurrent_layers_count']
    recurrent_layers = network.network_params['recurrent_layers']
    recurrent_layers_units = network.network_params['recurrent_layers_units']

    model = Sequential()

    if recurrent_layers_count > 0:
        input_shape = (1, input_shape[0])
        for i in range(recurrent_layers_count):
            units = get_random_from_list(recurrent_layers_units)
            if i == 0:
                if recurrent_layers[0] == 'GRU':
                    model.add(GRU(units, input_shape=input_shape, return_sequences=True))
                    network.model_append(
                        'model.add(GRU(' + str(units) + ', input_shape=' + str(
                            input_shape) + ', return_sequences=True))')
                elif recurrent_layers[0] == 'LSTM':
                    model.add(LSTM(units, input_shape=input_shape, return_sequences=True))
                    network.model_append(
                        'model.add(LSTM(' + str(units) + ', input_shape=' + str(
                            input_shape) + ', return_sequences=True))')

            for layer in recurrent_layers[1:]:
                if layer == 'GRU':
                    model.add(GRU(units, return_sequences=True))
                    network.model_append('model.add(GRU(' + str(units) + ', return_sequences=True))')
                elif layer == 'LSTM':
                    model.add(LSTM(units, return_sequences=True))
                    network.model_append('model.add(LSTM(' + str(units) + ', return_sequences=True))')
        model.add(Flatten())
        network.model_append('model.add(Flatten())')
    else:
        neurons_count_ = get_random_from_list(neurons_count)
        activation_ = get_random_from_list(activation)
        model.add(
            Dense(neurons_count_, activation=activation_, input_shape=input_shape))
        network.model_append(
            'model.add(Dense(' + str(neurons_count_) + ', activation=\'' + str(activation_) + '\', input_shape=' + str(
                input_shape) + '))')

    for i in range(layers_count - 1):
        neurons_count_ = get_random_from_list(neurons_count)
        activation_ = get_random_from_list(activation)
        model.add(Dense(neurons_count_, activation=activation_))
        network.model_append('model.add(Dense(' + str(neurons_count_) + ', activation=\'' + str(activation_) + '\'))')

        if random.randint(0, 100) < dropout_layers_probability * 100:
            model.add(Dropout(dropout_layers_values))
            network.model_append('model.add(Dropout(' + str(dropout_layers_values) + '))')

    model.add(Dense(nb_classes, activation=output_activation))
    network.model_append('model.add(Dense(' + str(nb_classes) + ', activation=\'' + str(output_activation) + '\'))')
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    network.model_append(
        'model.compile(loss=\'' + str(loss) + '\', optimizer=\'' + str(optimizer) + '\', metrics=[\'accuracy\'])')
    return model


def get_random_from_list(param_list):
    return param_list[random.randint(0, len(param_list) - 1)]
