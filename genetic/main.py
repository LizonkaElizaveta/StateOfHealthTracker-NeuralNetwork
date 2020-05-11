from numpy import loadtxt
from genetic import GeneticAlgorithm

# Setup common settings
common_settings = {
    'verbose': 0,
    'epochs': 10000,
    'early_stopping_patience': 5
}


def train_networks(networks, dataset, common_settings):
    for network in networks:
        network.train(dataset, common_settings)


def get_average_accuracy(networks):
    """
        Average accuracy
        :param networks: List of networks

        :return avarage accuracy
    """
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)


def generate(generations, population, mutation_config):
    """Generate a network with the genetic algorithm.

        :param generations: Number generations
        :param population: Number of populations in each generation
        :param mutation_config: Settings for mutation

    """
    ga = GeneticAlgorithm(mutation_config)
    networks = ga.generate_population(population)

    for i in range(generations):
        print("Generation %d of %d" % (i + 1, generations))

        train_networks(networks, get_dataset(), common_settings)

        average_accuracy = get_average_accuracy(networks)

        print("Average accuracy: %.2f%%" % (average_accuracy * 100))
        print('=' * 80)

        if i != generations - 1:
            networks = ga.evolution(networks)

    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    print('=' * 80)
    for network in networks[:5]:
        network.print_network()

def parkinson_dataset():
    nb_classes = 1
    batch_size = 512
    input_shape = (7,)

    dataset = loadtxt('data/Testing_patients.csv', delimiter=',')
    x_train = dataset[:, 0:7]
    y_train = dataset[:, 8]

    dataset = loadtxt('data/Testing_validation_patients.csv', delimiter=',')
    x_test = dataset[:, 0:7]
    y_test = dataset[:, 8]

    return nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test


def get_dataset():
    return parkinson_dataset()


def main():
    generations = 20
    population = 10

    mutation_config = {
        'neurons_count': [2 ** i for i in range(6, 12)],
        'layers_count': [i for i in range(1, 5)],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
                      'adadelta', 'adamax', 'nadam'],
        'loss': ['binary_crossentropy'],
        'output_activation': ['sigmoid'],
        'dropout_layers_probability': [0.0, 0.1, 0.2],
        'dropout_layers_values': [0.1, 0.15, 0.2],
        'recurrent_layers_count': [i for i in range(1, 5)],
        'recurrent_layers': ['LSTM', 'GRU']
    }

    generate(generations, population, mutation_config)


if __name__ == '__main__':
    main()
