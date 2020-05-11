import random
from train import train_and_score


class NeuralNetwork:

    def __init__(self, settings_config=None):
        """Initialize neural network.

        :param settings_config - settings for create neural network
        """
        self.accuracy = 0.
        self.settings = settings_config
        self.network = {}

    def create_random(self):
        for key in self.settings:
            self.network[key] = random.choice(self.settings[key])

    def setup_network(self, network):
        """Setup network properties.

        :param network: The network settings

        """
        self.network = network

    def train(self, dataset, common_settings):
        """Train the network and record the accuracy.
        """
        if self.accuracy == 0.:
            self.accuracy = train_and_score(self.network, dataset, common_settings)

    def print_network(self):
        print(self.network)
        print("Network accuracy: %.2f%%" % (self.accuracy * 100))
