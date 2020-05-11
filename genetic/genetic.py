import random
from network import NeuralNetwork


class GeneticAlgorithm:

    def __init__(self, mutation_config, mutate_chance=0.2, retain=0.4,
                 random_selection=0.1, same_network_new_generations_count=2):
        """Create a GeneticAlgorithm.

            :param mutation_config: Mutation configuration params
            :param retain: Percentage of population to retain after
                each generation
            :param random_selection: Probability of a rejected network
                remaining in the population
            :param mutate_chance: Mutation of networks
        """

        self.mutation_config = mutation_config
        self.mutate_chance = mutate_chance
        self.random_selection = random_selection
        self.retain = retain
        self.same_network_new_generations_count = same_network_new_generations_count

    def generate_population(self, count):
        """Create a random networks population.

        :param count: size of the population

        :return: List of populated neural networks
        """
        population = []
        for _ in range(0, count):
            network = NeuralNetwork(self.mutation_config)
            network.create_random()

            population.append(network)

        return population

    def fitness_score(self, network):
        """ Fitness score

            :return network score (accuracy for now)
        """
        return network.accuracy

    def grade(self, population):
        """Find average fitness for a population.

        :param population: The population of networks

        :return: List of networks with grade

        """
        return [(self.fitness_score(network), network) for network in population]

    def new_generation(self, first_network, second_network):
        """Make two children as parts of their parents.

            :param first_network: Network parameters
            :param second_network: Network parameters

            :return generated children

        """

        children = []
        for _ in range(self.same_network_new_generations_count):

            child = {}

            for param in self.mutation_config:
                child[param] = random.choice(
                    [first_network.network_params[param], second_network.network_params[param]]
                )

            network = NeuralNetwork(self.mutation_config)
            network.setup_network(child)

            if self.mutate_chance > random.random():
                network = self.random_mutate(network)

            children.append(network)

        return children

    def random_mutate(self, network):
        """Randomly mutate one part of the network.

            :param network: The network settings to mutate

            :return A randomly mutated network object
        """

        mutation_key = random.choice(list(self.mutation_config.keys()))

        network.network_params[mutation_key] = random.choice(self.mutation_config[mutation_key])

        return network

    def evolution(self, population):
        """Evolve a population.

            :param population: A list of populations

            :return new population of networks

        """
        graded_networks = self.grade(population)

        # Sort networks by fitness score
        graded_networks = [x[1] for x in sorted(graded_networks, key=lambda x: x[0], reverse=True)]

        retain_length = int(len(graded_networks) * self.retain)

        new_population = graded_networks[:retain_length]

        # Keep random number of bad networks
        for network in graded_networks[retain_length:]:
            if self.random_selection > random.random():
                new_population.append(network)

        new_population_length = len(new_population)
        desired_length = len(population) - new_population_length
        children = []

        while len(children) < desired_length:

            # Get random networks from new population
            first_network = random.randint(0, new_population_length - 1)
            second_network = first_network
            while second_network == first_network:
                second_network = random.randint(0, new_population_length - 1)

            first_network = new_population[first_network]
            second_network = new_population[second_network]

            descendants = self.new_generation(first_network, second_network)

            for descendant in descendants:
                if len(children) < desired_length:
                    children.append(descendant)

        new_population.extend(children)

        return new_population
