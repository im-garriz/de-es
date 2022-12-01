# Argument parsing for configuration
import argparse
# Numpy for mathematical expressions in arrays
import numpy as np


class Parameters:
    """
    Class that simply has a member for each parsed configuration parameter and has no methods. It is
    just initialized with all parsed values and stores them
    """
    def __init__(self, n, PL, DE_variant, F, CR, lower_bound, upper_bound, fitness_function,
                 max_fitness_value, max_generations):

        self._n = int(n)
        self._PL = int(PL)
        self._DE_variant = DE_variant
        self._F = np.float32(F)
        self._CR = np.float32(CR)
        self._boundaries = (np.float32(lower_bound), np.float32(upper_bound))
        self._fitness_function = fitness_function
        self._max_fitness_value = np.float32(max_fitness_value)
        self._max_generations = int(max_generations)


def parse_arguments():
    """
    Parses all configuration parameters, stores them in a Parameters() object and returns it

    :return: parameters
    """
    parser = argparse.ArgumentParser(description='Arguments to configure the Evolution Strategy')

    parser.add_argument('--n', help='Number of variables of the function to be optimized')
    parser.add_argument('--pl', help='Population length')
    parser.add_argument('--devariant', help='Method used in mutation step')
    parser.add_argument('-f', help='Value of F mutation parameter')
    parser.add_argument('--cr', help='Crossover parameter')
    parser.add_argument('--upperbound', help='Upper bound of the parameters of each individual')
    parser.add_argument('--lowerbound', help='Lower bound of the parameters of each individual')
    parser.add_argument('--function', help='Function to maximize/minimize')
    parser.add_argument('--solfitness', help='Maximum/minimum fitness value for valid solution')
    parser.add_argument('--numbergens', help='Number of generations to run')

    parser = parser.parse_args()

    parameters = Parameters(parser.n, parser.pl, parser.devariant, parser.f, parser.cr,
                            parser.upperbound, parser.lowerbound, parser.function,
                            parser.solfitness, parser.numbergens)

    return parameters