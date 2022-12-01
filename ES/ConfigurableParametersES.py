# Argument parsing for configuration
import argparse
# Numpy for mathematical expressions in arrays
import numpy as np


class Parameters:
    """
    Class that simply has a member for each parsed configuration parameter and has no methods. It is
    just initialized with all parsed values and stores them
    """

    def __init__(self, _evaluation_function, _upper_bound, _lower_bound, _maximization, _n, _mu, _lambda, _epsilon0,
                 _n_of_gens, _ro, _mutation_method, _tau_factor, _survival_selection_method,
                 _max_fitness_value_valid_solution, _mating_method):

        # Stores all parsed parameters (properly casted) in a member when constructed
        self._evaluation_function = _evaluation_function
        self._upper_bound = np.float32(_upper_bound)
        self._lower_bound = np.float32(_lower_bound)
        self._maximization = int(_maximization)
        self._n = int(_n)
        self._mu = int(_mu)
        self._lambda = int(_lambda)
        self._epsilon0 = np.float32(_epsilon0)
        self._n_of_gens = int(_n_of_gens)
        self._ro = int(_ro)
        self._mutation_method = _mutation_method
        self._tau_factor = int(_tau_factor)
        self._survival_selection_method = _survival_selection_method
        self._max_fitness_value_valid_solution = np.float32(_max_fitness_value_valid_solution)
        self._mating_method = _mating_method

def parse_arguments():
    """
    Parses all configuration parameters, stores them in a Parameters() object and returns it

    :return: parameters
    """
    parser = argparse.ArgumentParser(description='Arguments to configure the Evolution Strategy')

    parser.add_argument('-f', '--function', help='Function to maximize/minimize')
    parser.add_argument('--upperbound', help='Upper bound of the parameters of each individual')
    parser.add_argument('--lowerbound', help='Lower bound of the parameters of each individual')
    parser.add_argument('--maximization', help='True if it is a maximization problem')
    parser.add_argument('--n', help='Number of variables of the function to be optimized')
    parser.add_argument('--mu', help='Population length')
    parser.add_argument('--Lambda', help='Descendency length')
    parser.add_argument('--epsilon0', help='Minimum value for mutation parameters')
    parser.add_argument('--numbergens', help='Number of generations to run')
    parser.add_argument('--ro', help='Number of parents for each son')
    parser.add_argument('--mutationmethod', help='Method used in mutation step')
    parser.add_argument('--taufactor', help='Factor so that tau = Factor/sqrt(2n) and\
                                             tau2 = Factor/sqrt(2sqrt(n)) in strategy parameters mutation step')
    parser.add_argument('--survivalselection', help='Method used on survival selection')
    parser.add_argument('--solfitness', help='Maximum/minimum fitness value for valid solution')

    parser.add_argument('--strategymating', help='Method used to mate strategies')

    parser = parser.parse_args()

    parameters = Parameters(parser.function, parser.upperbound, parser.lowerbound, parser.maximization, parser.n,
                            parser.mu, parser.Lambda, parser.epsilon0, parser.numbergens, parser.ro,
                            parser.mutationmethod, parser.taufactor, parser.survivalselection,
                            parser.solfitness, parser.strategymating)

    return parameters
