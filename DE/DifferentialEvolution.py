# For operations with arrays
import numpy as np
# Random distributions
import random
# Function that parsed the  arguments from the shell script
from ConfigurableParametersDE import parse_arguments
# Logbook from deap.tools class for generating log files
from deap import tools


def calculate_fitness(population, fitness_function):
    """
    Function that calculates the fitness value of each infividual in population

    :param population: (PL,n) array that contains the population
    :param fitness_function: fitness function to evaluate (ackley or sphere)

    :return: array of size (PL,) with each fitness value
    """
    n = population.shape[1]

    if fitness_function == "Sphere":
        # Sphere function
        return np.sum(np.power(population, 2), axis=1)
    elif fitness_function == "Ackley":
        # Ackley function
        return -20 * np.exp(-0.2 * np.sqrt(1/n * np.sum(np.power(population, 2), axis=1))) -\
               np.exp(1/n * np.sum(np.cos(2 * np.pi * population), axis=1)) + 20 + np.e
    else:
        raise Exception("Fitness function \"{}\" not implemented".format(fitness_function))

def initialize_population(n, PL, boundaries, fitness_function):
    """
    Function that generates the initial population in a PL x n array (each row represents an
    individual and each col of row i ith individuals genes using a random uniform distribution between
    provided boundaries

    :param n: number of variables of each individual (genotype length)
    :param PL: lenght of the population
    :param boundaries: (lower, upper) tuple containing the upper and lower boundary of each gen
    :param fitness_function: fitness function to evaluate

    :return: the initialized population and their fitness values
    """

    population = boundaries[0] + np.random.random((PL, n)) * (boundaries[1] - boundaries[0])
    fitness_values = calculate_fitness(population, fitness_function)
    return population, fitness_values


def get_mutation_indexes(current_idx, DE_variant, population, fitness_values):
    """
    Function that generates indexes r, p and q necessary for the mutation step

    :param current_idx: current index in the iteration over individuals loop
    :param DE_variant: variant of the DE
    :param population: current population
    :param fitness_values: current fitness values

    :return: r, p and q indexes
    """
    PL, n = population.shape

    r, p, q = current_idx, current_idx, current_idx

    if DE_variant == "DE/best/1/bin":
        # in DE/best/1/bin variant r represents the idx of the best individual
        r = np.argmin(fitness_values)
    elif DE_variant == "DE/rand/1/bin":
        # in DE/rand/1/bin r is a random individual
        while r == current_idx:
            r = random.randint(0, PL - 1)
    else:
        raise Exception("DE variant \"{}\" not implemented".format(DE_variant))

    while (p == current_idx) or (p == r):
        p = random.randint(0, PL - 1)
    while (q == current_idx) or (q == r) or (q == p):
        q = random.randint(0, PL - 1)

    return r, p, q

def bounce_back(offspring_vector, base_vector, boundaries):
    """
    Function that performs bounce_back method on the mutated individual to avoid illegal values

    :param offspring_vector: mutated vector
    :param base_vector: base vector in mutation
    :param boundaries: boundaries of the fitness function

    :return: the corrected mutated individual
    """

    for i in range(len(base_vector)):
        if offspring_vector[i] < boundaries[1]:
            offspring_vector[i] = base_vector[i] + random.uniform(0., 1.) * (boundaries[0] - base_vector[i])
        elif offspring_vector[i] > boundaries[0]:
            offspring_vector[i] = base_vector[i] + random.uniform(0., 1.) * (base_vector[i] - boundaries[1])

    return offspring_vector


def variate_population(population, fitness_values, DE_variant, F, CR, fitness_function, boundaries):
    """
    Function that performs the variation (mutation + mating) step on the population

    :param population: current population
    :param fitness_values: current fitness values
    :param DE_variant: DE variant that is being used
    :param F: F parameter for mutation
    :param CR: CR parameter for mating
    :param fitness_function: fitness function that is being optimized

    :return: variated population and its respective fitness values
    (both with the same shape as population and fitness values)
    """

    PL, n = population.shape
    variated_population = np.copy(population)

    for i in range(PL):

        # Get the indexes
        r, p, q = get_mutation_indexes(i, DE_variant, population, fitness_values)

        # Perform mutation
        mutated_individual = population[r, :] + F * (population[p, :] - population[q, :])

        # Bounce-neck method to avoid illegal values
        mutated_individual = bounce_back(mutated_individual, population[r, :], boundaries)

        # Mating
        alpha = random.randint(0, PL - 1)
        for j in range(n):
            beta = random.uniform(0., 1.)

            if beta < CR or j == alpha:
                variated_population[i, j] = mutated_individual[j]

    # Calculate the fitness values of the variated population
    variated_population_fitness_values = calculate_fitness(variated_population, fitness_function)

    return variated_population, variated_population_fitness_values


def survival_selection(population, fitness_values, variated_population, variated_population_fitness_values):
    """
    Function that performs survival selection. Selects between Yi (variated individual) and Xi (current individual)
    comparing their fitness for each i in PL

    :param population: current population
    :param fitness_values: current fitness values
    :param variated_population: variated population
    :param variated_population_fitness_values: fitness values of variated population

    :return: survived population and their fitness values
    """
    indexes_to_replace = np.where(variated_population_fitness_values < fitness_values)

    population[indexes_to_replace, :] = variated_population[indexes_to_replace, :]
    fitness_values[indexes_to_replace] = variated_population_fitness_values[indexes_to_replace]

    return population, fitness_values


def DE_algorithm(parameters, finish=True, verbose=True):
    """
    Function that performs the DE algorithm provided its parameters

    :param parameters: configuration parameters of the DE algorithm
    :param finish: wheter to finish executing or not when solution reached
    (false to generate progress curves on X gens)
    :param verbose: Whether to print real-time logs or not

    :return: (solution_reached, solutions_fitness_value, number_of_evaluations), logbook
    """

    # Stores all configurable parameters in variables
    n = parameters._n
    PL = parameters._PL
    DE_variant = parameters._DE_variant
    F = parameters._F
    CR = parameters._CR
    boundaries = parameters._boundaries
    fitness_function = parameters._fitness_function
    max_fitness_value = parameters._max_fitness_value
    max_generations = parameters._max_generations

    # Initialization of output parameters
    solutions_fitness_value = -1
    number_of_evaluations = 0
    solution_reached = False

    # Initialization of the logbook
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'min', 'avg', 'max']

    # Initialization of the population
    population, fitness_values = initialize_population(n, PL, boundaries, fitness_function)
    number_of_evaluations += population.shape[0]

    # Upadates the logbook
    logbook.record(gen=0, nevals=number_of_evaluations, min=np.min(fitness_values),
                   avg=np.mean(fitness_values), max=np.max(fitness_values))

    # Checks if any individual in the generated population is good enough
    if np.any(fitness_values <= max_fitness_value):
        solution_reached = True
        solutions_fitness_value = np.min(fitness_values)
        if finish:
            print("Solution reached in generation {}. Best fitness value = {}, Average fitness value = {}".
                  format(0, np.min(fitness_values), np.mean(fitness_values)))
            return (solution_reached, number_of_evaluations, solutions_fitness_value), logbook

    # For each generation
    for generation in range(1, max_generations + 1):

        # Variates current population
        variated_population, variated_population_fitness_values =\
            variate_population(population, fitness_values, DE_variant, F, CR, fitness_function, boundaries)

        # If the solution hasnt been reached updates  the number of evaluations
        if not solution_reached:
            number_of_evaluations += population.shape[0]

        # Performs survival selection
        population, fitness_values =\
            survival_selection(population, fitness_values, variated_population, variated_population_fitness_values)

        # Writes the log in the logbook
        logbook.record(gen=0, nevals=number_of_evaluations, min=np.min(fitness_values),
                       avg=np.mean(fitness_values), max=np.max(fitness_values))

        # Checks if any individual in the generated population is good enough
        if np.any(fitness_values <= max_fitness_value) and solution_reached == False:
            solution_reached = True
            solutions_fitness_value = np.min(fitness_values)
            if finish:
                if verbose:
                    print("Solution reached in generation {}. Best fitness value = {}, Average fitness value = {}".
                          format(generation, np.min(fitness_values), np.mean(fitness_values)))
                return (solution_reached, number_of_evaluations, solutions_fitness_value), logbook

        if verbose:
            print("Generation number {} finished. Best fitness value = {}, Average fitness value = {}".format
                  (generation, np.min(fitness_values), np.mean(fitness_values)))

    return (solution_reached, number_of_evaluations, solutions_fitness_value), logbook


def run_DE():
    """
    Function that parse the configuration parameters from the shell script and runs the DE

    :return: convergence_info, logbook
    """

    parameters = parse_arguments()

    convergence_info, logbook = DE_algorithm(parameters, finish=True, verbose=True)

    return convergence_info, logbook
