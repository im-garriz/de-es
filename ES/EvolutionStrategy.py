# Array class
import array
# Random distributions
import random
# Numpy for mathematical expressions in arrays
import numpy as np
# Deap library for evolutionary computation tools and algorithms
from deap import base, benchmarks, creator, tools
# Argument parser programmed in ConfigurableParametersES.py
from ConfigurableParametersES import parse_arguments
# Mathematical functions
import math


def create_ES_classes(parameters):
    """
    Creates the classes Fitness, Individual and Strategy, which are needed in a ES

    :param parameters: parsed parameters

    :return:
    """

    # Created Fitness class with positive weights (maximization) or negative
    # (minimization) according to the parsed parameters
    if parameters._maximization:
        creator.create("Fitness", base.Fitness, weights=(1.0,))
    else:
        creator.create("Fitness", base.Fitness, weights=(-1.0,))

    # Creates Individual that inherits from array and has a strategy member
    creator.create("Individual", array.array, typecode="d", fitness=creator.Fitness, strategy=None)
    # Creates a Strategy class that inherits from array
    creator.create("Strategy", array.array, typecode="d")


def initialize_individual(individual_class, strategy_class, parameters):
    """
    Function that initializes each individual. Values of each individual are generated with a random uniform
    distribution between the parsed bound (Dom of the function) and strategy parameters with a random uniform
    distribution between -1.5 and 1.5, as values near to the unit are suggested at [Back, 1996]

    :param individual_class: class to which individual objects belongs to
    :param strategy_class: class to which strategy objects belongs toinici
    :param parameters: parsed parameters

    :return: an initialized individual
    """

    # Initialized values of the individual
    individual = individual_class(random.uniform(parameters._lower_bound, parameters._upper_bound)
                                  for _ in range(parameters._n))

    # Initializes strategy parameters of the individual
    if parameters._mutation_method == "no correlated n-steps":
        individual.strategy = strategy_class(random.uniform(0, 1.5) for _ in range(parameters._n))
    else:
        individual.strategy = [random.uniform(0, 1.5)]

    # Return the initialized individual
    return individual

def initialize_individual_with_values(individual_class, strategy_class, _individual, _strategy):
    """
    Initializes a new individual provided its values and strategy parameters

    :param individual_class: class to which individual objects belongs to
    :param strategy_class: class to which strategy objects belongs to
    :param _individual: values for the new individual
    :param _strategy: strategy prarameters for the new individual

    :return: initialized individual
    """

    individual = individual_class(_individual)
    individual.strategy = strategy_class(_strategy)

    # Return the initialized individual
    return individual


def no_correlated_1_step_mut(individual, c):
    """
    1 step no-correlated mutation method

    :param individual: individual to be mutated
    :param c: learning parameter (proportional to 1/sqrt(n) as suggested in [Schwefel, 1977]

    :return: mutated individual
    """
    n = len(individual)
    tau = c / math.sqrt(n)

    individual.strategy[0] *= math.exp(tau * random.gauss(0, 1))

    for idx in range(n):
        individual[idx] += individual.strategy[0] * random.gauss(0, 1)

    return individual,


def build_ES(parameters):
    """
    Function that builds the ES and mapd mate, mutate and select operators in function of parsed parameters

    :param parameters: parsed parameters

    :return: base.Toolbox() object with the configured algorithm
    """

    # Instanciates base.Toolbox() object
    toolbox = base.Toolbox()

    # Creates functions for individual and population initializers
    toolbox.register("individual", initialize_individual, creator.Individual, creator.Strategy, parameters)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Creates the mate operator
    toolbox.register("mate", gen_descendency, ro=parameters._ro, Lambda=parameters._lambda)

    # Creates the mutate operator
    if parameters._mutation_method == "no correlated n-steps":
        # If no correlated with n-steps uses the deap inplementation (tools.mutESLogNormal)
        toolbox.register("mutate", tools.mutESLogNormal, c=parameters._tau_factor, indpb=1)
    else:
        # If no correlated with 1-step uses no_correlated_1_step_mut function (deap does not implement it)
        toolbox.register("mutate", no_correlated_1_step_mut, c=parameters._tau_factor)

    # Generates the select operator (in ES survival selection is done selecting the best
    # mu individuals from lambda or from mu+lambda, so tools.selBest operator from deap has been used)
    toolbox.register("select", tools.selBest, k=parameters._mu)

    # Generated evaluate method (Sphere and Ackley are both implemented in deap)
    if parameters._evaluation_function == "Sphere":
        toolbox.register("evaluate", benchmarks.sphere)
    elif parameters._evaluation_function == "Ackley":
        toolbox.register("evaluate", benchmarks.ackley)
    else:
        print("No programmed evaluation function selected -> Sphere one selected by default")
        toolbox.register("evaluate", benchmarks.sphere)

    # Checks whether any strategy parameter is smaller than epsilos0, if so, sets in to epsilon0
    # (called after both mate and mutate operations)
    toolbox.decorate("mate", check_strategy_value(parameters._epsilon0))
    toolbox.decorate("mutate", check_strategy_value(parameters._epsilon0))

    # Return the configuration
    return toolbox

def check_strategy_value(minstrategy):
    """
    Checks whether any strategy parameter is smaller than epsilos0, if so, sets in to epsilon0

    :param minstrategy: minimun value for strategy parameters (epsilon0)

    :return: decorator
    """
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children
        return wrappper
    return decorator


def ES_algorithm(parameters, toolbox, stats=None, halloffame=None, verbose=False, finish=True):
    """
    Implementation of the ES algorithm

    :param parameters: parsed parameters
    :param toolbox: previously configured toolbox
    :param stats: stats to calculate on each iteration (avg fitness, std, etc..)
    :param halloffame: hallOfFame object that stores the best individuals on a execution
    :param verbose: if true, information is printed in terminal during execution
    :param finish: if true, execution is stopped when solution reached, else it continues

    :return: (solution_reached, number_of_evaluations, best_fitness), logbook
    """

    # Initialization of execution log parameters
    solution_reached = False
    number_of_evaluations = 0
    solution_reached_in_generation = -1
    fitness_value_of_the_solution = -1

    # Initialization of the logbook
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Initialization of the population
    population = toolbox.population(n=parameters._mu)
    invalid_ind = [individual for individual in population if not individual.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, population))
    for individual, fitness_value in zip(invalid_ind, fitnesses):
        individual.fitness.values = fitness_value

    # Initialization of the generation variable
    generation = 0

    # Gets the individual with the best fitness value and checks
    # if it is good enought to finish the execution
    best_fitness = list(map(toolbox.evaluate, population))[0][0]
    if best_fitness < parameters._max_fitness_value_valid_solution and solution_reached==False:
        solution_reached = True
        number_of_evaluations += len(invalid_ind)
        solution_reached_in_generation = 0
        if finish:
            generation = parameters._n_of_gens
    elif solution_reached==True:
        pass
    else:
        number_of_evaluations += len(invalid_ind)

    # Updates the HallOfFame object
    if halloffame is not None:
        halloffame.update(population)

    # Writes the stats of gen 0 in the notebook
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=generation, nevals=len(invalid_ind), **record)

    # If verbose prints that stats
    if verbose:
        print(logbook.stream)

    # While _n_of_gens generations
    while generation < parameters._n_of_gens:

        generation += 1

        # Variation step: parent selection + mating and mutation
        descendency = crossover_and_mutation(population, toolbox, parameters._ro, parameters._lambda,
                                             parameters._mating_method)

        # Check invalid values
        for individual in descendency:
            for i in range(len(individual)):
                if individual[i] > parameters._upper_bound:
                    individual[i] = parameters._upper_bound
                elif individual[i] < parameters._lower_bound:
                    individual[i] = parameters._lower_bound

        # Evaluation of the generated individuals
        invalid_ind = [individual for individual in descendency if not individual.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for individual, fitness_value in zip(invalid_ind, fitnesses):
            individual.fitness.values = fitness_value

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(descendency)

        # Survival selection
        if parameters._survival_selection_method == "mu_comma_lambda":
            # If mu,lambda they are only selected among the descendency
            population[:] = toolbox.select(descendency)
        else:
            # If mu+lambda they are selected among descendency+population
            population[:] = toolbox.select(descendency + population)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=generation, nevals=len(invalid_ind), **record)

        if verbose:
            print(logbook.stream)

        # Gets the individual with the best fitness value and checks
        # if it is good enought to finish the execution
        best_fitness = list(map(toolbox.evaluate, population))[0][0]
        if best_fitness < parameters._max_fitness_value_valid_solution and solution_reached == False:
            solution_reached = True
            number_of_evaluations += len(invalid_ind)
            solution_reached_in_generation = generation
            if finish:
                break
        elif solution_reached == True:
            pass
        else:
            number_of_evaluations += len(invalid_ind)

    # When the execution is finished prints the results and returns them
    if verbose:
        print("ES finished")
        print("Solution reached = {}".format(solution_reached))
        print("Number of evaluations = {}".format(number_of_evaluations))
        print("Solution reached in generation {}".format(solution_reached_in_generation))

    return (solution_reached, number_of_evaluations, best_fitness), logbook


def intermediate_mating(parents, method):
    """
    Intermediate mating function for strategy parameters, as recommended at [Back, 1996]

    :param parents: parents of the child
    :param method: method of intermediate mating: intermediate_0point5,
                   generalized intermediate or averaged generalized

    :return: mated child
    """

    # Initializes the output strategy
    son_strategy = []

    # If method intermediate_0point5 or generalized intermediate -> need a fixed parent
    if method == "intermediate_0point5" or method == "generalized intermediate":
        # Gets the fixed parent
        fixed_parent_idx = random.randint(0, len(parents)-1)
        fixed_parent = parents[fixed_parent_idx]

        # Calculates the corresponding r value of the method
        if method == "intermediate_0point5":
            r = 0.5
        else:
            r = random.uniform(0, 1)

        # Upadates each strategy parameter according to the formula of intermediate mating
        for i in range(len(parents[0].strategy)):
            random_parent_idx = random.randint(0, len(parents) - 1)
            random_parent = parents[random_parent_idx]
            value = r * fixed_parent.strategy[i] + (1-r) * random_parent.strategy[i]
            son_strategy.append(value)
    else:
        # averaged generalized mating
        for i in range(len(parents[0].strategy)):
            parents_i_val = [parent.strategy[i] for parent in parents]
            average = np.float32(np.sum(parents_i_val)) / np.float32(len(parents))
            son_strategy.append(average)

    # Returns the mated strategy
    return son_strategy


def gen_descendency(ro, Lambda, population, mating_method):
    """
    Generates a descendency of lambda individuals from mu parents

    :param ro: number of parents for each son
    :param Lambda: lumber of sons
    :param population: current population
    :param mating_method: method of intermediate mating: intermediate_0point5,
                          generalized intermediate or averaged generalized

    :return: descendency of length lambda
    """

    # Initialized the descendency container
    descendency = []

    # Mate a new individual lambda times
    for i in range(Lambda):

        # Gets ro random parents from population
        parents = [population[index] for index in np.random.randint(len(population), size=ro)]

        # Intermediate mating for strategy parameters
        son_strategy = intermediate_mating(parents, mating_method)

        # Discrete mating for variables
        son_values = [parents[parent_idx][pos] for pos, parent_idx in
                      enumerate(np.random.randint(ro, size=len(parents[0])))]

        # Generates the new individual with the mated strategy and values
        son = initialize_individual_with_values(creator.Individual, creator.Strategy, son_values, son_strategy)
        del son.fitness.values

        # Inserts it into the list
        descendency.append(son)

    # Return the generated descendency
    return descendency


def crossover_and_mutation(population, toolbox, ro, Lambda, mating_method):
    """
    Does the variation steps to the population: crossover and mutation

    :param population: current population
    :param toolbox: configured algorithm
    :param ro: number of parents for each son
    :param Lambda: number of sons
    :param mating_method: method of intermediate mating: intermediate_0point5,
                          generalized intermediate or averaged generalized

    :return: descendency (offspring)
    """

    # Clones all individuals so that they are not modified in following steps
    offspring = [toolbox.clone(individual) for individual in population]

    # Crossover
    offspring = gen_descendency(ro, Lambda, offspring, mating_method)

    # Mutation
    for i in range(len(offspring)):
        offspring[i], = toolbox.mutate(offspring[i])
        del offspring[i].fitness.values

    # Return generated offspring
    return offspring

def run_ES():
    """
    Runs the programmed ES

    :return: convergence info (converged to solution True/False, best individuals
             fitness and number of evaluations of the fitness function) and logbook (class with
             logs of the whole population on each generation)
    """

    # Gets the parsed arguments
    parameters = parse_arguments()


    # Builds the ES
    create_ES_classes(parameters)
    toolbox = build_ES(parameters)

    # Metrics that will be saved on the logbook
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Hall of fame with the best individuals on the whole execution
    hof = tools.HallOfFame(1)

    # Launchs the ES
    convergence_info, logbook = ES_algorithm(parameters, toolbox, stats=stats,
                                             halloffame=hof, verbose=True, finish=False)

    # Return results
    return convergence_info, logbook
