#!/usr/bin/python

# Evolution strategy programmed in EvolutionStrategy.py
from EvolutionStrategy import run_ES
# Utils programmed in utils.py with some utils as plotters or random seed setters
from utils import print_progress_curve, set_random_seed
# Numpy for mathematical expressions in arrays
import numpy as np


# Main program
def main():
    """
    Runs the evolution strategy NUMBER_OF_EXECUTIONS times and generates a
    log file with the execution info

    :return:
    """
    NUMBER_OF_EXECUTIONS = 30
    LOGFILE_NAME = "log.txt"

    # Random seed setters for reproducibility
    set_random_seed(42)
    random_seeds = np.random.randint(9999,  size=NUMBER_OF_EXECUTIONS)

    # Number of successes counter for calculation of TE
    number_of_successes = 0
    # Best individuals fitness on each execution for calculation of VAMM
    best_individuals_fitnesses = []
    # Number of evaluation function on each successful execution, for calculating PEX
    successes_number_of_evaluations = []

    # For each random seed (NUMBER_OF_EXECUTIONS)
    for seed in random_seeds:

        # Set the seed
        set_random_seed(seed)
        # Run the algorithm
        convergence_parameters, logbook = run_ES()

        # Convergence parameters:
        #     - 0: Successful execution: True/False
        #     - 1: Number of evaluations of the evaluation function until convergence
        #     - 2: Fitness of the reached best individual

        # If successful execution increase number_of_successes by 1 and
        # add the number of evaluations to successes_number_of_evaluations
        if convergence_parameters[0]:
            number_of_successes += 1
            successes_number_of_evaluations.append(convergence_parameters[1])

        # Append the best individuals fitness value to best_individuals_fitnesses whether the algorithm has
        # converged or not
        best_individuals_fitnesses.append(convergence_parameters[2])


        # Generate the log file:
        #     - Each line is one execution of the algorithm (NUMBER_OF_EXECUTIONS lines):

        #       * best_inviduals_fitness on each gen separated by ',' + ';' +
        #       average_fitness on each gen separated by ',' + ';' + number of evaluations of the
        #       evaluation function until convergence

        #     - Last 3 lines:
        #       * TE=obtained_te
        #       * VAMM=obtained_vamm
        #       * PEX=obtained_pex

        # Write the first NUMBER_OF_EXECUTIONS lines
        with open(LOGFILE_NAME, 'a') as file:
            for n, value in enumerate(logbook.select("min")):
                file.write("{}".format(value))

                if n < len(logbook.select("min"))-1:
                    file.write(',')
                else:
                    file.write(';')

            for n, value in enumerate(logbook.select("avg")):
                file.write("{}".format(value))
                if n < len(logbook.select("avg"))-1:
                    file.write(',')
                else:
                    file.write(';')

            file.write("{}".format(convergence_parameters[1]))
            file.write('\n')

    # Get TE, VAMM and PEX
    TE = np.float32(number_of_successes) * 100.0 / np.float32(NUMBER_OF_EXECUTIONS)
    VAMM = np.mean(best_individuals_fitnesses)
    if len(successes_number_of_evaluations) > 0:
        PEX = np.mean(successes_number_of_evaluations)
    else:
        PEX = -1

    # Write them into the file
    with open(LOGFILE_NAME, 'a') as file:
        file.write("TE={}\n".format(TE))
        file.write("VAMM={}\n".format(VAMM))
        file.write("PEX={}\n".format(PEX))


# Main program caller and exception handler
if __name__ == '__main__':
    try:
        # Main program is stored in main() function
        main()
    except KeyboardInterrupt:
        print("Exiting...")
        exit()
