#!/bin/bash

## Shell script that configures all configurable options and parameters of the DE ##

### GENERAL PARAMETERS:

  # Function to maximize/minimize. The problem asks to use both sphere and Ackeley
  # functions, so only them have been implememted
  #FUNCTION_N="Sphere"
  FUNCTION_N="Ackley"

  # Boundaries for both sphere and Ackeley functions
  if [ $FUNCTION_N = "Sphere" ]
  then
    # Sphere
    UPPER_BOUND="5.12"
    LOWER_BOUND="-5.12"
  else
    # Ackley
    UPPER_BOUND="32.77"
    LOWER_BOUND="-32.77"
  fi

  # MAXIMIZATION="1" -> maximization problem, MAXIMIZATION="0" -> minimization problem
  MAXIMIZATION="0"

  # Lenght of the genotype
  N="10"

  # Mu (population length) and lambda (generated childs)
  MU="30"
  LAMBDA="200"

  # Minimum strategy value
  EPSILON0="0.0000001"

  # Number of generations to run
  MAX_GENERATIONS="1200"

  # Number of parents to generate a son
  RO="7"

  # Maximum/minimum fitness value to consider an execution as successful
  MAX_FITNESS_VAL="0.000001"

### METHODS TO USE ON EACH STEP:

# MATING:

  # Mating method for strategy parameters(intermediate for strategies, discrete for individual values)
  STRATEGY_MATING_METHOD="intermediate_0point5"
  #STRATEGY_MATING_METHOD="generalized intermediate"
  #STRATEGY_MATING_METHOD="averaged generalized"

# MUTATION:

  # Strategy parameters mutation method
  MUTATION_METHOD="no correlated n-steps"
  #MUTATION_METHOD="no correlated 1-step"
  #MUTATION_METHOD="correlated" # Not implemented as not used on the exercise

  # Proportional factor to calculate each tau value
  # (proportional to 1/sqrt(n) in 1-step and to 1/sqrt(2n) and 1/sqrt(2sqrt(n)) in n-steps)
  TAU_FACTOR="3"

# SURVIVAL SELECTION:

  # Survival selection method
  SURVIVAL_SELECTION_METHOD="mu_plus_lambda"
  SURVIVAL_SELECTION_METHOD="mu_comma_lambda"

#######################################################################################################################

python3 ES.py --function "$FUNCTION_N" --upperbound "$UPPER_BOUND" --lowerbound \
          "$LOWER_BOUND" --maximization "$MAXIMIZATION" --n "$N" --mu "$MU" --Lambda "$LAMBDA" --epsilon0 "$EPSILON0" \
          --numbergens "$MAX_GENERATIONS" --ro "$RO" --mutationmethod "$MUTATION_METHOD" --taufactor "$TAU_FACTOR" \
          --survivalselection "$SURVIVAL_SELECTION_METHOD" --solfitness "$MAX_FITNESS_VAL" \
