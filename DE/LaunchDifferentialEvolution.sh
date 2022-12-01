#!/bin/bash

## Shell script that configures all configurable options and parameters of the DE ##

### GENERAL PARAMETERS:

  # Function to minimize. The problem asks to use both sphere and Ackeley
  # functions, so only them have been implememted
  FUNCTION_N="Sphere"
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

  # Lenght of the genotype
  N="10"

  # PL (population length)
  PL="30"

  # F mutation factor
  F="0.4"

  # CR Crossover parameter
  CR="0.5"

  # Number of generations to run
  MAX_GENERATIONS="1200"

  # Minimum fitness value to consider an execution as successful
  MAX_FITNESS_VAL="0.000001"

### METHODS TO USE ON EACH STEP:

  # Variant of DE
  DE_VARIANT="DE/best/1/bin"
  DE_VARIANT="DE/rand/1/bin"


#######################################################################################################################

python3 DE.py --function "$FUNCTION_N" --upperbound "$UPPER_BOUND" --lowerbound \
          "$LOWER_BOUND" --n "$N" --pl "$PL" -f "$F" --cr "$CR" --numbergens "$MAX_GENERATIONS" \
          --devariant "$DE_VARIANT" --solfitness "$MAX_FITNESS_VAL"
