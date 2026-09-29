#!/bin/bash

# This script runs benchmarking experiments for different model types in the PtychoNN project.
# It iterates over a predefined list of model types and executes the corresponding Python script
# with the specified model type as an argument.
# Usage: ./exp_collect_model_profile.sh
# Make sure to have the necessary Python environment set up before running this script.
# "1.25M" "5M" "10M" "20M" "100M" "200M"
for MODEL_TYPE in "1.25M" "5M" "10M" "20M" "100M" "200M";do
    python3 exp_step_benchmarking_ptychonn.py --model-type $MODEL_TYPE
done