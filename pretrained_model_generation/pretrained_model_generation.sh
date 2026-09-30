#!/bin/bash

# you need to run this script from the parent of pretrained_model_generation directory
SCRIPT_PATH=pretrained_model_generation/train.py

# This script is used to train the pretrained models for the 1.25M, 5M, 10M, and 20M parameter versions of the model.
# It will save the best model to the pretrained_model directory with a name that includes the model type and dataset size.

# Train the 1.25M parameter model on the small dataset
python $SCRIPT_PATH --model-type 1.25M --linecount 33 --model-dir pretrained_model_1.25M_small
python $SCRIPT_PATH --model-type 1.25M --linecount 40 --large-dataset --model-dir pretrained_model_1.25M_large