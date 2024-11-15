#!/bin/bash

# this script is to short test datastreamer's sanity
# we will use unipipe to test datastreamer's sanity

TRAIN_INFER_SCRIPT="unipipe.py"
TRAIN_INFER_CMD="python3 ${TRAIN_INFER_SCRIPT}"

INTERVAL_COUNT=5
INTERVAL_DURATION=10

eval "${TRAIN_INFER_CMD} --deadline 30 --datarate 259 -idur ${INTERVAL_DURATION} -icount ${INTERVAL_COUNT} -csvlog result_logs/datastreamer_test/dstest_${INTERVAL_COUNT}_${INTERVAL_DURATION}.csv"
