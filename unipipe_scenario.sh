#!/bin/bash

INTERVAL_COUNT=5

TRAIN_INFER_SCRIPT="unipipe.py"

TRAIN_INFER_CMD="python3 ${TRAIN_INFER_SCRIPT}"


for c in 5;
do
    for t in {4..16};
    do
        rm result_logs/unipipe/unipipe_${c}_${t}.csv
        rm result_logs/unipipe/unipipe_${c}_${t}_traininfer.log
        for  i in {1..1};
        do
            rm /dev/shm/unipipe_exp*
            rm model_unipipe/*

            eval "${TRAIN_INFER_CMD} -idur ${t} -icount ${c} -csvlog result_logs/unipipe/unipipe_${c}_${t}.csv >> result_logs/unipipe/unipipe_${c}_${t}_traininfer.log"
        done
    done
done
