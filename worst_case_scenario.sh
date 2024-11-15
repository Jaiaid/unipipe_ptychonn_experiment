#!/bin/bash

INTERVAL_COUNT=5

TRAIN_INFER_SCRIPT="worst_case.py"

TRAIN_INFER_CMD="python3 ${TRAIN_INFER_SCRIPT}"

for c in 5;
do
    for t in {4..16};
    do
        rm result_logs/worst_case/worst_case_${c}_${t}.csv
        rm result_logs/worst_case/worst_case_${c}_${t}_traininfer.log
        for  i in {1..1};
        do
            rm /dev/shm/unipipe_exp*
            rm model_worst_case/*

            eval "${TRAIN_INFER_CMD} -idur ${t} -icount ${c} -csvlog result_logs/worst_case/worst_case_${c}_${t}.csv >> result_logs/worst_case/worst_case_${c}_${t}_traininfer.log"
            # eval "${TRAIN_INFER_CMD} -idur ${t} -icount ${c} -csvlog log.csv"
        done
    done
done
