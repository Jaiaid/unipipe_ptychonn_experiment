#!/bin/bash

INTERVAL_COUNT=5

TRAIN_SCRIPT="multicontext_train.py"
INFER_SCRIPT="multicontext_infer.py"

TRAIN_CMD="python3 ${TRAIN_SCRIPT}"
INFER_CMD="python3 ${INFER_SCRIPT}"


for c in 5;
do
    for t in {4..16};
    do
        rm result_logs/multicontext/multicontext_${c}_${t}.csv
        rm result_logs/multicontext/multicontext_${c}_${t}_infer.log
        for  i in {1..1};
        do
            rm /dev/shm/unipipe_exp*
            rm model_multicontext/inctrained_interaval*

            eval "${TRAIN_CMD} -idur ${t} -icount ${c} > result_logs/multicontext/multicontext_${c}_${t}_train.log &"
            # eval "${TRAIN_CMD} -idur ${t} -icount ${c} &"
            TRAIN_PID=$!
            # eval "${INFER_CMD} -idur ${t} -icount ${c} -csvlog result_log.csv &"
            eval "${INFER_CMD} -idur ${t} -icount ${c} -csvlog result_logs/multicontext/multicontext_${c}_${t}.csv >> result_logs/multicontext/multicontext_${c}_${t}_infer.log &"
            INFER_PID=$!

            while ps -p ${TRAIN_PID} > /dev/null
            do
				# echo "training alive"
                sleep 5
            done
            while ps -p ${INFER_PID} > /dev/null
            do
				# echo "inference alive"
                sleep 5
            done

            echo "inference and training process are done"
            # eval "${TRAIN_CMD} -idur ${t} -icount ${c} &"
            # eval "${INFER_CMD} -idur ${t} -icount ${c}"
        done
    done
done
