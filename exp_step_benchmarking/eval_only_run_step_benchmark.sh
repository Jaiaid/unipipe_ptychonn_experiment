#!/bin/bash

mkdir -p sysstat_data_eval_only_run

for network in "resnet18" "resnet50"
do
    for bs in {1..64}
    do
        echo $network, $bs
        python3 monitor.py 0 sysstat_data_eval_only_run/${network}_${bs}_sysstat.csv &
        MONITOR_PROCESS_PID=$!

        python3 coincidental_train_infer_step_benchmakring.py --network $network -bs $bs --eval
        
        # kill the monitor process
        kill -2 ${MONITOR_PROCESS_PID}

        sleep 10
    done
done

for network in "ptychonn"
do
    for bs in {1..64}
    do
        echo $network, $bs
        python3 monitor.py 0 sysstat_data_eval_only_run/${network}_${bs}_sysstat.csv &
        MONITOR_PROCESS_PID=$!

        python3 coincidental_train_infer_step_benchmarking_ptychonn.py --network $network -bs $bs --eval

        kill -2 ${MONITOR_PROCESS_PID}

        sleep 10
    done
done
