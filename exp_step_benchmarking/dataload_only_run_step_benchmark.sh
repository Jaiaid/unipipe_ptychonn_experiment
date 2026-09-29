#!/bin/bash


SYSSTAT_DATADIR=sysstat_dataload_only_run
mkdir -p ${SYSSTAT_DATADIR}

for network in "resnet18" "resnet50"
do
    for bs in {1..64}
    do
        echo $network, $bs
        python3 monitor.py 0 ${SYSSTAT_DATADIR}/${network}_${bs}_sysstat.csv &
        MONITOR_PROCESS_PID=$!

        python3 dataload_only_benchmarking.py --network $network -bs $bs --eval
        
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
        python3 monitor.py 0 ${SYSSTAT_DATADIR}/${network}_${bs}_sysstat.csv &
        MONITOR_PROCESS_PID=$!

        python3 dataload_only_benchmarking_ptychonn.py --network $network -bs $bs --eval

        kill -2 ${MONITOR_PROCESS_PID}

        sleep 10
    done
done
