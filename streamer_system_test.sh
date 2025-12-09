#!/bin/bash

for DRATE in 3000 2000 1000; do
    for DEADLINE in 1 2; do
        rm /dev/shm/*.raw
        rm -r /dev/shm/PTYCHO_STREAM*

        echo "==========================================================================="
        echo "Starting data streamer with rate: $DRATE and deadline: $DEADLINE"

        python3 datastreamer_process.py --no-sync --rate $DRATE -dmsec $DEADLINE --skip-line 33 &
        STREAMER_PID=$!
        python3 stream_shm_verifier.py --data_rate $DRATE --deadline $DEADLINE

        while ps -p ${STREAMER_PID} > /dev/null; do
            sleep 1
        done
        
        echo "Data streamer with PID: $STREAMER_PID has finished."
        echo "==========================================================================="
        exit
    done
done
