#!/bin/bash

# To find out which effect of different deadline for particular datarate and ipr throughput combination
# particular datarate and ipr throughput is chosen from experiment in `exp_perf_datacollection_deadline_tuning.sh`

set -x

# will first switch to this directory

ROOTDIR=.
GTDEFAULT= #--gtdefault
CKPTTEST=--allckpttest
CONSTANTTRAINBS= #--constant-trainbs
pushd $ROOTDIR
INTERVAL_COUNT=5
# set the particular datarate and ipr throughput
IPR_THROUGHPUT_TUNED=16

# DATA SAMPLE COUNT
# THIS IS IMPORTANT AS IT WILL DETERMINE THE DURATION FOR GIVEN INTERVAL COUNT
DSCOUNT=$((161*161))
DSCOUNT_PER_INTERVAL=$((DSCOUNT/INTERVAL_COUNT))
SKIPLINE=33

EXP_RESULT_DIR=result_logs/smalldataset_unipipebs
mkdir -p $EXP_RESULT_DIR

for c in $INTERVAL_COUNT;do
    iprt=160
    deadlinemsec=500

    for rate in 1000 2000 3000;do
        dur=$(($DSCOUNT_PER_INTERVAL/$rate))
    
        for bs in 1 2 4 8 16 32 64
        do
            set +x
            rm /dev/shm/*.raw
            rm -r /dev/shm/PTYCHO_STREAM*
            set -x

            python3 datastreamer_process.py -r $rate -dmsec $deadlinemsec -skipline $SKIPLINE &
            STREAM_PROCESS_PID=$!
            echo $STREAM_PROCESS_PID

            python3 phase_retrieval_mockprocess.py -ar $rate -gr $iprt -icount $c -idur $dur -d $deadlinemsec -skipline $SKIPLINE &
            IPR_PROCESS_PID=$!
            echo $IPR_PROCESS_PID

            # True for per iter validation activation
            bash unipipe_run.sh $c $dur $rate $deadlinemsec $SKIPLINE $iprt $bs
            set +x
            while ps -p ${STREAM_PROCESS_PID} > /dev/null
            do
                # echo "data streamer alive"
                sleep 1
            done

            while ps -p ${IPR_PROCESS_PID} > /dev/null
            do
                # echo "IPR process alive"
                sleep 1
            done
            set -x

            # move generated files for later analysis
            mv ipr_generation_state.csv ${EXP_RESULT_DIR}/unipipe_ipr_generation_state_${c}_${deadlinemsec}_${rate}_${iprt}_${bs}.csv
            mv transmission_state.csv ${EXP_RESULT_DIR}/unipipe_transmission_state_${c}_${deadlinemsec}_${rate}_${iprt}_${bs}.csv
            mv tmp.csv ${EXP_RESULT_DIR}/unipipe_${c}_${deadlinemsec}_${rate}_${iprt}_${bs}.csv
            mv tmp.log ${EXP_RESULT_DIR}/unipipe_${c}_${deadlinemsec}_${rate}_${iprt}_${bs}.log
            mv tmp_sysstat.csv ${EXP_RESULT_DIR}/unipipe_sysstat_${c}_${deadlinemsec}_${rate}_${iprt}_${bs}.csv
            mv /dev/shm/traindatalist_unipipe_${c}_${dur}_${rate}_${iprt}.csv ${EXP_RESULT_DIR}/traindatalist_unipipe_${c}_${deadlinemsec}_${dur}_${rate}_${iprt}_${bs}.csv
            mv /dev/shm/inferdatalist_unipipe_${c}_${dur}_${rate}_${iprt}.csv ${EXP_RESULT_DIR}/inferdatalist_unipipe_${c}_${deadlinemsec}_${dur}_${rate}_${iprt}_${bs}.csv
        done
    done
done

popd

set +x
