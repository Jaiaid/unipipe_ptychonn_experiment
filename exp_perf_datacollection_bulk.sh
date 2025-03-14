#!/bin/bash

set -x

# will first switch to this directory

ROOTDIR=.
GTDEFAULT= #--gtdefault
CKPTTEST=--allckpttest
CONSTANTTRAINBS= #--constant-trainbs
pushd $ROOTDIR
INTERVAL_COUNT=5

# DATA SAMPLE COUNT
# THIS IS IMPORTANT AS IT WILL DETERMINE THE DURATION FOR GIVEN INTERVAL COUNT
DSCOUNT=$((161*161))
DSCOUNT_PER_INTERVAL=$((DSCOUNT/INTERVAL_COUNT))
IPR_THROUGPUT=16


EXP_RESULT_DIR=result_logs/bulk
mkdir -p $EXP_RESULT_DIR

mkdir -p $EXP_RESULT_DIR/pretrained
for c in $INTERVAL_COUNT;do
    rate=100
    while [ $rate -le 200 ];
    do
        dur=$(($DSCOUNT_PER_INTERVAL/$rate))
        for accumulationallow in 100;do
            set +x
            rm /dev/shm/*.raw
            rm -r /dev/shm/PTYCHO_STREAM*
            set -x

            deadlinemsec=$((1000*$dur))
            python3 ptychonn/datastreamer_process.py -r $rate -dmsec $deadlinemsec &
            STREAM_PROCESS_PID=$!
            echo $STREAM_PROCESS_PID

            python3 ptychonn/phase_retrieval_mockprocess.py -ar $rate -gr $IPR_THROUGPUT -icount $c -d $deadlinemsec &
            IPR_PROCESS_PID=$!
            echo $IPR_PROCESS_PID

            bash pretrained_run.sh $c $dur $rate $deadlinemsec $IPR_THROUGPUT
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
            mv ipr_generation_state.csv ${EXP_RESULT_DIR}/unipipe/unipipe_ipr_generation_state_${c}_${dur}_${rate}_${accumulationallow}.csv
            mv transmission_state.csv ${EXP_RESULT_DIR}/unipipe/unipipe_transmission_state_${c}_${dur}_${rate}_${accumulationallow}.csv
            mv tmp.csv ${EXP_RESULT_DIR}/pretrained/pretrained_${c}_${dur}_${rate}_${accumulationallow}.csv
            mv tmp.log ${EXP_RESULT_DIR}/pretrained/pretrained_${c}_${dur}_${rate}_${accumulationallow}.log
            mv tmp_sysstat.csv ${EXP_RESULT_DIR}/pretrained/pretrained_sysstat_${c}_${dur}_${rate}_${accumulationallow}.csv 
        done
        rate=$(($rate*2))
    done
done

mkdir -p $EXP_RESULT_DIR/unipipe
for c in $INTERVAL_COUNT;do
    rate=100
    while [ $rate -le 200 ];
    do
        dur=$(($DSCOUNT_PER_INTERVAL/$rate))
        for accumulationallow in 100;do
            set +x
            rm /dev/shm/*.raw
            rm -r /dev/shm/PTYCHO_STREAM*
            set -x

            deadlinemsec=$((1000*$dur))
            python3 ptychonn/datastreamer_process.py -r $rate -dmsec $deadlinemsec &
            STREAM_PROCESS_PID=$!
            echo $STREAM_PROCESS_PID

            python3 ptychonn/phase_retrieval_mockprocess.py -ar $rate -gr $IPR_THROUGPUT -icount $c -d $deadlinemsec &
            IPR_PROCESS_PID=$!
            echo $IPR_PROCESS_PID

            bash unipipe_run.sh $c $dur $rate $deadlinemsec $IPR_THROUGPUT
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
            mv ipr_generation_state.csv ${EXP_RESULT_DIR}/unipipe/unipipe_ipr_generation_state_${c}_${dur}_${rate}_${accumulationallow}.csv
            mv transmission_state.csv ${EXP_RESULT_DIR}/unipipe/unipipe_transmission_state_${c}_${dur}_${rate}_${accumulationallow}.csv
            mv tmp.csv ${EXP_RESULT_DIR}/unipipe/unipipe_${c}_${dur}_${rate}_${accumulationallow}.csv
            mv tmp.log ${EXP_RESULT_DIR}/unipipe/unipipe_${c}_${dur}_${rate}_${accumulationallow}.log
            mv tmp_sysstat.csv ${EXP_RESULT_DIR}/unipipe/unipipe_sysstat_${c}_${dur}_${rate}_${accumulationallow}.csv
        done
        rate=$(($rate*2))
    done
done

mkdir -p $EXP_RESULT_DIR/multicontext
for c in $INTERVAL_COUNT;do
    rate=100
    while [ $rate -le 200 ];
    do
        dur=$(($DSCOUNT_PER_INTERVAL/rate))
        for accumulationallow in 100;do
            set +x
            rm /dev/shm/*.raw
            rm -r /dev/shm/PTYCHO_STREAM*
            set -x

            deadlinemsec=$((1000*$dur))
            python3 ptychonn/datastreamer_process.py -r $rate -dmsec $deadlinemsec &
            STREAM_PROCESS_PID=$!
            echo $STREAM_PROCESS_PID

            python3 ptychonn/phase_retrieval_mockprocess.py -ar $rate -gr $IPR_THROUGPUT -icount $c -d $deadlinemsec &
            IPR_PROCESS_PID=$!
            echo $IPR_PROCESS_PID

            bash multicontext_run.sh $c $dur $rate $deadlinemsec $IPR_THROUGPUT
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
            mv ipr_generation_state.csv ${EXP_RESULT_DIR}/unipipe/unipipe_ipr_generation_state_${c}_${dur}_${rate}_${accumulationallow}.csv
            mv transmission_state.csv ${EXP_RESULT_DIR}/unipipe/unipipe_transmission_state_${c}_${dur}_${rate}_${accumulationallow}.csv
            mv tmp.csv ${EXP_RESULT_DIR}/multicontext/multicontext_${c}_${dur}_${rate}_${accumulationallow}.csv
            mv tmp_infer.log ${EXP_RESULT_DIR}/multicontext/multicontext_${c}_${dur}_${rate}_${accumulationallow}_infer.log
            mv tmp_train.log ${EXP_RESULT_DIR}/multicontext/multicontext_${c}_${dur}_${rate}_${accumulationallow}_train.log
            mv tmp_sysstat.csv ${EXP_RESULT_DIR}/multicontext/multicontext_sysstat_${c}_${dur}_${rate}_${accumulationallow}.csv
        done
        rate=$(($rate*2))
    done
done

popd

set +x
