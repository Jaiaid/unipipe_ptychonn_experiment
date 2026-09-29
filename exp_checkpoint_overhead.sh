#!/bin/bash

set -x

# will first switch to this directory

ROOTDIR=.
GTDEFAULT= #--gtdefault
CKPTTEST=
CONSTANTTRAINBS= #--constant-trainbs
pushd $ROOTDIR
INTERVAL_COUNT=5

# DATA SAMPLE COUNT
# THIS IS IMPORTANT AS IT WILL DETERMINE THE DURATION FOR GIVEN INTERVAL COUNT
DSCOUNT=$((161*161))
DSCOUNT_PER_INTERVAL=$((DSCOUNT/INTERVAL_COUNT))
SKIPLINE=33

EXP_RESULT_DIR=result_logs/ckpt_overhead_exp
mkdir -p $EXP_RESULT_DIR

mkdir -p $EXP_RESULT_DIR/multicontext
for c in $INTERVAL_COUNT;do
    iprt=16
    deadlinemsec=200
    rate=1000

    for modeltype in 1.25M 5M 10M 20M 100M  
    do
        dur=$(($DSCOUNT_PER_INTERVAL/rate))
        set +x
        rm /dev/shm/*.raw
        rm -r /dev/shm/PTYCHO_STREAM*
        rm -r /dev/shm/MODEL_MULTICONTEXT*
        
        set -x

        python3 datastreamer_process.py -r $rate -dmsec $deadlinemsec -skipline $SKIPLINE &
        STREAM_PROCESS_PID=$!
        echo $STREAM_PROCESS_PID

        python3 phase_retrieval_mockprocess.py -ar $rate -gr $iprt -icount $c -idur $dur -d $deadlinemsec -skipline $SKIPLINE &
        IPR_PROCESS_PID=$!
        echo $IPR_PROCESS_PID

        bash multicontext_run.sh $c $dur $rate $deadlinemsec $SKIPLINE $iprt $modeltype
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
        mv ipr_generation_state.csv ${EXP_RESULT_DIR}/multicontext/multicontext_ipr_generation_state_${c}_${deadlinemsec}_${dur}_${rate}_${iprt}_${modeltype}.csv
        mv transmission_state.csv ${EXP_RESULT_DIR}/multicontext/multicontext_transmission_state_${c}_${deadlinemsec}_${dur}_${rate}_${iprt}_${modeltype}.csv
        mv tmp.csv ${EXP_RESULT_DIR}/multicontext/multicontext_${c}_${deadlinemsec}_${dur}_${rate}_${iprt}_${modeltype}.csv
        mv tmp_infer.log ${EXP_RESULT_DIR}/multicontext/multicontext_${c}_${deadlinemsec}_${dur}_${rate}_${iprt}_${modeltype}_infer.log
        mv tmp_train.log ${EXP_RESULT_DIR}/multicontext/multicontext_${c}_${deadlinemsec}_${dur}_${rate}_${iprt}_${modeltype}_train.log
        mv tmp_sysstat.csv ${EXP_RESULT_DIR}/multicontext/multicontext_sysstat_${c}_${dur}_${rate}_${iprt}_${modeltype}.csv
        mv /dev/shm/traindatalist_multicontext_${c}_${dur}_${rate}_${iprt}.csv ${EXP_RESULT_DIR}/multicontext/traindatalist_multicontext_${c}_${deadlinemsec}_${dur}_${rate}_${iprt}_${modeltype}.csv
        mv /dev/shm/inferdatalist_multicontext_${c}_${dur}_${rate}_${iprt}.csv ${EXP_RESULT_DIR}/multicontext/inferdatalist_multicontext_${c}_${deadlinemsec}_${dur}_${rate}_${iprt}_${modeltype}.csv
    done
done

popd

set +x
