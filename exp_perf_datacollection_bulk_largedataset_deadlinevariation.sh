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
DSCOUNT=$((186*186))
DSCOUNT_PER_INTERVAL=$((DSCOUNT/INTERVAL_COUNT))
SKIPLINE=0

EXP_RESULT_DIR=result_logs/largedataset_deadlinevariation
mkdir -p $EXP_RESULT_DIR

mkdir -p $EXP_RESULT_DIR/pretrained
for c in $INTERVAL_COUNT;do
    iprt=$IPR_THROUGHPUT_TUNED
    rate=3000
    dur=$(($DSCOUNT_PER_INTERVAL/$rate))

    for deadlinemsec in 80 160 320
    do
        set +x
        rm /dev/shm/*.raw
        rm -r /dev/shm/PTYCHO_STREAM*
        set -x

        python3 datastreamer_process.py -r $rate -dmsec $deadlinemsec -skipline $SKIPLINE --large-dataset &
        STREAM_PROCESS_PID=$!
        echo $STREAM_PROCESS_PID

        python3 phase_retrieval_mockprocess.py -ar $rate -gr $iprt -icount $c -idur $dur -d $deadlinemsec -skipline $SKIPLINE --large-dataset &
        IPR_PROCESS_PID=$!
        echo $IPR_PROCESS_PID

        bash pretrained_run.sh $c $dur $rate $deadlinemsec $SKIPLINE $iprt --large-dataset
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
        mv ipr_generation_state.csv ${EXP_RESULT_DIR}/pretrained/pretrained_ipr_generation_state_${c}_${deadlinemsec}_${rate}_${iprt}.csv
        mv transmission_state.csv ${EXP_RESULT_DIR}/pretrained/pretrained_transmission_state_${c}_${deadlinemsec}_${rate}_${iprt}.csv
        mv tmp.csv ${EXP_RESULT_DIR}/pretrained/pretrained_${c}_${deadlinemsec}_${rate}_${iprt}.csv
        mv tmp.log ${EXP_RESULT_DIR}/pretrained/pretrained_${c}_${deadlinemsec}_${rate}_${iprt}.log
        mv tmp_sysstat.csv ${EXP_RESULT_DIR}/pretrained/pretrained_sysstat_${c}_${deadlinemsec}_${rate}_${iprt}.csv
        mv /dev/shm/inferdatalist_pretrained_${c}_${dur}_${rate}_${iprt}.csv ${EXP_RESULT_DIR}/pretrained/inferdatalist_pretrained_${c}_${deadlinemsec}_${rate}_${iprt}.csv
    done
done
exit

mkdir -p $EXP_RESULT_DIR/unipipe
for c in $INTERVAL_COUNT;do
    iprt=$IPR_THROUGHPUT_TUNED
    rate=3000
    dur=$(($DSCOUNT_PER_INTERVAL/$rate))
    
    for deadlinemsec in 80 160 320
    do
        set +x
        rm /dev/shm/*.raw
        rm -r /dev/shm/PTYCHO_STREAM*
        set -x

        python3 datastreamer_process.py -r $rate -dmsec $deadlinemsec -skipline $SKIPLINE -largedataset &
        STREAM_PROCESS_PID=$!
        echo $STREAM_PROCESS_PID

        python3 phase_retrieval_mockprocess.py -ar $rate -gr $iprt -icount $c -idur $dur -d $deadlinemsec -skipline $SKIPLINE -largedataset &
        IPR_PROCESS_PID=$!
        echo $IPR_PROCESS_PID

        # True for per iter validation activation
        bash unipipe_run.sh $c $dur $rate $deadlinemsec $SKIPLINE $iprt 1.25M --large-dataset
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
        mv ipr_generation_state.csv ${EXP_RESULT_DIR}/unipipe/unipipe_ipr_generation_state_${c}_${deadlinemsec}_${rate}_${iprt}.csv
        mv transmission_state.csv ${EXP_RESULT_DIR}/unipipe/unipipe_transmission_state_${c}_${deadlinemsec}_${rate}_${iprt}.csv
        mv tmp.csv ${EXP_RESULT_DIR}/unipipe/unipipe_${c}_${deadlinemsec}_${rate}_${iprt}.csv
        mv tmp.log ${EXP_RESULT_DIR}/unipipe/unipipe_${c}_${deadlinemsec}_${rate}_${iprt}.log
        mv tmp_sysstat.csv ${EXP_RESULT_DIR}/unipipe/unipipe_sysstat_${c}_${deadlinemsec}_${rate}_${iprt}.csv
        mv /dev/shm/traindatalist_unipipe_${c}_${dur}_${rate}_${iprt}.csv ${EXP_RESULT_DIR}/unipipe/traindatalist_unipipe_${c}_${deadlinemsec}_${dur}_${rate}_${iprt}.csv
        mv /dev/shm/inferdatalist_unipipe_${c}_${dur}_${rate}_${iprt}.csv ${EXP_RESULT_DIR}/unipipe/inferdatalist_unipipe_${c}_${deadlinemsec}_${dur}_${rate}_${iprt}.csv
    done
done
exit

# mkdir -p $EXP_RESULT_DIR/unipipe_valid
# for c in $INTERVAL_COUNT;do
#     dur=$(($DSCOUNT_PER_INTERVAL/DATARATE_TUNED))
#     rate=$DATARATE_TUNED
#     iprt=$IPR_THROUGHPUT_TUNED

#     for deadlinemsec in 50000 25000 12000 6000 3000 1000 500
#     do
#         set +x
#         rm /dev/shm/*.raw
#         rm -r /dev/shm/PTYCHO_STREAM*
#         set -x

#         python3 datastreamer_process.py -r $rate -dmsec $deadlinemsec -skipline $SKIPLINE &
#         STREAM_PROCESS_PID=$!
#         echo $STREAM_PROCESS_PID

#         python3 phase_retrieval_mockprocess.py -ar $rate -gr $iprt -icount $c -idur $dur -d $deadlinemsec -skipline $SKIPLINE &
#         IPR_PROCESS_PID=$!
#         echo $IPR_PROCESS_PID

#         # True for per iter validation activation
#         bash unipipe_run.sh $c $dur $rate $deadlinemsec $SKIPLINE $iprt 1.25M --large-dataset
#         set +x
#         while ps -p ${STREAM_PROCESS_PID} > /dev/null
#         do
#             # echo "data streamer alive"
#             sleep 1
#         done

#         while ps -p ${IPR_PROCESS_PID} > /dev/null
#         do
#             # echo "IPR process alive"
#             sleep 1
#         done
#         set -x

#         # move generated files for later analysis
#         mv ipr_generation_state.csv ${EXP_RESULT_DIR}/unipipe_valid/unipipe_valid_ipr_generation_state_${c}_${deadlinemsec}_${dur}_${rate}_${iprt}.csv
#         mv transmission_state.csv ${EXP_RESULT_DIR}/unipipe_valid/unipipe_valid_transmission_state_${c}_${deadlinemsec}_${dur}_${rate}_${iprt}.csv
#         mv tmp.csv ${EXP_RESULT_DIR}/unipipe_valid/unipipe_valid_${c}_${deadlinemsec}_${dur}_${rate}_${iprt}.csv
#         mv tmp.log ${EXP_RESULT_DIR}/unipipe_valid/unipipe_valid_${c}_${deadlinemsec}_${dur}_${rate}_${iprt}.log
#         mv tmp_sysstat.csv ${EXP_RESULT_DIR}/unipipe_valid/unipipe_valid_sysstat_${c}_${dur}_${rate}_${iprt}.csv
#         mv /dev/shm/traindatalist_unipipe_${c}_${dur}_${rate}_${iprt}.csv ${EXP_RESULT_DIR}/unipipe_valid/traindatalist_unipipe_${c}_${deadlinemsec}_${dur}_${rate}_${iprt}.csv
#         mv /dev/shm/inferdatalist_unipipe_${c}_${dur}_${rate}_${iprt}.csv ${EXP_RESULT_DIR}/unipipe_valid/inferdatalist_unipipe_${c}_${deadlinemsec}_${dur}_${rate}_${iprt}.csv
#     done
# done

mkdir -p $EXP_RESULT_DIR/multicontext
for c in $INTERVAL_COUNT;do
    iprt=$IPR_THROUGHPUT_TUNED
    rate=3000
    dur=$(($DSCOUNT_PER_INTERVAL/$rate))

    for deadlinemsec in 80 160 320;
    do
        set +x
        rm /dev/shm/*.raw
        rm -r /dev/shm/PTYCHO_STREAM*
        rm -r /dev/shm/MODEL_MULTICONTEXT*
        
        set -x

        python3 datastreamer_process.py -r $rate -dmsec $deadlinemsec -skipline $SKIPLINE --large-dataset &
        STREAM_PROCESS_PID=$!
        echo $STREAM_PROCESS_PID

        python3 phase_retrieval_mockprocess.py -ar $rate -gr $iprt -icount $c -idur $dur -d $deadlinemsec -skipline $SKIPLINE --large-dataset &
        IPR_PROCESS_PID=$!
        echo $IPR_PROCESS_PID

        bash multicontext_run.sh $c $dur $rate $deadlinemsec $SKIPLINE $iprt 1.25M --large-dataset 
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
        mv ipr_generation_state.csv ${EXP_RESULT_DIR}/multicontext/multicontext_ipr_generation_state_${c}_${deadlinemsec}_${rate}_${iprt}.csv
        mv transmission_state.csv ${EXP_RESULT_DIR}/multicontext/multicontext_transmission_state_${c}_${deadlinemsec}_${rate}_${iprt}.csv
        mv tmp.csv ${EXP_RESULT_DIR}/multicontext/multicontext_${c}_${deadlinemsec}_${rate}_${iprt}.csv
        mv tmp_infer.log ${EXP_RESULT_DIR}/multicontext/multicontext_${c}_${deadlinemsec}_${rate}_${iprt}_infer.log
        mv tmp_train.log ${EXP_RESULT_DIR}/multicontext/multicontext_${c}_${deadlinemsec}_${rate}_${iprt}_train.log
        mv tmp_sysstat.csv ${EXP_RESULT_DIR}/multicontext/multicontext_sysstat_${c}_${deadlinemsec}_${rate}_${iprt}.csv
        mv /dev/shm/traindatalist_multicontext_${c}_${dur}_${rate}_${iprt}.csv ${EXP_RESULT_DIR}/multicontext/traindatalist_multicontext_${c}_${deadlinemsec}_${dur}_${rate}_${iprt}.csv
        mv /dev/shm/inferdatalist_multicontext_${c}_${dur}_${rate}_${iprt}.csv ${EXP_RESULT_DIR}/multicontext/inferdatalist_multicontext_${c}_${deadlinemsec}_${dur}_${rate}_${iprt}.csv
    done
done

popd

set +x
