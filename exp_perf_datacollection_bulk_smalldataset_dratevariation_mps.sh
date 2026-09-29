#!/bin/bash

# Multicontext with MPS thread percentage splits for inference:training
# Runs two configurations: 50:50 and 70:30 (inference:training)
# Results saved under result_logs/smalldataset_dratevariation/multicontext_mps/mps_<INFER>_<TRAIN>/

set -x

source /home/jm5071/unipipe_experiment/venv/bin/activate

ROOTDIR=.
GTDEFAULT= #--gtdefault
CKPTTEST=--allckpttest
CONSTANTTRAINBS= #--constant-trainbs
pushd $ROOTDIR

INTERVAL_COUNT=5
IPR_THROUGHPUT_TUNED=16
DEADLINEMSEC=80
GTCOUNT_DPINPUT=1
EXP_REPEAT=6

# DATA SAMPLE COUNT
# THIS IS IMPORTANT AS IT WILL DETERMINE THE DURATION FOR GIVEN INTERVAL COUNT
DSCOUNT=$((161*161))
DSCOUNT_PER_INTERVAL=$((DSCOUNT/INTERVAL_COUNT))
SKIPLINE=33

EXP_RESULT_DIR=result_logs/smalldataset_dratevariation

# MPS splits: "INFER_PCT TRAIN_PCT"
# 50:50 => each process gets up to 50% of GPU SM threads
# 70:30 => inference gets 70%, training gets 30%
# 90:10 => inference gets 90%, training gets 10%
for MPS_SPLIT in "10 90" "30 70" "50 50" "70 30" "90 10"; do
    MPS_INFER_PCT=$(echo $MPS_SPLIT | awk '{print $1}')
    MPS_TRAIN_PCT=$(echo $MPS_SPLIT | awk '{print $2}')

    SUBDIR=$EXP_RESULT_DIR/multicontext_mps_${MPS_INFER_PCT}_${MPS_TRAIN_PCT}
    mkdir -p $SUBDIR

    for c in $INTERVAL_COUNT; do
        iprt=$IPR_THROUGHPUT_TUNED
        deadlinemsec=$DEADLINEMSEC

        for repeat in $(seq 1 $EXP_REPEAT); do
            for rate in 1000 2000 3000 4000 5000; do
                dur=$(($DSCOUNT_PER_INTERVAL/$rate))

                set +x
                rm /dev/shm/*.raw
                rm -r /dev/shm/PTYCHO_STREAM*
                rm -r /dev/shm/MODEL_MULTICONTEXT*
                rm -r /dev/shm/*

                set -x
                python3 datastreamer_process.py -r $rate -dmsec $deadlinemsec -skipline $SKIPLINE &
                STREAM_PROCESS_PID=$!
                echo $STREAM_PROCESS_PID

                python3 phase_retrieval_mockprocess.py -ar $rate -gr $iprt -icount $c -idur $dur -d $deadlinemsec -skipline $SKIPLINE -unipipedp &
                IPR_PROCESS_PID=$!
                echo $IPR_PROCESS_PID

                bash multicontext_mps_run.sh $c $dur $rate $deadlinemsec $SKIPLINE $iprt $MPS_INFER_PCT $MPS_TRAIN_PCT

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
                cat ipr_generation_state.csv >> ${SUBDIR}/multicontext_mps_${MPS_INFER_PCT}_${MPS_TRAIN_PCT}_ipr_generation_state_${c}_${deadlinemsec}_${rate}_${iprt}.csv
                cat transmission_state.csv >> ${SUBDIR}/multicontext_mps_${MPS_INFER_PCT}_${MPS_TRAIN_PCT}_transmission_state_${c}_${deadlinemsec}_${rate}_${iprt}.csv
                cat tmp.csv >> ${SUBDIR}/multicontext_mps_${MPS_INFER_PCT}_${MPS_TRAIN_PCT}_${c}_${deadlinemsec}_${rate}_${iprt}.csv
                mv tmp_infer.log ${SUBDIR}/multicontext_mps_${MPS_INFER_PCT}_${MPS_TRAIN_PCT}_${c}_${deadlinemsec}_${rate}_${iprt}_infer.log
                mv tmp_train.log ${SUBDIR}/multicontext_mps_${MPS_INFER_PCT}_${MPS_TRAIN_PCT}_${c}_${deadlinemsec}_${rate}_${iprt}_train.log
                mv tmp_sysstat.csv ${SUBDIR}/multicontext_mps_${MPS_INFER_PCT}_${MPS_TRAIN_PCT}_sysstat_${c}_${deadlinemsec}_${rate}_${iprt}.csv
                mv /dev/shm/traindatalist_multicontext_${c}_${dur}_${rate}_${iprt}.csv ${SUBDIR}/traindatalist_multicontext_mps_${MPS_INFER_PCT}_${MPS_TRAIN_PCT}_${c}_${deadlinemsec}_${dur}_${rate}_${iprt}.csv
                mv /dev/shm/inferdatalist_multicontext_${c}_${dur}_${rate}_${iprt}.csv ${SUBDIR}/inferdatalist_multicontext_mps_${MPS_INFER_PCT}_${MPS_TRAIN_PCT}_${c}_${deadlinemsec}_${dur}_${rate}_${iprt}.csv
                mv stitched_construction.png ${SUBDIR}/stitched_construction_${c}_${deadlinemsec}_${rate}_${iprt}.png
                mv nn_generated_data_ph_error.png ${SUBDIR}/nn_generated_data_ph_error_${c}_${deadlinemsec}_${rate}_${iprt}.png
                mv nn_generated_data_amp_error.png ${SUBDIR}/nn_generated_data_amp_error_${c}_${deadlinemsec}_${rate}_${iprt}.png
            done
        done
    done
done

popd

set +x
