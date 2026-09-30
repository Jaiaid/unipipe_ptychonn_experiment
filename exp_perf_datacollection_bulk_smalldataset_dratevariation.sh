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
EXPERIMENT_REPEAT_COUNT=6
INTERVAL_COUNT=5
# set the particular datarate and ipr throughput
IPR_THROUGHPUT_TUNED=16
DEADLINEMSEC=80
GTCOUNT_DPINPUT=1

# DATA SAMPLE COUNT
# THIS IS IMPORTANT AS IT WILL DETERMINE THE DURATION FOR GIVEN INTERVAL COUNT
DSCOUNT=$((161*161))
DSCOUNT_PER_INTERVAL=$((DSCOUNT/INTERVAL_COUNT))
SKIPLINE=33

EXP_RESULT_DIR=result_logs/smalldataset_dratevariation
mkdir -p $EXP_RESULT_DIR

mkdir -p $EXP_RESULT_DIR/unipipe_dp
for c in $INTERVAL_COUNT;do
    iprt=$IPR_THROUGHPUT_TUNED
    deadlinemsec=$DEADLINEMSEC

   # 1000 2000 3000 4000 5000
   # 6000 7000 8000 9000 10000
    for rate in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000;do
        for repeat in $(seq 1 $EXPERIMENT_REPEAT_COUNT); do
            dur=$(($DSCOUNT_PER_INTERVAL/$rate))
        
            set +x
            rm /dev/shm/*.raw
            # rsync -a --delete empty_folder/ /dev/shm
            rm -r /dev/shm/PTYCHO_STREAM*
            rm -r /dev/shm/*
            set -x

            python3 datastreamer_process.py -r $rate -dmsec $deadlinemsec -skipline $SKIPLINE &
            STREAM_PROCESS_PID=$!
            echo $STREAM_PROCESS_PID

            python3 phase_retrieval_mockprocess.py -ar $rate -gr $iprt -icount $c -idur $dur -d $deadlinemsec -skipline $SKIPLINE -unipipedp &
            IPR_PROCESS_PID=$!
            echo $IPR_PROCESS_PID

            # True for per iter validation activation
            bash unipipe_dp_run.sh $c $dur $rate $deadlinemsec $SKIPLINE $iprt 64 1.25M $GTCOUNT_DPINPUT
            
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

            # append results for later analysis
            cat ipr_generation_state.csv >> ${EXP_RESULT_DIR}/unipipe_dp/unipipe_dp_ipr_generation_state_${c}_${deadlinemsec}_${rate}_${iprt}.csv
            cat transmission_state.csv >> ${EXP_RESULT_DIR}/unipipe_dp/unipipe_dp_transmission_state_${c}_${deadlinemsec}_${rate}_${iprt}.csv
            cat tmp.csv >> ${EXP_RESULT_DIR}/unipipe_dp/unipipe_dp_${c}_${deadlinemsec}_${rate}_${iprt}.csv
            mv tmp.log ${EXP_RESULT_DIR}/unipipe_dp/unipipe_dp_${c}_${deadlinemsec}_${rate}_${iprt}.log
            mv tmp_sysstat.csv ${EXP_RESULT_DIR}/unipipe_dp/unipipe_dp_sysstat_${c}_${deadlinemsec}_${rate}_${iprt}.csv
            mv /dev/shm/traindatalist_unipipe_dp_${c}_${dur}_${rate}_${iprt}.csv ${EXP_RESULT_DIR}/unipipe_dp/traindatalist_unipipe_dp_${c}_${deadlinemsec}_${dur}_${rate}_${iprt}.csv
            mv /dev/shm/inferdatalist_unipipe_dp_${c}_${dur}_${rate}_${iprt}.csv ${EXP_RESULT_DIR}/unipipe_dp/inferdatalist_unipipe_dp_${c}_${deadlinemsec}_${dur}_${rate}_${iprt}.csv
            mv stitched_construction.png ${EXP_RESULT_DIR}/unipipe_dp/stitched_construction_${c}_${deadlinemsec}_${rate}_${iprt}.png
            mv nn_generated_data_ph_error.png ${EXP_RESULT_DIR}/unipipe_dp/nn_generated_data_ph_error_${c}_${deadlinemsec}_${rate}_${iprt}.png
            mv nn_generated_data_amp_error.png ${EXP_RESULT_DIR}/unipipe_dp/nn_generated_data_amp_error_${c}_${deadlinemsec}_${rate}_${iprt}.png
        done
    done
done

mkdir -p $EXP_RESULT_DIR/pretrained_noipr
for c in $INTERVAL_COUNT;do
    iprt=$IPR_THROUGHPUT_TUNED
    deadlinemsec=$DEADLINEMSEC

    for rate in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000;do
        for repeat in $(seq 1 $EXPERIMENT_REPEAT_COUNT); do
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

            # True for per iter validation activation
            bash pretrained_noipr_run.sh $c $dur $rate $deadlinemsec $SKIPLINE $iprt
            set +x
            while ps -p ${STREAM_PROCESS_PID} > /dev/null
            do
                # echo "data streamer alive"
                sleep 1
            done

            set -x

            # move generated files for later analysis
            cat transmission_state.csv >> ${EXP_RESULT_DIR}/pretrained_noipr/pretrained_noipr_transmission_state_${c}_${deadlinemsec}_${rate}_${iprt}.csv
            cat tmp.csv >> ${EXP_RESULT_DIR}/pretrained_noipr/pretrained_noipr_${c}_${deadlinemsec}_${rate}_${iprt}.csv
            mv tmp.log ${EXP_RESULT_DIR}/pretrained_noipr/pretrained_noipr_${c}_${deadlinemsec}_${rate}_${iprt}.log
            mv tmp_sysstat.csv ${EXP_RESULT_DIR}/pretrained_noipr/pretrained_noipr_sysstat_${c}_${deadlinemsec}_${rate}_${iprt}.csv
            mv /dev/shm/inferdatalist_pretrained_${c}_${dur}_${rate}_${iprt}.csv ${EXP_RESULT_DIR}/pretrained_noipr/inferdatalist_pretrained_noipr_${c}_${deadlinemsec}_${rate}_${iprt}.csv
            mv stitched_construction.png ${EXP_RESULT_DIR}/pretrained_noipr/stitched_construction_${c}_${deadlinemsec}_${rate}_${iprt}.png
            mv nn_generated_data_ph_error.png ${EXP_RESULT_DIR}/pretrained_noipr/nn_generated_data_ph_error_${c}_${deadlinemsec}_${rate}_${iprt}.png
            mv nn_generated_data_amp_error.png ${EXP_RESULT_DIR}/pretrained_noipr/nn_generated_data_amp_error_${c}_${deadlinemsec}_${rate}_${iprt}.png
        done
    done
done

# mkdir -p $EXP_RESULT_DIR/pretrained
# for c in $INTERVAL_COUNT;do
#     iprt=$IPR_THROUGHPUT_TUNED
#     deadlinemsec=$DEADLINEMSEC

#     for rate in 1000 2000 3000 4000 5000;do
#         dur=$(($DSCOUNT_PER_INTERVAL/$rate))
    
#         set +x
#         rm /dev/shm/*.raw
#         rm -r /dev/shm/PTYCHO_STREAM*
#         rm -r /dev/shm/MODEL_MULTICONTEXT*
#         rm -r /dev/shm/*
#         set -x

#         python3 datastreamer_process.py -r $rate -dmsec $deadlinemsec -skipline $SKIPLINE &
#         STREAM_PROCESS_PID=$!
#         echo $STREAM_PROCESS_PID

#         python3 phase_retrieval_mockprocess.py -ar $rate -gr $iprt -icount $c -idur $dur -d $deadlinemsec -skipline $SKIPLINE -pretrained &
#         IPR_PROCESS_PID=$!
#         echo $IPR_PROCESS_PID

#         # True for per iter validation activation
#         bash pretrained_run.sh $c $dur $rate $deadlinemsec $SKIPLINE $iprt
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
#         mv ipr_generation_state.csv ${EXP_RESULT_DIR}/pretrained/pretrained_ipr_generation_state_${c}_${deadlinemsec}_${rate}_${iprt}.csv
#         mv transmission_state.csv ${EXP_RESULT_DIR}/pretrained/pretrained_transmission_state_${c}_${deadlinemsec}_${rate}_${iprt}.csv
#         mv tmp.csv ${EXP_RESULT_DIR}/pretrained/pretrained_${c}_${deadlinemsec}_${rate}_${iprt}.csv
#         mv tmp.log ${EXP_RESULT_DIR}/pretrained/pretrained_${c}_${deadlinemsec}_${rate}_${iprt}.log
#         mv tmp_sysstat.csv ${EXP_RESULT_DIR}/pretrained/pretrained_sysstat_${c}_${deadlinemsec}_${rate}_${iprt}.csv
#         mv /dev/shm/inferdatalist_pretrained_${c}_${dur}_${rate}_${iprt}.csv ${EXP_RESULT_DIR}/pretrained/inferdatalist_pretrained_${c}_${deadlinemsec}_${rate}_${iprt}.csv
#         mv stitched_construction.png ${EXP_RESULT_DIR}/pretrained/stitched_construction_${c}_${deadlinemsec}_${rate}_${iprt}.png
#         mv nn_generated_data_ph_error.png ${EXP_RESULT_DIR}/pretrained/nn_generated_data_ph_error_${c}_${deadlinemsec}_${rate}_${iprt}.png
#         mv nn_generated_data_amp_error.png ${EXP_RESULT_DIR}/pretrained/nn_generated_data_amp_error_${c}_${deadlinemsec}_${rate}_${iprt}.png
#     done
# done

# mkdir -p $EXP_RESULT_DIR/unipipe
# for c in $INTERVAL_COUNT;do
#     iprt=$IPR_THROUGHPUT_TUNED
#     deadlinemsec=$DEADLINEMSEC

#     for rate in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000;do
#         dur=$(($DSCOUNT_PER_INTERVAL/$rate))
    
#         set +x
#         rm /dev/shm/*.raw
#         rm -r /dev/shm/PTYCHO_STREAM*
#         rm -r /dev/shm/MODEL_MULTICONTEXT*
#         rm -r /dev/shm/*
        
#         set -x
#         python3 datastreamer_process.py -r $rate -dmsec $deadlinemsec -skipline $SKIPLINE &
#         STREAM_PROCESS_PID=$!
#         echo $STREAM_PROCESS_PID

#         python3 phase_retrieval_mockprocess.py -ar $rate -gr $iprt -icount $c -idur $dur -d $deadlinemsec -skipline $SKIPLINE -unipipedp &
#         IPR_PROCESS_PID=$!
#         echo $IPR_PROCESS_PID

#         # True for per iter validation activation
#         bash unipipe_run.sh $c $dur $rate $deadlinemsec $SKIPLINE $iprt
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
#         mv ipr_generation_state.csv ${EXP_RESULT_DIR}/unipipe/unipipe_ipr_generation_state_${c}_${deadlinemsec}_${rate}_${iprt}.csv
#         mv transmission_state.csv ${EXP_RESULT_DIR}/unipipe/unipipe_transmission_state_${c}_${deadlinemsec}_${rate}_${iprt}.csv
#         mv tmp.csv ${EXP_RESULT_DIR}/unipipe/unipipe_${c}_${deadlinemsec}_${rate}_${iprt}.csv
#         mv tmp.log ${EXP_RESULT_DIR}/unipipe/unipipe_${c}_${deadlinemsec}_${rate}_${iprt}.log
#         mv tmp_sysstat.csv ${EXP_RESULT_DIR}/unipipe/unipipe_sysstat_${c}_${deadlinemsec}_${rate}_${iprt}.csv
#         mv /dev/shm/traindatalist_unipipe_${c}_${dur}_${rate}_${iprt}.csv ${EXP_RESULT_DIR}/unipipe/traindatalist_unipipe_${c}_${deadlinemsec}_${dur}_${rate}_${iprt}.csv
#         mv /dev/shm/inferdatalist_unipipe_${c}_${dur}_${rate}_${iprt}.csv ${EXP_RESULT_DIR}/unipipe/inferdatalist_unipipe_${c}_${deadlinemsec}_${dur}_${rate}_${iprt}.csv
#         mv stitched_construction.png ${EXP_RESULT_DIR}/unipipe/stitched_construction_${c}_${deadlinemsec}_${rate}_${iprt}.png
#         mv nn_generated_data_ph_error.png ${EXP_RESULT_DIR}/unipipe/nn_generated_data_ph_error_${c}_${deadlinemsec}_${rate}_${iprt}.png
#         mv nn_generated_data_amp_error.png ${EXP_RESULT_DIR}/unipipe/nn_generated_data_amp_error_${c}_${deadlinemsec}_${rate}_${iprt}.png
#     done
# done


mkdir -p $EXP_RESULT_DIR/multicontext
for c in $INTERVAL_COUNT;do
    iprt=$IPR_THROUGHPUT_TUNED
    deadlinemsec=$DEADLINEMSEC

    for rate in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000;do
        for repeat in $(seq 1 $EXPERIMENT_REPEAT_COUNT); do
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

            bash multicontext_run.sh $c $dur $rate $deadlinemsec $SKIPLINE $iprt
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
            cat ipr_generation_state.csv >> ${EXP_RESULT_DIR}/multicontext/multicontext_ipr_generation_state_${c}_${deadlinemsec}_${rate}_${iprt}.csv
            cat transmission_state.csv >> ${EXP_RESULT_DIR}/multicontext/multicontext_transmission_state_${c}_${deadlinemsec}_${rate}_${iprt}.csv
            cat tmp.csv >> ${EXP_RESULT_DIR}/multicontext/multicontext_${c}_${deadlinemsec}_${rate}_${iprt}.csv
            mv tmp_infer.log ${EXP_RESULT_DIR}/multicontext/multicontext_${c}_${deadlinemsec}_${rate}_${iprt}_infer.log
            mv tmp_train.log ${EXP_RESULT_DIR}/multicontext/multicontext_${c}_${deadlinemsec}_${rate}_${iprt}_train.log
            mv tmp_sysstat.csv ${EXP_RESULT_DIR}/multicontext/multicontext_sysstat_${c}_${deadlinemsec}_${rate}_${iprt}.csv
            mv /dev/shm/traindatalist_multicontext_${c}_${dur}_${rate}_${iprt}.csv ${EXP_RESULT_DIR}/multicontext/traindatalist_multicontext_${c}_${deadlinemsec}_${dur}_${rate}_${iprt}.csv
            mv /dev/shm/inferdatalist_multicontext_${c}_${dur}_${rate}_${iprt}.csv ${EXP_RESULT_DIR}/multicontext/inferdatalist_multicontext_${c}_${deadlinemsec}_${dur}_${rate}_${iprt}.csv
            mv stitched_construction.png ${EXP_RESULT_DIR}/multicontext/stitched_construction_${c}_${deadlinemsec}_${rate}_${iprt}.png
            mv nn_generated_data_ph_error.png ${EXP_RESULT_DIR}/multicontext/nn_generated_data_ph_error_${c}_${deadlinemsec}_${rate}_${iprt}.png
            mv nn_generated_data_amp_error.png ${EXP_RESULT_DIR}/multicontext/nn_generated_data_amp_error_${c}_${deadlinemsec}_${rate}_${iprt}.png
        done
    done
done

popd

set +x
