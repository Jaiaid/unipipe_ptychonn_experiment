#!/bin/bash

set -x

# will first switch to this directory

ROOTDIR=.
GTDEFAULT= #--gtdefault
CKPTTEST= #--allckpttest
CONSTANTTRAINBS= #--constant-trainbs
pushd $ROOTDIR
mkdir -p result_logs/bulk


# EXP_SCRIPT=pretrained.py
# mkdir -p result_logs/bulk/pretrained
# for c in 5;do
#     dur=20
#     while [ $dur -le 40 ];
#     do
#         rate=30
#         while [ $rate -le 60 ];
#         do
#             for accumulationallow in {5..10};
#             do
#                 rm /dev/shm/unipipe_exp*
#                 rm model_unipipe/*

#                 deadlinemsec=$((1000*$accumulationallow/$rate))
#                 python3 $EXP_SCRIPT $GTDEFAULT --datarate ${rate} --deadline ${deadlinemsec} -idur ${dur} -icount ${c} -csvlog result_logs/bulk/pretrained/pretrained_${c}_${dur}_${rate}_${accumulationallow}.csv
#             done
#             rate=$(($rate*2))
#         done
#         dur=$(($dur+20))
#     done
# done

EXP_SCRIPT=unipipe.py
mkdir -p result_logs/bulk/unipipe
for c in 5;do
    dur=20
    while [ $dur -le 40 ];
    do
        rate=30
        while [ $rate -le 60 ];
        do
            for accumulationallow in {5..10};
            do
                rm /dev/shm/unipipe_exp*
                rm model_unipipe/*

                deadlinemsec=$((1000*$accumulationallow/$rate))
                python3 $EXP_SCRIPT $GTDEFAULT $CKPTTEST $CONSTANTTRAINBS --datarate ${rate} --deadline ${deadlinemsec} -idur ${dur} -icount ${c} -csvlog result_logs/bulk/unipipe/unipipe_${c}_${dur}_${rate}_${accumulationallow}.csv
            done
            rate=$(($rate*2))
        done
        dur=$(($dur+20))
    done
done

# EXP_SCRIPT=worst_case.py
# mkdir -p result_logs/bulk/worst_case
# for c in 5;do
#     dur=20
#     while [ $dur -le 40 ];
#     do
#         rate=30
#         while [ $rate -le 60 ];
#         do
#             for accumulationallow in {5..10};
#             do
#                 rm /dev/shm/unipipe_exp*
#                 rm model_worst_case/*

#                 deadlinemsec=$((1000*$accumulationallow/$rate))
#                 python3 $EXP_SCRIPT $GTDEFAULT --datarate ${rate} --deadline ${deadlinemsec} -idur ${dur} -icount ${c} -csvlog result_logs/bulk/worst_case/worst_case_${c}_${dur}_${rate}_${accumulationallow}.csv
#             done
#             rate=$(($rate*2))
#         done
#         dur=$(($dur+20))
#     done
# done

# TRAIN_SCRIPT="multicontext_train.py"
# INFER_SCRIPT="multicontext_infer.py"
# mkdir -p result_logs/bulk/multicontext
# for c in 5;do
#     dur=20
#     while [ $dur -le 40 ];
#     do
#         rate=30
#         while [ $rate -le 60 ];
#         do
#             for accumulationallow in {5..10};
#             do
#                 rm /dev/shm/unipipe_exp*
#                 rm model_multicontext/inctrained_interaval*
                
#                 deadlinemsec=$((1000*$accumulationallow/$rate))
                
#                 python3 $TRAIN_SCRIPT -idur ${dur} -icount ${c} -csvlog result_logs/bulk/multicontext/multicontext_${c}_${dur}_${rate}_${accumulationallow}.csv &
#                 TRAIN_PID=$!

#                 python3 $INFER_SCRIPT $GTDEFAULT --datarate ${rate} --deadline ${deadlinemsec} -idur ${dur} -icount ${c} -csvlog result_logs/bulk/multicontext/multicontext_${c}_${dur}_${rate}_${accumulationallow}.csv &
#                 INFER_PID=$!

#                 while ps -p ${TRAIN_PID} > /dev/null
#                 do
#                     # echo "training alive"
#                     sleep 5
#                 done
#                 while ps -p ${INFER_PID} > /dev/null
#                 do
#                     # echo "inference alive"
#                     sleep 5
#                 done

#                 echo "inference and training process are done"
#             done
            
#             rate=$(($rate*2))
#         done
#         dur=$(($dur+20))
#     done
# done
# popd

set +x
