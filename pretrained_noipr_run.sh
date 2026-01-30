#!/bin/bash

set -x

# will first switch to this directory

ROOTDIR=.
GTDEFAULT= #--gtdefault
CKPTTEST=--allckpttest
CONSTANTTRAINBS= #--constant-trainbs
pushd $ROOTDIR

c=$1
dur=$2
rate=$3
deadlinemsec=$4
skipline=$5
iprt=$6

if [ $# -ge 7 -a $7 != "None" ];then
    MODEL_TYPE="--model-type "$7
    echo $MODEL_TYPE
fi

if [ $# -ge 8 -a $8 != "None" ];then
    DATASET_TYPE="--large-dataset"
    echo $DATASET_TYPE
fi

if [ $# -ge 9 -a $9 != "None" ];then
    MAXINFERBS="--maxinfer-bs "$9
    echo $MAXINFERBS
fi

echo "Deleting data from /dev/shm"
set +x
rm /dev/shm/unipipe_exp*
rm model_multicontext/inctrained_interaval*
rm /dev/shm/inctrained*
rm /dev/shm/*_fastlog.log
set -x

# deadlinemsec=$((1000*$accumulationallow/$rate))

EXP_SCRIPT=pretrained_noipr.py

python3 $EXP_SCRIPT $GTDEFAULT --datarate ${rate} --deadline ${deadlinemsec} -idur ${dur} -icount ${c} -skipline $skipline -iprt ${iprt} -csvlog tmp.csv $MODEL_TYPE $DATASET_TYPE $MAXINFERBS&
PRETRAIN_PID=$!

python3 monitor.py 0 tmp_sysstat.csv &
MONITOR_PROCESS_PID=$!

set +x
while ps -p ${PRETRAIN_PID} > /dev/null
do
    # echo "training alive"
    sleep 5
done
set -x

kill -2 ${MONITOR_PROCESS_PID}
kill -2 ${MONITOR_PROCESS_PID}
set +x 
while ps -p ${MONITOR_PROCESS_PID} > /dev/null
do
    # echo "inference alive"
    sleep 1
done
set -x

set +x