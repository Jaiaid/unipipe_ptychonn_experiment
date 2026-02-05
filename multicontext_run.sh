#!/bin/bash

set -x

# will first switch to this directory

ROOTDIR=.
GTDEFAULT= #--gtdefault
CKPTTEST=--allckpttest
CONSTANTTRAINBS= #--constant-trainbs
pushd $ROOTDIR

TRAIN_SCRIPT="multicontext_train.py"
INFER_SCRIPT="multicontext_infer.py"

c=$1
dur=$2
rate=$3
deadlinemsec=$4
skipline=$5

if [ $# -ge 6 -a $6 != "None" ];then
    iprrate=$6
    IPRRATE="-iprt "${iprrate}
    echo $IPRRATE
fi

if [ $# -ge 7 -a $7 != "None" ];then
    MODEL_TYPE="--model-type "$7
    echo $MODEL_TYPE
fi

if [ $# -ge 8 -a $8 != "None" ];then
    DATASET_TYPE="--large-dataset"
    echo $DATASET_TYPE
fi

if [ $# -eq 9 ];then
    BSARG="-bs "${9}
    echo $BSARG
fi

if [ $# -ge 10 -a ${10} != "None" ];then
    MAXINFERBS="--maxinfer-bs "${10}
    echo $MAXINFERBS
fi

if [ $# -ge 11 -a ${11} != "None" ];then
    GTCOUNT="-gtcount "${11}
    echo $GTCOUNT
fi

echo "Deleting data from /dev/shm"
set +x
rm /dev/shm/unipipe_exp*
rm model_multicontext/inctrained_interaval*
rm /dev/shm/inctrained*
rm /dev/shm/*_fastlog.log
set -x

# deadlinemsec=$((1000*$accumulationallow/$rate))

python3 $TRAIN_SCRIPT --datarate ${rate} $IPRRATE --deadline ${deadlinemsec} -idur ${dur} -icount ${c} --csvlog tmp.csv $MODEL_TYPE $DATASET_TYPE $GTCOUNT &
TRAIN_PID=$!

# it needs dataset type information as it will 
python3 $INFER_SCRIPT $GTDEFAULT --datarate ${rate} $IPRRATE --deadline ${deadlinemsec} -skipline $skipline -idur ${dur} -icount ${c} $DATASET_TYPE $BSARG -csvlog tmp.csv $MODEL_TYPE $MAXINFERBS $GTCOUNT &
INFER_PID=$!

python3 monitor.py 0 tmp_sysstat.csv &
MONITOR_PROCESS_PID=$!

set +x
while ps -p ${TRAIN_PID} > /dev/null
do
    # echo "training alive"
    sleep 5
done
while ps -p ${INFER_PID} > /dev/null
do
    # echo "inference alive"
    sleep 5
done
set -x

kill -2 ${MONITOR_PROCESS_PID}

echo "inference and training process are done"
set +x
while ps -p ${MONITOR_PROCESS_PID} > /dev/null
do
    # echo "inference alive"
    sleep 1
done
set -x

popd

set +x