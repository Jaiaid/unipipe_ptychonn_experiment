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

if [ $# -ge 6 -a $6 != "None" ];then
    iprrate=$6
    IPRRATE="-iprt "${iprrate}
    echo $IPRRATE
fi

if [ $# -ge 7 -a $7 != "None" ];then
    TRAINBS="-trainbs "$7
    echo $TRAINBS
fi

if [ $# -ge 8 -a $8 != "None" ];then
    MODEL_TYPE="--model-type "$8
    echo $MODEL_TYPE
fi

if [ $# -ge 9 -a $9 != "None" ];then
    DATASET_TYPE="--large-dataset"
    echo $DATASET_TYPE
fi

if [ $# -ge 10 -a ${10} != "None" ];then
    MAXINFERBS="--maxinfer-bs "${10}
    echo $MAXINFERBS
fi

if [ $# -ge 11 -a ${11} != "None" ];then
    TRAINVALIDFLAG="--validation-training"
    echo $TRAINVALIDFLAG
fi


if [ $# -eq 12 -a ${12} != "None" ];then
    ratio=${12}
    RATIO="-iprfrac "${ratio}
    echo $RATIO
fi

if [ $# -eq 13 ];then
    inferbsfactor=${13}
    INFFACT="-inffac "${inferbsfactor}
    echo $INFFACT
fi


echo "Deleting data from /dev/shm"
set +x
rm /dev/shm/unipipe_exp*
rm model_multicontext/inctrained_interaval*
rm /dev/shm/inctrained*
rm /dev/shm/*_fastlog.log
set -x

# deadlinemsec=$((1000*$accumulationallow/$rate))

EXP_SCRIPT=unipipe.py

python3 $EXP_SCRIPT $GTDEFAULT $CKPTTEST $CONSTANTTRAINBS --datarate ${rate} -skipline $skipline $IPRRATE $DATASET_TYPE $MODEL_TYPE $RATIO $INFFRAC $TRAINVALIDFLAG $TRAINBS $MAXINFERBS -constbs --deadline ${deadlinemsec} -idur ${dur} -icount ${c} -csvlog tmp.csv &
UNIPIPE_PID=$!

python3 monitor.py 0 tmp_sysstat.csv &
MONITOR_PROCESS_PID=$!

set +x
while ps -p ${UNIPIPE_PID} > /dev/null
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