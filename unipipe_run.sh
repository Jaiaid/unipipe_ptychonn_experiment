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
    TRAINVALIDFLAG="--validation-training"
    echo $TRAINVALIDFLAG
fi

if [ $# -eq 8 -a $8 != "None" ];then
    ratio=$8
    RATIO="-iprfrac "${ratio}
    echo $RATIO
fi

if [ $# -eq 9 ];then
    inferbsfactor=$9
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

python3 $EXP_SCRIPT $GTDEFAULT $CKPTTEST $CONSTANTTRAINBS --datarate ${rate} -skipline $skipline $IPRRATE $RATIO $INFFRAC $TRAINVALIDFLAG --deadline ${deadlinemsec} -idur ${dur} -icount ${c} -csvlog tmp.csv &
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