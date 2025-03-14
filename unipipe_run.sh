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

if [ $# -ge 5 -a $5 != "None" ];then
    iprrate=$5
    IPRRATE="-iprt "${iprrate}
    echo $IPRRATE
fi

if [ $# -eq 6 -a $6 != "None" ];then
    ratio=$6
    RATIO="-iprfrac "${ratio}
    echo $RATIO
fi

if [ $# -eq 7 ];then
    inferbsfactor=$7
    INFFACT="-inffac "${inferbsfactor}
    echo $INFFACT
fi

rm /dev/shm/unipipe_exp*
rm model_multicontext/inctrained_interaval*
rm /dev/shm/inctrained*
rm /dev/shm/*_fastlog.log

# deadlinemsec=$((1000*$accumulationallow/$rate))

EXP_SCRIPT=unipipe.py

python3 $EXP_SCRIPT $GTDEFAULT $CKPTTEST $CONSTANTTRAINBS --datarate ${rate} $IPRRATE $RATIO $INFFRAC --deadline ${deadlinemsec} -idur ${dur} -icount ${c} -csvlog tmp.csv &
UNIPIPE_PID=$!

python3 monitor.py 0 tmp_sysstat.csv &
MONITOR_PROCESS_PID=$!

while ps -p ${UNIPIPE_PID} > /dev/null
do
    # echo "training alive"
    sleep 5
done

kill -2 ${MONITOR_PROCESS_PID}
kill -2 ${MONITOR_PROCESS_PID}
while ps -p ${MONITOR_PROCESS_PID} > /dev/null
do
    # echo "inference alive"
    sleep 1
done

set +x