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

rm /dev/shm/unipipe_exp*
rm model_multicontext/inctrained_interaval*
rm /dev/shm/inctrained*
rm /dev/shm/*_fastlog.log

# deadlinemsec=$((1000*$accumulationallow/$rate))

EXP_SCRIPT=pretrained.py

python3 $EXP_SCRIPT $GTDEFAULT --datarate ${rate} --deadline ${deadlinemsec} -idur ${dur} -icount ${c} -csvlog tmp.csv &
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