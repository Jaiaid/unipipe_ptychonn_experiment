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
accumulationallow=$4
bs=$5

if [ $# -eq 5 ];then
    BSARG="-bs "${bs}
    echo $BSARG
fi

rm /dev/shm/unipipe_exp*
rm model_multicontext/inctrained_interaval*
rm /dev/shm/inctrained*
rm /dev/shm/*_fastlog.log

deadlinemsec=$((1000*$accumulationallow/$rate))

python3 $TRAIN_SCRIPT -idur ${dur} -icount ${c} -csvlog tmp.csv &
TRAIN_PID=$!

python3 $INFER_SCRIPT $GTDEFAULT --datarate ${rate} --deadline ${deadlinemsec} -idur ${dur} -icount ${c} $BSARG -csvlog tmp.csv &
INFER_PID=$!

python3 monitor.py 0 tmp_sysstat.csv &
MONITOR_PROCESS_PID=$!

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

kill -2 ${MONITOR_PROCESS_PID}

echo "inference and training process are done"
while ps -p ${MONITOR_PROCESS_PID} > /dev/null
do
    # echo "inference alive"
    sleep 1
done

popd

set +x