#!/bin/bash

set -x

# will first switch to this directory

ROOTDIR=.
GTDEFAULT= #--gtdefault
CKPTTEST=--allckpttest
CONSTANTTRAINBS= #--constant-trainbs
pushd $ROOTDIR

EXP_RESULT_DIR=result_logs/bulk
mkdir -p $EXP_RESULT_DIR

mkdir -p $EXP_RESULT_DIR/pretrained
for c in 5;do
    dur=20
    while [ $dur -le 40 ];
    do
        rate=30
        while [ $rate -le 60 ];
        do
            for accumulationallow in 5 10;
            do
                deadlinemsec=$((1000*$accumulationallow/$rate))

                bash pretrained_run.sh $c $dur $rate $accumulationallow
                # move generated files for later analysis
                mv tmp.csv ${EXP_RESULT_DIR}/pretrained/pretrained_${c}_${dur}_${rate}_${accumulationallow}.csv
                mv tmp.log ${EXP_RESULT_DIR}/pretrained/pretrained_${c}_${dur}_${rate}_${accumulationallow}.log
                mv tmp_sysstat.csv ${EXP_RESULT_DIR}/pretrained/pretrained_sysstat_${c}_${dur}_${rate}_${accumulationallow}.csv 
            done
            rate=$(($rate*2))
        done
        dur=$(($dur+20))
    done
done

mkdir -p $EXP_RESULT_DIR/unipipe
for c in 5;do
    dur=20
    while [ $dur -le 40 ];
    do
        rate=30
        while [ $rate -le 60 ];
        do
            for accumulationallow in 5 10;
            do    
                bash unipipe_run.sh $c $dur $rate $accumulationallow

                # move generated files for later analysis
                mv tmp.csv ${EXP_RESULT_DIR}/unipipe/unipipe_${c}_${dur}_${rate}_${accumulationallow}.csv
                mv tmp.log ${EXP_RESULT_DIR}/unipipe/unipipe_${c}_${dur}_${rate}_${accumulationallow}.log
                mv tmp_sysstat.csv ${EXP_RESULT_DIR}/unipipe/unipipe_sysstat_${c}_${dur}_${rate}_${accumulationallow}.csv
            done
            rate=$(($rate*2))
        done
        dur=$(($dur+20))
    done
done

mkdir -p $EXP_RESULT_DIR/worst_case
for c in 5;do
    dur=20
    while [ $dur -le 40 ];
    do
        rate=30
        while [ $rate -le 60 ];
        do
            for accumulationallow in 5 10;
            do
                deadlinemsec=$((1000*$accumulationallow/$rate))
                bash worst_case_run.sh $c $dur $rate $accumulationallow
                # move generated files for later analysis
                mv tmp.csv ${EXP_RESULT_DIR}/worst_case/worst_case_${c}_${dur}_${rate}_${accumulationallow}.csv
                mv tmp.log ${EXP_RESULT_DIR}/worst_case/worst_case_${c}_${dur}_${rate}_${accumulationallow}.log
                mv tmp_sysstat.csv ${EXP_RESULT_DIR}/worst_case/worst_case_sysstat_${c}_${dur}_${rate}_${accumulationallow}.csv
            done
            rate=$(($rate*2))
        done
        dur=$(($dur+20))
    done
done

mkdir -p $EXP_RESULT_DIR/multicontext
for c in 5;do
    dur=20
    while [ $dur -le 40 ];
    do
        rate=30
        while [ $rate -le 60 ];
        do
            for accumulationallow in 5 10;
            do
                deadlinemsec=$((1000*$accumulationallow/$rate))
                
                # for greedy selection
                bash multicontext_run.sh $c $dur $rate $accumulationallow
                # move generated files for later analysis
                mv tmp.csv ${EXP_RESULT_DIR}/multicontext/multicontext_${c}_${dur}_${rate}_${accumulationallow}.csv
                mv tmp_infer.log ${EXP_RESULT_DIR}/multicontext/multicontext_${c}_${dur}_${rate}_${accumulationallow}_infer.log
                mv tmp_train.log ${EXP_RESULT_DIR}/multicontext/multicontext_${c}_${dur}_${rate}_${accumulationallow}_train.log
                mv tmp_sysstat.csv ${EXP_RESULT_DIR}/multicontext/multicontext_sysstat_${c}_${dur}_${rate}_${accumulationallow}.csv
            done
            
            rate=$(($rate*2))
        done
        dur=$(($dur+20))
    done
done
popd

set +x
