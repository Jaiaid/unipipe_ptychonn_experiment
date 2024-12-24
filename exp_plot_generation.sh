#!/bin/bash

set -x

# will first switch to this directory

RESULTDIR=$1

ROOTDIR=.
pushd $ROOTDIR

python3 analysis_scripts/plot_mean_infer_acc_curve_vs_datarate.py --dir $RESULTDIR
python3 analysis_scripts/plot_mean_missrate_vs_datarate.py --dir $RESULTDIR
python3 analysis_scripts/plot_gpuutilization_summary.py --dir $RESULTDIR

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
                python3 analysis_scripts/plot_throughput_inference.py \
                --dir $RESULTDIR -dur $dur -rate $rate --accumallow $accumulationallow

                python3 analysis_scripts/plot_throughput_training.py \
                --dir $RESULTDIR -dur $dur -rate $rate --accumallow $accumulationallow

                python3 analysis_scripts/plot_inference_iteration_gaptime.py \
                --dir $RESULTDIR -dur $dur -rate $rate --accumallow $accumulationallow

                python3 analysis_scripts/plot_pherror_missrate_firstlastinterval.py \
                --dir $RESULTDIR -dur $dur -rate $rate --accumallow $accumulationallow

                python3 analysis_scripts/plot_gpuutilization_crossreffed.py \
                -rlog $RESULTDIR/pretrained/pretrained_${c}_${dur}_${rate}_${accumulationallow}.log \
                -statcsv $RESULTDIR/pretrained/pretrained_sysstat_${c}_${dur}_${rate}_${accumulationallow}.csv

                mv figure_gpuutil_curve_pretrained_${dur}_${rate}_${accumulationallow}.pdf $RESULTDIR/pretrained/
            done
            rate=$(($rate*2))
        done
        dur=$(($dur+20))
    done
done

for c in 5;do
    dur=20
    while [ $dur -le 40 ];
    do
        rate=30
        while [ $rate -le 60 ];
        do
            for accumulationallow in 5 10;
            do    
                python3 analysis_scripts/plot_gpuutilization_crossreffed.py \
                -rlog $RESULTDIR/unipipe/unipipe_${c}_${dur}_${rate}_${accumulationallow}.log \
                -statcsv $RESULTDIR/unipipe/unipipe_sysstat_${c}_${dur}_${rate}_${accumulationallow}.csv

                mv figure_gpuutil_curve_unipipe_${dur}_${rate}_${accumulationallow}.pdf $RESULTDIR/unipipe/
            done
            rate=$(($rate*2))
        done
        dur=$(($dur+20))
    done
done

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

                python3 analysis_scripts/plot_gpuutilization_crossreffed.py \
                -rlog $RESULTDIR/worst_case/worst_case_${c}_${dur}_${rate}_${accumulationallow}.log \
                -statcsv $RESULTDIR/worst_case/worst_case_sysstat_${c}_${dur}_${rate}_${accumulationallow}.csv

                mv figure_gpuutil_curve_worst_case_${dur}_${rate}_${accumulationallow}.pdf $RESULTDIR/worst_case/
            done
            rate=$(($rate*2))
        done
        dur=$(($dur+20))
    done
done

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
                
                python3 analysis_scripts/plot_gpuutilization_crossreffed.py \
                -rlog $RESULTDIR/multicontext/multicontext_${c}_${dur}_${rate}_${accumulationallow}_infer.log \
                -statcsv $RESULTDIR/multicontext/multicontext_sysstat_${c}_${dur}_${rate}_${accumulationallow}.csv

                mv figure_gpuutil_curve_multicontext_${dur}_${rate}_${accumulationallow}.pdf $RESULTDIR/multicontext/
            done
            
            rate=$(($rate*2))
        done
        dur=$(($dur+20))
    done
done
popd

set +x
