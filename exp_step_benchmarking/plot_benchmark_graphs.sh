#!/bin/bash

# # for single multicontext benchmark plot which shows inference latency fluctuation 
# python3 plot_single_multicontext_benchmark.py -istsv1 benchmark_resnet18_nn_step.csv -imtsv1 benchmark_coincidental_resnet18_infer_nn_step.csv -istsv2 benchmark_resnet50_nn_step.csv -imtsv2 benchmark_coincidental_resnet50_infer_nn_step.csv -bs 8 -o fig_single_multicontext_benchmark_bs8.pdf
# python3 plot_single_multicontext_benchmark.py -istsv1 benchmark_resnet18_nn_step.csv -imtsv1 benchmark_coincidental_resnet18_infer_nn_step.csv -istsv2 benchmark_resnet50_nn_step.csv -imtsv2 benchmark_coincidental_resnet50_infer_nn_step.csv -bs 32 -o fig_single_multicontext_benchmark_bs32.pdf

python3 plot_single_multicontext_benchmark_ptychonn.py -istsv1 benchmark_ptychonn_nn_step.csv -imtsv1 benchmark_coincidental_ptychonn_infer_nn_step.csv -o fig_single_multicontext_benchmark_ptychonn.pdf

# # for forward pass and backward pass linearity demonstration
# python3 plot_forward_backward_pass_benchmark.py -idir .

# # for continual training improvement and data drift demonstration
# python3 exp_data_drift_observation_oracle_pretrained_intervaltrained.py

