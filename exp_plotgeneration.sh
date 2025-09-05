#!/bin/bash


# # drate variation experiment
# python3 plot_error_barclusters.py --dir result_logs/smalldataset/ -o fig_error_smalldataset
# python3 plot_error_barclusters.py --dir result_logs/largedataset/ -o fig_error_largedataset
# python3 plot_nnerror_barclusters.py --dir result_logs/largedataset/ -o fig_nnerror_largedataset
# python3 plot_nnerror_barclusters.py --dir result_logs/smalldataset/ -o fig_nnerror_smalldataset
python3 plot_missrate_barclusters.py --dir result_logs/largedataset/ -o fig_missrate_largedataset
python3 plot_missrate_barclusters.py --dir result_logs/smalldataset/ -o fig_missrate_smalldataset


# # deadline variation experiment
# python3 plot_error_barclusters_deadlinevariation.py --dir result_logs/smalldataset_deadlinevariation/ -o fig_error_smalldataset_deadlinevariation
# python3 plot_error_barclusters_deadlinevariation.py --dir result_logs/largedataset_deadlinevariation/ -o fig_error_largedataset_deadlinevariation
# python3 plot_nnerror_barclusters_deadlinevariation.py --dir result_logs/largedataset_deadlinevariation/ -o fig_nnerror_largedataset_deadlinevariation
# python3 plot_nnerror_barclusters_deadlinevariation.py --dir result_logs/smalldataset_deadlinevariation/ -o fig_nnerror_smalldataset_deadlinevariation
# python3 plot_missrate_barclusters_deadlinevariation.py --dir result_logs/largedataset_deadlinevariation/ -o fig_missrate_largedataset_deadlinevariation
# python3 plot_missrate_barclusters_deadlinevariation.py --dir result_logs/smalldataset_deadlinevariation/ -o fig_missrate_smalldataset_deadlinevariation


# # infer bs consumption comparison
# # large dataset
# python3 plot_inferbs_consumption.py -log1 result_logs/largedataset/multicontext/multicontext_5_80_1000_16_infer.log -log2 result_logs/largedataset/unipipe/unipipe_5_80_1000_16.log -outfbname fig_inferbs_large_5_80_1000_16
# python3 plot_inferbs_consumption.py -log1 result_logs/largedataset/multicontext/multicontext_5_80_2000_16_infer.log -log2 result_logs/largedataset/unipipe/unipipe_5_80_2000_16.log -outfbname fig_inferbs_large_5_80_2000_16
# python3 plot_inferbs_consumption.py -log1 result_logs/largedataset/multicontext/multicontext_5_80_3000_16_infer.log -log2 result_logs/largedataset/unipipe/unipipe_5_80_3000_16.log -outfbname fig_inferbs_large_5_80_3000_16
# # small dataset
# python3 plot_inferbs_consumption.py -log1 result_logs/smalldataset/multicontext/multicontext_5_80_1000_16_infer.log -log2 result_logs/smalldataset/unipipe/unipipe_5_80_1000_16.log -outfbname fig_inferbs_small_5_80_1000_16
# python3 plot_inferbs_consumption.py -log1 result_logs/smalldataset/multicontext/multicontext_5_80_2000_16_infer.log -log2 result_logs/smalldataset/unipipe/unipipe_5_80_2000_16.log -outfbname fig_inferbs_small_5_80_2000_16
# python3 plot_inferbs_consumption.py -log1 result_logs/smalldataset/multicontext/multicontext_5_80_3000_16_infer.log -log2 result_logs/smalldataset/unipipe/unipipe_5_80_3000_16.log -outfbname fig_inferbs_small_5_80_3000_16
