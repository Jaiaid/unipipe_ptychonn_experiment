"""
This script will follow similar structure to plot_missrate_barclusters_dratevariation.py
but will plot two things the stream wait time and batch size in each read.

How it will get the data?
It will read from same basename format but log file
Using regex it will extract the inference invocation gap values and plot them.
The line of interest in the log file looks like: [1765483658.649488] INFER READ LATENCY,3.4332275390625e-05,0

It will plot bar clusters for different systems across varying data rates.
Each bar will have error bars representing standard deviation if multiple reads are available.

Write the code for me copilot, please
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import argparse

from plot_parameters import SYSTEM_NAME_LIST, SYSTEM_NAME_TO_LEGEND_DICT, SYSTEM_NAME_TO_HATCH_DICT, CSV_FILENAME_FMT

IPR_RATE_LIST = [16]
DEADLINE_LIST = [80]
DATARATE_LIST = [1000, 2000, 3000, 4000, 5000]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, help="path containing the result directory in approproate format")
    parser.add_argument("--output-file-basename", "-o", type=str, help="output name without extension")
    args = parser.parse_args()

    # Dictionary to store average values: {param: [sys1_val, sys2_val, ...]}
    results_bs = {param: [] for param in DATARATE_LIST}
    fluctuations_bs = {param: [] for param in DATARATE_LIST}
    results_sw = {param: [] for param in DATARATE_LIST}
    fluctuations_sw = {param: [] for param in DATARATE_LIST}

    # Read values and compute means
    for param in DATARATE_LIST:
        for sys in SYSTEM_NAME_LIST:
            # for log file multicontext is named differently
            if sys == "multicontext":
                log_path = os.path.join(args.dir, sys, CSV_FILENAME_FMT.format(sys, DEADLINE_LIST[0], param, IPR_RATE_LIST[0]).replace('.csv', '_infer.log'))
            else:
                log_path = os.path.join(args.dir, sys, CSV_FILENAME_FMT.format(sys, DEADLINE_LIST[0], param, IPR_RATE_LIST[0]).replace('.csv', '.log'))

            with open(log_path, 'r') as f:
                gap_values = []
                batch_sizes = []
                for line in f.readlines():
                    match = re.search(r'INFER READ LATENCY,([\d.eE+-]+),(\d+)', line)
                    if match:
                        # to fix for mistakes which caused negative of actual values to be calculated
                        gap_value = abs(float(match.group(1)))
                        batch_size = int(match.group(2))
                        gap_values.append(gap_value)
                        batch_sizes.append(batch_size)

                if gap_values:
                    avg_gap = np.mean(gap_values)
                    fluctuation = (avg_gap-np.min(gap_values), np.max(gap_values)-avg_gap)
                    fluctuations_sw[param].append(fluctuation)
                    results_sw[param].append(avg_gap)
                else:
                    fluctuations_sw[param].append(0)
                    results_sw[param].append(0)  # or handle missing data appropriately

                if batch_sizes:
                    avg_bs = np.mean(batch_sizes)
                    fluctuation_bs = (avg_bs-np.min(batch_sizes), np.max(batch_sizes)-avg_bs)
                    fluctuations_bs[param].append(fluctuation_bs)
                    results_bs[param].append(avg_bs)
                else:
                    fluctuations_bs[param].append((0,0))
                    results_bs[param].append(0)  # or handle missing data appropriately

    # Plotting
    num_params = len(DATARATE_LIST)
    num_systems = len(SYSTEM_NAME_LIST)
    bar_width = 0.8 / num_systems
    x = np.arange(num_params)
    
    # inferread plot
    fig, ax = plt.subplots(figsize=(4, 2.25))
    
    for i, sys in enumerate(SYSTEM_NAME_LIST):
        sys_values = [results_sw[param][i] for param in DATARATE_LIST]
        sys_fluctuations = np.array([fluctuations_sw[param][i] for param in DATARATE_LIST]).T
        ax.bar(x + i * bar_width, sys_values, yerr=sys_fluctuations, width=bar_width, label=SYSTEM_NAME_TO_LEGEND_DICT[sys], hatch=SYSTEM_NAME_TO_HATCH_DICT[sys], edgecolor='black')
    
    ax.set_xlabel('Data Rate (req/s)')
    ax.set_ylabel('Avg. Inference Read Latency (s)')
    ax.set_xticks(x + bar_width * (num_systems - 1) / 2)
    ax.set_xticklabels([str(param) for param in DATARATE_LIST])
    ax.legend()
    
    output_path = f"{args.output_file_basename}.png"
    fig.savefig(output_path, dpi=600, bbox_inches="tight")

    # bs plot
    fig, ax = plt.subplots(figsize=(4, 2.25))
    
    for i, sys in enumerate(SYSTEM_NAME_LIST):
        sys_values = [results_bs[param][i] for param in DATARATE_LIST]
        sys_fluctuations = [fluctuations_bs[param][i] for param in DATARATE_LIST]
        sys_fluctuations = np.array([(min_val, max_val) for min_val, max_val in sys_fluctuations]).T
        ax.bar(x + i * bar_width, sys_values, yerr=sys_fluctuations, width=bar_width, label=SYSTEM_NAME_TO_LEGEND_DICT[sys], hatch=SYSTEM_NAME_TO_HATCH_DICT[sys], edgecolor='black')

    ax.set_xlabel('Data Rate (req/s)')
    ax.set_ylabel('Avg. Batch Size (requests)')
    ax.set_xticks(x + bar_width * (num_systems - 1) / 2)
    ax.set_xticklabels([str(param) for param in DATARATE_LIST])
    ax.legend()
    
    output_path = f"{args.output_file_basename}_bs.png"
    fig.savefig(output_path, dpi=600, bbox_inches="tight")