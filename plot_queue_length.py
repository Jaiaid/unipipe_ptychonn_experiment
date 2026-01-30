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
    results_qlen = {param: [] for param in DATARATE_LIST}
    fluctuations_qlen = {param: [] for param in DATARATE_LIST}

    # Read values and compute means
    for param in DATARATE_LIST:
        for sys in SYSTEM_NAME_LIST:
            # for log file multicontext is named differently
            if sys == "multicontext":
                log_path = os.path.join(args.dir, sys, CSV_FILENAME_FMT.format(sys, DEADLINE_LIST[0], param, IPR_RATE_LIST[0]).replace('.csv', '_infer.log'))
            else:
                log_path = os.path.join(args.dir, sys, CSV_FILENAME_FMT.format(sys, DEADLINE_LIST[0], param, IPR_RATE_LIST[0]).replace('.csv', '.log'))

            with open(log_path, 'r') as f:
                queue_sizes = []
                for line in f.readlines():
                    match = re.search(r'QUEUE BACKLOG LENGTH,(\d+)', line)
                    if match:
                        # to fix for mistakes which caused negative of actual values to be calculated
                        queue_size = int(match.group(1))
                        queue_sizes.append(queue_size)
                if queue_sizes:
                    avg_qlen = np.mean(queue_sizes)
                    fluctuation_qlen = np.std(queue_sizes)
                    fluctuations_qlen[param].append((avg_qlen-np.min(queue_sizes), np.max(queue_sizes)-avg_qlen))
                    results_qlen[param].append(avg_qlen)
                else:
                    fluctuations_qlen[param].append((0, 0))
                    results_qlen[param].append(0)  # or handle missing data appropriately
    # Plotting
    num_params = len(DATARATE_LIST)
    num_systems = len(SYSTEM_NAME_LIST)
    bar_width = 0.8 / num_systems
    x = np.arange(num_params)
    
    # sw plot
    fig, ax = plt.subplots(figsize=(4, 2.25))
    
    for i, sys in enumerate(SYSTEM_NAME_LIST):
        sys_values = [results_qlen[param][i] for param in DATARATE_LIST]
        sys_fluctuations = [fluctuations_qlen[param][i] for param in DATARATE_LIST]
        sys_error = np.array([[low, high] for low, high in sys_fluctuations]).T
        # print(len(sys_values), len(sys_error), sys)
        ax.bar(x + i * bar_width, sys_values, yerr=sys_error, width=bar_width, label=SYSTEM_NAME_TO_LEGEND_DICT[sys], hatch=SYSTEM_NAME_TO_HATCH_DICT[sys], edgecolor='black')
        # ax.scatter(x + i * bar_width, sys_values, label=SYSTEM_NAME_TO_LEGEND_DICT[sys], zorder=5)
    
    ax.set_xlabel('Data Rate (req/s)')
    ax.set_ylabel('Avg. Queue Length (requests)')
    ax.set_xticks(x + bar_width * (num_systems - 1) / 2)
    ax.set_xticklabels([str(param) for param in DATARATE_LIST])
    ax.legend()
    
    output_path = f"{args.output_file_basename}_sw.png"
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
