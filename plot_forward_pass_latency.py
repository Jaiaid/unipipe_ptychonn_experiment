"""
This script will follow similar structure to plot_missrate_barclusters_dratevariation.py
but will plot the forward pass latency.

How it will get the data?
It will read from same basename format but log file
Using regex it will extract the inference invocation gap values and plot them.
The line of interest in the log file looks like: [1765484226.5689094] FORWARD PASS TOOK(sec.),0.0014719963073730469

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
DATARATE_LIST = [1000, 2000, 3000]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, help="path containing the result directory in approproate format")
    parser.add_argument("--output-file-basename", "-o", type=str, help="output name without extension")
    args = parser.parse_args()

    # Dictionary to store average values: {param: [sys1_val, sys2_val, ...]}
    results = {param: [] for param in DATARATE_LIST}
    fluctuations = {param: [] for param in DATARATE_LIST}

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
                for line in f.readlines():
                    match = re.search(r'FORWARD PASS TOOK\(sec.\),([\d.eE+-]+)', line)
                    if match:
                        gap_value = float(match.group(1))
                        gap_values.append(gap_value)

                if gap_values:
                    avg_gap = np.mean(gap_values)
                    fluctuation = np.std(gap_values)
                    fluctuations[param].append(fluctuation)
                    results[param].append(avg_gap)
                else:
                    fluctuations[param].append(0)
                    results[param].append(0)  # or handle missing data appropriately

    # Plotting
    num_params = len(DATARATE_LIST)
    num_systems = len(SYSTEM_NAME_LIST)
    bar_width = 0.8 / num_systems
    x = np.arange(num_params)
    
    fig, ax = plt.subplots(figsize=(4, 2.25))
    
    # plot bars and error bars
    for i, sys in enumerate(SYSTEM_NAME_LIST):
        sys_values = [results[param][i] for param in DATARATE_LIST]
        sys_fluctuations = [fluctuations[param][i] for param in DATARATE_LIST]
        ax.bar(x + i * bar_width, sys_values, yerr=sys_fluctuations, width=bar_width, label=SYSTEM_NAME_TO_LEGEND_DICT[sys], hatch=SYSTEM_NAME_TO_HATCH_DICT[sys], edgecolor='black')
    
    ax.set_xlabel('Data Rate (req/s)')
    ax.set_ylabel('Avg. Forward Pass Latency (s)')
    ax.set_xticks(x + bar_width * (num_systems - 1) / 2)
    ax.set_xticklabels([str(param) for param in DATARATE_LIST])
    ax.legend()
    
    output_path = f"{args.output_file_basename}.png"
    fig.savefig(output_path, dpi=600, bbox_inches="tight")

