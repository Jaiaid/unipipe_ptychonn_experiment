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

from plot_parameters import SYSTEM_NAME_LIST, SYSTEM_NAME_TO_LEGEND_DICT, SYSTEM_NAME_TO_HATCH_DICT, CSV_FILENAME_FMT, FIGSIZE, AXLABEL_KW, YTICK_LABEL_KW, LEGEND_COLSPACING, LEGEND_PROP

IPR_RATE_LIST = [16]
DEADLINE_LIST = [80]

CUSTOM_SYSTEM_NAME_TO_LEGEND_DICT={
    "pretrained_noipr": "Pretrained", "pretrained": "Pretrained-\nComp.",
    "multicontext": "MultiContext",
    "unipipe": "Unipipe\n(static)", "unipipe_dp": "Unipipe"
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, help="path containing the result directory in approproate format")
    parser.add_argument("--output-file-basename", "-o", type=str, help="output name without extension")
    parser.add_argument("--datarate", "-r", type=int,  help="list of data rates to consider")
    parser.add_argument("--total", "-t", type=int, help="total number of data samples")
    args = parser.parse_args()

    total_consumed_results = {sys: [] for sys in SYSTEM_NAME_LIST}
    infer_delay_miss_count_results = {sys: [] for sys in SYSTEM_NAME_LIST}

    # Read values and compute means
    for sys in SYSTEM_NAME_LIST:
        # for log file multicontext is named differently
        if sys == "multicontext":
            log_path = os.path.join(args.dir, sys, CSV_FILENAME_FMT.format(sys, DEADLINE_LIST[0], args.datarate, IPR_RATE_LIST[0]).replace('.csv', '_infer.log'))
        else:
            log_path = os.path.join(args.dir, sys, CSV_FILENAME_FMT.format(sys, DEADLINE_LIST[0], args.datarate, IPR_RATE_LIST[0]).replace('.csv', '.log'))

        total_consumed = 0
        infer_delay_miss_count = 0
        start_timestamp = None
        end_timestamp = None
        with open(log_path, 'r') as f:
            gap_values = []
            for line in f.readlines():
                if sys == "unipipe_dp" or sys == "unipipe":
                    match = re.search(r'\[(\d+\.\d+)\] TOTAL CONSUMED,MISSED,(\d+),(\d+)', line)
                else:
                    match = re.search(r'\[(\d+\.\d+)\] TOTAL CONSUMED,(\d+)', line)
                if match:
                    total_consumed += int(match.group(2))
                    # print(f"Updated end timestamp: {end_timestamp}, total consumed: {total_consumed}")

                match = re.search(r'\[(\d+\.\d+)\] INFER DELAY MISS COUNT,(\d+)', line)
                if match:
                    infer_delay_miss_count = int(match.group(2))

            infer_delay_miss_count_results[sys] = infer_delay_miss_count
            total_consumed_results[sys] = total_consumed
    
    # Plotting
    num_systems = len(SYSTEM_NAME_LIST)
    bar_width = 0.5
    x = np.arange(num_systems)
    
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # plot bars and error bars
    sys_values1 = [(total_consumed_results[sys] - infer_delay_miss_count_results[sys])*100/args.total for sys in SYSTEM_NAME_LIST]
    sys_values2 = [infer_delay_miss_count_results[sys]*100/args.total for sys in SYSTEM_NAME_LIST]
        
    # print(sys, sys_values)
    ax.bar(x, sys_values1, width=bar_width, label="Served", hatch=SYSTEM_NAME_TO_HATCH_DICT[sys], edgecolor='black')
    ax.bar(x, sys_values2, bottom=sys_values1, width=bar_width, label="Delay Missed", hatch=SYSTEM_NAME_TO_HATCH_DICT[sys], edgecolor='black')
    
    ax.set_ylabel('Processed Requests (%)', **AXLABEL_KW)
    ax.set_xticks(x)
    ax.set_xticklabels([CUSTOM_SYSTEM_NAME_TO_LEGEND_DICT[sys] for sys in SYSTEM_NAME_LIST], fontsize=6.2, fontweight="bold")
    ax.set_ylim([0, 120])
    ax.set_yticks(np.arange(0, 120, 20), labels=np.arange(0, 120, 20), **YTICK_LABEL_KW)
    ax.legend(frameon=False, ncol=2, loc="upper center", prop=LEGEND_PROP)
    
    output_path = f"{args.output_file_basename}.png"
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    output_path = f"{args.output_file_basename}.pdf"
    fig.savefig(output_path, format="pdf", dpi=600, bbox_inches="tight")
