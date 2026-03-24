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
DATARATE_LIST = [1000, 2000, 3000, 4000, 5000]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, help="path containing the result directory in approproate format")
    parser.add_argument("--output-file-basename", "-o", type=str, help="output name without extension")
    args = parser.parse_args()

    # Dictionary to store average values: {param: [sys1_val, sys2_val, ...]}
    results = {param: [] for param in DATARATE_LIST}
    fluctuations = {param: [] for param in DATARATE_LIST}

    total_consumed_results = {param: [] for param in DATARATE_LIST}
    start_timestamp = None
    end_timestamp = None

    infer_delay_miss_count_results = {param: [] for param in DATARATE_LIST}

    # infer bs detection state
    infer_bs_regex_match_state = True

    # Read values and compute means
    for param in DATARATE_LIST:
        for sys in SYSTEM_NAME_LIST:
            # for log file multicontext is named differently
            if sys == "multicontext":
                log_path = os.path.join(args.dir, sys, CSV_FILENAME_FMT.format(sys, DEADLINE_LIST[0], param, IPR_RATE_LIST[0]).replace('.csv', '_infer.log'))
            else:
                log_path = os.path.join(args.dir, sys, CSV_FILENAME_FMT.format(sys, DEADLINE_LIST[0], param, IPR_RATE_LIST[0]).replace('.csv', '.log'))

            total_consumed = 0
            infer_delay_miss_count = 0
            start_timestamp = None
            end_timestamp = None
            with open(log_path, 'r') as f:
                gap_values = []
                for line in f.readlines():
                    if infer_bs_regex_match_state:
                        match = re.search(r'INFER READ LATENCY,([\d.eE+-]+),(\d+)', line)
                        if match:
                            batch_size = int(match.group(2))
                            infer_bs_regex_match_state = False

                    match = re.search(r'ITERATION TAKES\(sec.\),([\d.eE+-]+)', line)
                    if match:
                        gap_value = float(match.group(1))
                        gap_values.append(batch_size / gap_value)
                        infer_bs_regex_match_state = True

                    match = re.search(r'ITERATION TAKES\(sec.\),([\d.eE+-]+)', line)
                    if match:
                        gap_value = float(match.group(1))
                        gap_values.append(batch_size / gap_value)
                        infer_bs_regex_match_state = True

                    if start_timestamp is None:
                        if sys == "unipipe_dp":
                            match = re.search(r'\[(\d+\.\d+)\] UNIPIPE DP ITERATION START,0', line)
                        elif sys == "pretrained":
                            match = re.search(r'\[(\d+\.\d+)\] PRETRAINED CONSUMPTION START,(\d+\.\d+)', line)
                        elif sys == "pretrained_noipr":
                            match = re.search(r'\[(\d+\.\d+)\] PRETRAINED NOIPR CONSUMPTION START,(\d+\.\d+)', line)
                        else:
                            match = re.search(r'\[(\d+\.\d+)\] IPR ITERATION START,0', line)
                        
                        if match:
                            start_timestamp = float(match.group(1))
                            # print(f"Detected start timestamp: {start_timestamp}")

                    if sys == "unipipe_dp" or sys == "unipipe":
                        match = re.search(r'\[(\d+\.\d+)\] TOTAL CONSUMED,MISSED,(\d+),(\d+)', line)
                    else:
                        match = re.search(r'\[(\d+\.\d+)\] TOTAL CONSUMED,(\d+)', line)
                    if match:
                        end_timestamp = float(match.group(1))
                        total_consumed += int(match.group(2))
                        # print(f"Updated end timestamp: {end_timestamp}, total consumed: {total_consumed}")

                    match = re.search(r'\[(\d+\.\d+)\] INFER DELAY MISS COUNT,(\d+)', line)
                    if match:
                        infer_delay_miss_count = int(match.group(2))

                if gap_values:
                    avg_gap = np.mean(gap_values)
                    fluctuation = np.std(gap_values)
                    fluctuations[param].append((avg_gap-np.min(gap_values), np.max(gap_values)-avg_gap))
                    results[param].append(avg_gap)
                else:
                    fluctuations[param].append((0, 0))
                    results[param].append(0)  # or handle missing data appropriately

                infer_delay_miss_count_results[param].append((total_consumed-infer_delay_miss_count)/(end_timestamp-start_timestamp))
                total_consumed_results[param].append(total_consumed/(end_timestamp-start_timestamp))
                # print(sys, param, end_timestamp-start_timestamp, total_consumed, total_consumed/(end_timestamp-start_timestamp))
    
    # Plotting
    num_params = len(DATARATE_LIST)
    num_systems = len(SYSTEM_NAME_LIST)
    bar_width = 0.8 / num_systems
    x = np.arange(num_params)
    
    # iter level throughput
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    # plot bars and error bars
    for i, sys in enumerate(SYSTEM_NAME_LIST):
        sys_values = [results[param][i] for param in DATARATE_LIST]
        sys_fluctuations = [fluctuations[param][i] for param in DATARATE_LIST]
        sys_error = np.array([[low, high] for low, high in sys_fluctuations]).T
        ax.bar(x + i * bar_width, sys_values, yerr=sys_error, width=bar_width, label=SYSTEM_NAME_TO_LEGEND_DICT[sys], hatch=SYSTEM_NAME_TO_HATCH_DICT[sys], edgecolor='black')
    
    ax.set_xlabel('Data Rate (req./sec.)', AXLABEL_KW)
    ax.set_ylabel('Avg. Infer. Thpt. (req./sec.)', AXLABEL_KW)

    ax.set_xticks(x + bar_width * (num_systems - 1) / 2)
    ax.set_xticklabels([str(param) for param in DATARATE_LIST], **YTICK_LABEL_KW)

    ax.set_ylim([500, 5000])
    ax.set_yticks(np.arange(0, 6000, 1000))
    ax.set_yticklabels(np.arange(0, 6000, 1000), **YTICK_LABEL_KW)

    ax.legend(frameon=False, columnspacing=LEGEND_COLSPACING, prop=LEGEND_PROP, loc="upper center", ncol=2)
    
    output_path = f"{args.output_file_basename}_iterlevel.png"
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    output_path = f"{args.output_file_basename}_iterlevel.pdf"
    fig.savefig(output_path, format="pdf", dpi=600, bbox_inches="tight")


    # global throughput
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    # plot bars and error bars
    for i, sys in enumerate(SYSTEM_NAME_LIST):
        sys_values = [total_consumed_results[param][i] for param in DATARATE_LIST]
        # print(sys, sys_values)
        ax.bar(x + i * bar_width, sys_values, width=bar_width, label=SYSTEM_NAME_TO_LEGEND_DICT[sys], hatch=SYSTEM_NAME_TO_HATCH_DICT[sys], edgecolor='black')
    
    ax.set_xlabel('Data Rate (req./sec.)', AXLABEL_KW)
    ax.set_ylabel('Avg. Infer. Thpt. (req./sec.)', AXLABEL_KW)

    ax.set_xticks(x + bar_width * (num_systems - 1) / 2)
    ax.set_xticklabels([str(param) for param in DATARATE_LIST], **YTICK_LABEL_KW)

    ax.set_ylim([500, 5000])
    ax.set_yticks(np.arange(0, 6000, 1000))
    ax.set_yticklabels(np.arange(0, 6000, 1000), **YTICK_LABEL_KW)

    ax.legend(frameon=False, columnspacing=LEGEND_COLSPACING, prop=LEGEND_PROP, loc="upper center", ncol=2)
    
    output_path = f"{args.output_file_basename}_global.png"
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    output_path = f"{args.output_file_basename}_global.pdf"
    fig.savefig(output_path, format="pdf", dpi=600, bbox_inches="tight")


    # goodput 
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    # plot bars and error bars
    for i, sys in enumerate(SYSTEM_NAME_LIST):
        sys_values = [infer_delay_miss_count_results[param][i] for param in DATARATE_LIST]
        # print(sys, sys_values)
        ax.bar(x + i * bar_width, sys_values, width=bar_width, label=SYSTEM_NAME_TO_LEGEND_DICT[sys], hatch=SYSTEM_NAME_TO_HATCH_DICT[sys], edgecolor='black')
    
    ax.set_xlabel('Data Rate (req./sec.)', AXLABEL_KW)
    ax.set_ylabel('Avg. Infer. Thpt. (req./sec.)', AXLABEL_KW)

    ax.set_xticks(x + bar_width * (num_systems - 1) / 2)
    ax.set_xticklabels([str(param) for param in DATARATE_LIST], **YTICK_LABEL_KW)

    ax.set_ylim([500, 5000])
    ax.set_yticks(np.arange(0, 6000, 1000))
    ax.set_yticklabels(np.arange(0, 6000, 1000), **YTICK_LABEL_KW)

    ax.legend(frameon=False, columnspacing=LEGEND_COLSPACING, prop=LEGEND_PROP, loc="upper center", ncol=2)
    
    output_path = f"{args.output_file_basename}_global_goodput.png"
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    output_path = f"{args.output_file_basename}_global_goodput.pdf"
    fig.savefig(output_path, format="pdf", dpi=600, bbox_inches="tight")
