"""
Courtesy of ChatGPT

It expects data like following format
data_dir/
├── SystemA/
│   ├── param1.csv
│   ├── param2.csv
│   └── ...
├── SystemB/
│   ├── param1.csv
│   ├── param2.csv
│   └── ...
...
"""

import re
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

from plot_parameters import CSV_FILENAME_FMT, SYSTEM_NAME_LIST, SYSTEM_NAME_TO_LEGEND_DICT, SYSTEM_NAME_TO_HATCH_DICT, CSV_FILENAME_FMT_TRANSMISSION, AXLABEL_KW, YTICK_LABEL_KW, LEGEND_PROP, LEGEND_COLSPACING

IPR_RATE_LIST = [16]
DEADLINE_LIST = [80]
DATARATE_LIST = [1000, 2000, 3000, 4000, 5000] # [5000 ,6000, 7000, 8000, 9000, 10000] # 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, help="path containing the result directory in approproate format")
    args = parser.parse_args()

    # Dictionary to store average values: {param: [sys1_val, sys2_val, ...]}
    missrate_results = {param: [] for param in DATARATE_LIST}

    # Read values and compute means
    for param in DATARATE_LIST:
        for sys in SYSTEM_NAME_LIST:
            param_path = os.path.join(args.dir, sys, CSV_FILENAME_FMT_TRANSMISSION.format(sys, DEADLINE_LIST[0], param, IPR_RATE_LIST[0]))

            with open(param_path, 'r') as f:
                for line in f.readlines():
                    tokens = line.split(",")

                    missrate_percentage = float(tokens[6])*100
                    missrate_results[param].append(missrate_percentage)

    # Plotting
    num_params = len(DATARATE_LIST)
    num_systems = len(SYSTEM_NAME_LIST)
    bar_width = 0.8 / num_systems
    x = np.arange(num_params)

    fig, axs = plt.subplots(2, 2)

    ax = axs[1][0]
    for i, sys in enumerate(SYSTEM_NAME_LIST):
        offsets = x + i * bar_width
        values = [missrate_results[param][i] for param in DATARATE_LIST]
        print("missrate ", sys, values)
        ax.bar(
            offsets, values, width=bar_width,
            label=SYSTEM_NAME_TO_LEGEND_DICT[sys],
            hatch=SYSTEM_NAME_TO_HATCH_DICT[sys]
        )

    ax.set_xticks(x + bar_width * (num_systems - 1) / 2)
    ax.set_xticklabels([param for param in DATARATE_LIST], **AXLABEL_KW)
    ax.set_yticks(np.arange(0, 90, 20))
    ax.set_yticklabels(np.arange(0, 90, 20), **YTICK_LABEL_KW)
    ax.set_ylabel("Miss Rate(%)", **AXLABEL_KW)
    ax.set_ylim([0, 80])
    ax.set_xlabel("Data Rate (req./sec.)", **AXLABEL_KW)
    ax.set_title("(c) Miss Rate", fontsize=12, ha='center', y=-0.5, va='top')
    # ax.text(2.5, -20,"(c) Miss Rate", ha="center", va="bottom", fontsize=12)
    
    
    # nn error results
    # Dictionary to store average values: {param: [sys1_val, sys2_val, ...]}
    mse_results = {param: [] for param in DATARATE_LIST}
    mse_results_whole = {param: [] for param in DATARATE_LIST}

    # Read values and compute means
    for param in DATARATE_LIST:
        for sys in SYSTEM_NAME_LIST:
            param_path = os.path.join(args.dir, sys, CSV_FILENAME_FMT.format(sys, DEADLINE_LIST[0], param, IPR_RATE_LIST[0]))

            with open(param_path, 'r') as f:
                for line in f.readlines():
                    tokens = line.split(",")

                    ph_error = float(tokens[1])
                    amp_error = float(tokens[0])
                    ph_error_inferonly = float(tokens[3])
                    amp_error_inferonly = float(tokens[2])
                    mse_results[param].append(ph_error_inferonly)
                    mse_results_whole[param].append(ph_error)

    # Plotting
    num_params = len(DATARATE_LIST)
    num_systems = len(SYSTEM_NAME_LIST)
    bar_width = 0.8 / num_systems
    x = np.arange(num_params)

    ax = axs[0][0]

    for i, sys in enumerate(SYSTEM_NAME_LIST):
        offsets = x + i * bar_width
        values = [mse_results[param][i]/mse_results[param][0] for param in DATARATE_LIST]
        print("mse ", sys, values)
        ax.bar(
            offsets, values, width=bar_width,
            hatch=SYSTEM_NAME_TO_HATCH_DICT[sys]
        )

    ax.set_xticks(x + bar_width * (num_systems - 1) / 2)
    ax.set_xticklabels([param for param in DATARATE_LIST], **YTICK_LABEL_KW)
    ax.set_ylabel("Normalized MSE", **AXLABEL_KW)
    ax.set_xlabel("Data Rate (req./sec.)", **AXLABEL_KW)
    ax.set_title("(a) NN Inference MSE", fontsize=12, ha='center', y=-0.5, va='top')
    

    ax = axs[0][1]
    for i, sys in enumerate(SYSTEM_NAME_LIST):
        offsets = x + i * bar_width
        values = [mse_results_whole[param][i]/mse_results_whole[param][0] for param in DATARATE_LIST]
        print("re. mse ", sys, values)
        ax.bar(
            offsets, values, width=bar_width,
            hatch=SYSTEM_NAME_TO_HATCH_DICT[sys]
        )

    ax.set_xticks(x + bar_width * (num_systems - 1) / 2)
    ax.set_xticklabels([param for param in DATARATE_LIST], **YTICK_LABEL_KW)
    ax.set_ylabel("Normalized MSE", **AXLABEL_KW)
    ax.set_xlabel("Data Rate (req./sec.)", **AXLABEL_KW)
    ax.set_title("(b) Reconstruction MSE", fontsize=12, ha='center', y=-0.5, va='top')

    # infer throughput results
    ax = axs[1][1]

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
    # plot bars and error bars
    for i, sys in enumerate(SYSTEM_NAME_LIST):
        sys_values = [total_consumed_results[param][i] for param in DATARATE_LIST]
        print("throughput ", sys, sys_values)
        # print(sys, sys_values)
        ax.bar(x + i * bar_width, sys_values, width=bar_width, hatch=SYSTEM_NAME_TO_HATCH_DICT[sys], edgecolor='black')
    
    ax.set_xlabel('Data Rate (req./sec.)', AXLABEL_KW)
    ax.set_ylabel('Avg. Infer. Thpt. (req./sec.)', AXLABEL_KW)

    ax.set_xticks(x + bar_width * (num_systems - 1) / 2)
    ax.set_xticklabels([str(param) for param in DATARATE_LIST], **YTICK_LABEL_KW)

    ax.set_ylim([500, 5000])
    ax.set_yticks(np.arange(0, 6000, 1000))
    ax.set_yticklabels(np.arange(0, 6000, 1000), **YTICK_LABEL_KW)
    ax.set_title("(d) Inference Throughput", fontsize=12, ha='center', y=-0.5, va='top')

    fig.legend(frameon=False, loc="upper center", ncol=3, prop=LEGEND_PROP, columnspacing=LEGEND_COLSPACING)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.subplots_adjust(wspace=0.4, hspace=0.7)

    fig.savefig("fig_europar_final_eval.png", dpi=600, bbox_inches="tight")
    fig.savefig("fig_europar_final_eval.pdf", format="pdf", dpi=600, bbox_inches="tight")
