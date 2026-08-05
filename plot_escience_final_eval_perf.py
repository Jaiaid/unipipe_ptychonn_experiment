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

from plot_parameters import SYSTEM_NAME_TO_MARKER_DICT, CSV_FILENAME_FMT, SYSTEM_NAME_LIST, SYSTEM_NAME_TO_LEGEND_DICT, SYSTEM_NAME_TO_HATCH_DICT, CSV_FILENAME_FMT_TRANSMISSION, AXLABEL_KW, YTICK_LABEL_KW, LEGEND_PROP, LEGEND_COLSPACING


DIR1 = "result_logs/smalldataset_dratevariation"
DIR2 = "result_logs/smalldataset_bratevariation_old_old"
DIR3 = "result_logs/smalldataset_bratevariation_old"

LEGEND_PROP = {"size": 12, "weight": "bold"}
LEGEND_COLSPACING = 1
AXLABEL_KW = {"fontsize": 13, "fontweight": "bold"} 
YTICK_LABEL_KW = {"fontsize": 11, "fontweight": "bold" }
SUBCAPTION_FONT_SIZE = 14

if __name__ == "__main__":
    IPR_RATE_LIST = [16]
    DEADLINE_LIST = [80]
    DATARATE_LIST = [1000, 2000, 3000, 4000, 5000] # [5000 ,6000, 7000, 8000, 9000, 10000] # 

    # Plotting
    num_params = len(DATARATE_LIST)
    num_systems = len(SYSTEM_NAME_LIST)
    bar_width = 0.8 / num_systems
    x = np.arange(num_params)

    fig, axs = plt.subplots(1, 4, figsize=(16, 3.9))
    # Prevent overlap by shifting margins manually as fractions of the figure size
    # fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)


    # Dictionary to store average values: {param: [sys1_val, sys2_val, ...]}
    missrate_results = {param: [[] for _ in SYSTEM_NAME_LIST] for param in DATARATE_LIST}

    # Read values and compute means
    for param in DATARATE_LIST:
        for i, sys in enumerate(SYSTEM_NAME_LIST):
            param_path = os.path.join(DIR1, sys, CSV_FILENAME_FMT_TRANSMISSION.format(sys, DEADLINE_LIST[0], param, IPR_RATE_LIST[0]))

            with open(param_path, 'r') as f:
                for line in f.readlines():
                    tokens = line.split(",")

                    missrate_percentage = float(tokens[6])*100
                    missrate_results[param][i].append(missrate_percentage)


    ax = axs[0]
    for i, sys in enumerate(SYSTEM_NAME_LIST):
        offsets = x + i * bar_width
        values = [np.mean(missrate_results[param][i]) for param in DATARATE_LIST]
        print("missrate ", sys, values)
        ax.bar(
            offsets, values, width=bar_width,
            label=SYSTEM_NAME_TO_LEGEND_DICT[sys],
            hatch=SYSTEM_NAME_TO_HATCH_DICT[sys],
            yerr=np.array([(values[j]-np.min(missrate_results[param][i]), np.max(missrate_results[param][i])-values[j]) for j, param in enumerate(DATARATE_LIST)]).T,
            edgecolor='black'
        )
        print(np.array([(values[j]-np.min(missrate_results[param][i]), np.max(missrate_results[param][i])-values[j]) for j, param in enumerate(DATARATE_LIST)]))

    ax.set_xticks(x + bar_width * (num_systems - 1) / 2)
    ax.set_xticklabels([param for param in DATARATE_LIST], **YTICK_LABEL_KW)
    ax.set_yticks(np.arange(0, 90, 20))
    ax.set_yticklabels(np.arange(0, 90, 20), **YTICK_LABEL_KW)
    ax.set_ylabel("Miss Rate(%)", **AXLABEL_KW)
    ax.set_ylim([0, 80])
    ax.set_xlabel("Data Rate (req./sec.)", **AXLABEL_KW)
    ax.set_title("(a) Miss Rate", fontsize=SUBCAPTION_FONT_SIZE, ha='center', y=-0.4, va='top')
    # ax.text(2.5, -20,"(c) Miss Rate", ha="center", va="bottom", fontsize=12)
    
    
    # infer throughput results
    ax = axs[1]

    results = {param: [] for param in DATARATE_LIST}
    fluctuations = {param: [] for param in DATARATE_LIST}
    total_consumed_results = {param: [] for param in DATARATE_LIST}
    start_timestamp = None
    end_timestamp = None

    infer_delay_miss_count_results = {param: [[] for _ in SYSTEM_NAME_LIST] for param in DATARATE_LIST}
    iter_latency_results = {param: [[] for _ in SYSTEM_NAME_LIST] for param in DATARATE_LIST}
    # infer bs detection state
    infer_bs_regex_match_state = True

    # Read values and compute means
    for param in DATARATE_LIST:
        for i, sys in enumerate(SYSTEM_NAME_LIST):
            # for log file multicontext is named differently
            if sys == "multicontext":
                log_path = os.path.join(DIR1, sys, CSV_FILENAME_FMT.format(sys, DEADLINE_LIST[0], param, IPR_RATE_LIST[0]).replace('.csv', '_infer.log'))
            else:
                log_path = os.path.join(DIR1, sys, CSV_FILENAME_FMT.format(sys, DEADLINE_LIST[0], param, IPR_RATE_LIST[0]).replace('.csv', '.log'))

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
                        iter_latency_results[param][i].append(gap_value*1000/batch_size) # convert to ms
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

                infer_delay_miss_count_results[param][i].append((total_consumed-infer_delay_miss_count)/(end_timestamp-start_timestamp))
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
    ax.set_title("(b) Infer. Throughput", fontsize=SUBCAPTION_FONT_SIZE, ha='center', y=-0.4, va='top')
    
    ax.set_xlabel('Data Rate (req./sec.)', AXLABEL_KW)
    ax.set_ylabel('Avg. Infer. Thpt.\n(req./sec.)', AXLABEL_KW)

    ax.set_xticks(x + bar_width * (num_systems - 1) / 2)
    ax.set_xticklabels([str(param) for param in DATARATE_LIST], **YTICK_LABEL_KW)


    # forward pass latency results
    ax = axs[3]
    # plot bars and error bars
    for i, sys in enumerate(SYSTEM_NAME_LIST):
        sys_values = [np.mean(iter_latency_results[param][i]) for param in DATARATE_LIST]
        yerr = np.array([(sys_values[j]-np.min(iter_latency_results[param][i]), np.quantile(iter_latency_results[param][i], 0.95)-sys_values[j]) for j, param in enumerate(DATARATE_LIST)]).T
        yerr_percentage = [(yerr[0][j]/sys_values[j]*100, yerr[1][j]/sys_values[j]*100) for j in range(len(sys_values))]
        print("iteration latency ", sys, sys_values, yerr_percentage)
        # print(sys, sys_values)
        # print(sys, [(np.min(iter_latency_results[param][i]), np.max(iter_latency_results[param][i])) for param in DATARATE_LIST])
        # we do not take std as error bar lower value because it goes to negative for some cases, 
        # instead we take min value as lower error bar and std as upper error bar to remove outliers effect and show the fluctuation more clearly
        ax.bar(
            x + i * bar_width, sys_values, width=bar_width, hatch=SYSTEM_NAME_TO_HATCH_DICT[sys],
            yerr=yerr, edgecolor='black',
        )
    
    ax.set_xlabel('Data Rate (req./sec.)', AXLABEL_KW)
    ax.set_ylabel('Iteration Latency (ms)', AXLABEL_KW)

    ax.set_xticks(x + bar_width * (num_systems - 1) / 2)
    ax.set_xticklabels([str(param) for param in DATARATE_LIST], **AXLABEL_KW)

    # ax.set_ylim([500, 5000])
    # ax.set_yticks(np.arange(0, 6000, 1000))
    # ax.set_yticklabels(np.arange(0, 6000, 1000), **YTICK_LABEL_KW)
    ax.set_title("(d) Iteration Latency", fontsize=SUBCAPTION_FONT_SIZE, ha='center', y=-0.4, va='top')
    
    ax.set_xlabel('Data Rate (req./sec.)', AXLABEL_KW)
    ax.set_ylabel('Iteration Latency (ms)', AXLABEL_KW)

    ax.set_xticks(x + bar_width * (num_systems - 1) / 2)
    ax.set_xticklabels([str(param) for param in DATARATE_LIST], **YTICK_LABEL_KW)


    IPR_RATE_LIST = [10, 20, 30, 40, 50, 60]
    DEADLINE_LIST = [80]
    DATARATE_LIST = [3000] # [5000 ,6000, 7000, 8000, 9000, 10000] # 

    ax = axs[2]
    num_params = len(IPR_RATE_LIST)
    num_systems = len(SYSTEM_NAME_LIST)
    bar_width = 0.8 / num_systems
    x = np.arange(num_params)

    # Dictionary to store average values: {param: [sys1_val, sys2_val, ...]}
    results = {param: [[] for _ in SYSTEM_NAME_LIST] for param in IPR_RATE_LIST}

    # Read values and compute means
    for param in IPR_RATE_LIST:
        for i, sys in enumerate(SYSTEM_NAME_LIST):
            param_path = os.path.join(DIR2, sys, CSV_FILENAME_FMT_TRANSMISSION.format(sys, DEADLINE_LIST[0], DATARATE_LIST[0], param))

            with open(param_path, 'r') as f:
                for line in f.readlines():
                    tokens = line.split(",")

                    missrate_percentage = float(tokens[6])*100
                    results[param][i].append(missrate_percentage)

    mse_results_dataset1 = {param: [] for param in IPR_RATE_LIST}
    mse_results_whole_dataset1 = {param: [] for param in IPR_RATE_LIST}

    # Read values and compute means
    for param in IPR_RATE_LIST:
        for sys in SYSTEM_NAME_LIST:
            param_path = os.path.join(DIR2, sys, CSV_FILENAME_FMT.format(sys, DEADLINE_LIST[0], DATARATE_LIST[0], param))

            with open(param_path, 'r') as f:
                for line in f.readlines():
                    tokens = line.split(",")

                    ph_error = float(tokens[1])
                    amp_error = float(tokens[0])
                    ph_error_inferonly = float(tokens[3])
                    amp_error_inferonly = float(tokens[2])
                    mse_results_dataset1[param].append(ph_error_inferonly)
                    mse_results_whole_dataset1[param].append(ph_error)
                    break

    # for i, sys in enumerate(SYSTEM_NAME_LIST):
    #     offsets = x + i * bar_width
    #     values = [mse_results_dataset1[param][i]/mse_results_dataset1[param][0] for _, param in enumerate(IPR_RATE_LIST)]
    #     print("mse ", sys, values)
    #     ax.bar(
    #         offsets, values, width=bar_width,
    #         hatch=SYSTEM_NAME_TO_HATCH_DICT[sys]
    #     )


    for i, sys in enumerate(SYSTEM_NAME_LIST):
        offsets = x + i * bar_width
        values = [np.mean(results[param][i]) for param in IPR_RATE_LIST]
        yerr = np.array([(values[j] - min(results[param][i]), max(results[param][i]) - values[j]) for j, param in enumerate(IPR_RATE_LIST)]).T # [np.std(results[param][i]) for param in IPR_RATE_LIST]
        print(sys, values)
        ax.bar(
            offsets, values, width=bar_width,
            hatch=SYSTEM_NAME_TO_HATCH_DICT[sys],
            yerr=yerr
        )

    # Dictionary to store average values: {param: [sys1_val, sys2_val, ...]}
    iter_latency_results = {param: [[] for _ in SYSTEM_NAME_LIST] for param in IPR_RATE_LIST}
    infer_bs_regex_match_state = True
    # Read values and compute means
    for param in IPR_RATE_LIST:
        for i, sys in enumerate(SYSTEM_NAME_LIST):
            # for log file multicontext is named differently
            if sys == "multicontext":
                log_path = os.path.join(DIR2, sys, CSV_FILENAME_FMT.format(sys, DEADLINE_LIST[0], DATARATE_LIST[0], param).replace('.csv', '_infer.log'))
            else:
                log_path = os.path.join(DIR2, sys, CSV_FILENAME_FMT.format(sys, DEADLINE_LIST[0], DATARATE_LIST[0], param).replace('.csv', '.log'))

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
                        iter_latency_results[param][i].append(gap_value*1000/batch_size) # convert to ms
                        infer_bs_regex_match_state = True
    
    total_consumed_results = {param: [] for param in IPR_RATE_LIST}
    # Read values and compute means
    for param in IPR_RATE_LIST:
        for i, sys in enumerate(SYSTEM_NAME_LIST):
            # for log file multicontext is named differently
            if sys == "multicontext":
                log_path = os.path.join(DIR2, sys, CSV_FILENAME_FMT.format(sys, DEADLINE_LIST[0], DATARATE_LIST[0], param).replace('.csv', '_train.log'))
            else:
                log_path = os.path.join(DIR2, sys, CSV_FILENAME_FMT.format(sys, DEADLINE_LIST[0], DATARATE_LIST[0], param).replace('.csv', '.log'))

            total_consumed = 0
            infer_delay_miss_count = 0
            start_timestamp = None
            end_timestamp = None
            
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    for line in f.readlines():
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
                            continue

                        if sys == "unipipe_dp" or sys == "unipipe":
                            match = re.search(r'\[(\d+\.\d+)\] UNIPIPE DP TRAIN, INFER BS,(\d+),(\d+),*', line)
                            if match:
                                total_consumed += int(match.group(2))
                        else:
                            match = re.search(r'\[(\d+\.\d+)\] MULTICONTEXT TRAIN BS,(\d+)', line)
                            if match:
                                total_consumed += int(match.group(2))
                        if match:
                            end_timestamp = float(match.group(1))
                            # print(f"Updated end timestamp: {end_timestamp}, total consumed: {total_consumed}")
                if start_timestamp is not None and end_timestamp is not None and end_timestamp > start_timestamp:
                    total_consumed_results[param].append(total_consumed/(end_timestamp-start_timestamp))
                else:
                    total_consumed_results[param].append(0)
            else:
                total_consumed_results[param].append(0)
            if param == IPR_RATE_LIST[-1] and sys in ["multicontext", "unipipe_dp"]:
                print("Hi", sys, param, end_timestamp-start_timestamp, total_consumed, total_consumed/(end_timestamp-start_timestamp))    


    ax.set_xticks(x + bar_width * (num_systems - 1) / 2)
    ax.set_xticklabels([param for param in IPR_RATE_LIST], **YTICK_LABEL_KW)
    ax.set_yticks(np.arange(0, 90, 20))
    ax.set_yticklabels(np.arange(0, 90, 20), **YTICK_LABEL_KW)
    ax.set_ylabel("Miss Rate(%)", **AXLABEL_KW)
    ax.set_ylim([0, 100])
    ax.set_xlabel("Computation Rate (sample/sec.)", **AXLABEL_KW)
    ax.legend(frameon=False, ncol=1, prop=LEGEND_PROP, columnspacing=LEGEND_COLSPACING)

    ax1 = ax.twinx()
    for i, sys in enumerate(SYSTEM_NAME_LIST):
        sys_values = [total_consumed_results[param][i] for param in IPR_RATE_LIST]
        print(sys, sys_values)
        # sys_error = np.array([(sys_values[j] - np.min(iter_latency_results[param][i]), np.quantile(iter_latency_results[param][i], 0.99)-sys_values[j]) for j, param in enumerate(IPR_RATE_LIST)]).T
        ax1.errorbar(x + i * bar_width, sys_values, linestyle='--', marker=SYSTEM_NAME_TO_MARKER_DICT[sys], markersize=5, capsize=5)
        # ax1.errorbar(x, sys_values, yerr=sys_error, fmt='o', color='black', capsize=5)

    ax1.set_ylabel('Avg. Train Thpt.\n(sample/sec.)', **AXLABEL_KW)
    ax1.set_yticks(np.arange(0, 80, 20))
    ax1.set_yticklabels(np.arange(0, 80, 20), **YTICK_LABEL_KW)
    ax1.set_xticks(x + bar_width * (num_systems - 1) / 2)
    ax1.set_xticklabels([str(param) for param in IPR_RATE_LIST], **YTICK_LABEL_KW)

    # ax1 = ax.twinx()
    # for i, sys in enumerate(SYSTEM_NAME_LIST):
    #     sys_values = [np.mean(iter_latency_results[param][i]) for param in IPR_RATE_LIST]
    #     sys_error = np.array([(sys_values[j] - np.min(iter_latency_results[param][i]), np.quantile(iter_latency_results[param][i], 0.99)-sys_values[j]) for j, param in enumerate(IPR_RATE_LIST)]).T
    #     ax1.errorbar(x + i * bar_width, sys_values, yerr=sys_error, linestyle='--', marker=SYSTEM_NAME_TO_MARKER_DICT[sys], markersize=5, capsize=5)
        # ax1.errorbar(x, sys_values, yerr=sys_error, fmt='o', color='black', capsize=5)

    # ax1.set_ylabel('Avg. Iteration Latency (ms)')
    # ax1.set_xticks(x + bar_width * (num_systems - 1) / 2)
    # ax1.set_xticklabels([str(param) for param in IPR_RATE_LIST])

    ax.set_title("(c) Computation Rate Sensitivity", fontsize=SUBCAPTION_FONT_SIZE, ha='center', y=-0.4, va='top')


    fig.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1), prop=LEGEND_PROP, columnspacing=LEGEND_COLSPACING)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.subplots_adjust(wspace=0.55)
    fig.savefig("fig_escience_final_eval_perf.png", dpi=600, bbox_inches="tight")
    fig.savefig("fig_escience_final_eval_perf.pdf", format="pdf", dpi=600, bbox_inches="tight")
