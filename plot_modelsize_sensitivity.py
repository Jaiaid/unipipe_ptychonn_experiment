"""
Courtesy of Gemini, ChatGPT

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

import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

from plot_parameters import CSV_FILENAME_FMT, SYSTEM_NAME_LIST, SYSSTAT_FILENAME_FMT, SYSTEM_NAME_TO_LEGEND_DICT, SYSTEM_NAME_TO_MARKER_DICT, SYSTEM_NAME_TO_HATCH_DICT, CSV_FILENAME_FMT_TRANSMISSION, FIGSIZE, AXLABEL_KW, YTICK_LABEL_KW, LEGEND_COLSPACING, LEGEND_PROP

IPR_RATE_LIST = [16]
DEADLINE_LIST = [80]
DATARATE_LIST = [3000]
MODEL_LIST = ['1.25M', '5M', '10M', '20M']
# modifying SYSTEM_NAME_LIST to be different from the other script
# as this experiment only ran a subset of systems, multicontext, unipipe, unipipe_dp
SYSTEM_NAME_LIST = ["multicontext", "unipipe_dp"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, help="path containing the result directory in approproate format")
    parser.add_argument("--output-file-basename", "-o", type=str, help="output name without extension")
    args = parser.parse_args()

    # Dictionary to store average values: {param: [sys1_val, sys2_val, ...]}
    results = {sys: [] for sys in SYSTEM_NAME_LIST}
    gpuutil_dict = {sys: [] for sys in SYSTEM_NAME_LIST}
    mse_error_dict = {sys: [] for sys in SYSTEM_NAME_LIST}

    # Read values and compute means
    for param in MODEL_LIST:
        for sys in SYSTEM_NAME_LIST:
            param_path = os.path.join(
                args.dir, "modeltype_{0}".format(param), sys, 
                CSV_FILENAME_FMT_TRANSMISSION.format(
                    sys, DEADLINE_LIST[0], DATARATE_LIST[0], IPR_RATE_LIST[0]
                )
            )

            sysstat_path = os.path.join(
                args.dir, "modeltype_{0}".format(param), sys, 
                SYSSTAT_FILENAME_FMT.format(
                    sys, DEADLINE_LIST[0], DATARATE_LIST[0], IPR_RATE_LIST[0]
                )
            )

            mse_error_path = os.path.join(
                args.dir, "modeltype_{0}".format(param), sys, 
                CSV_FILENAME_FMT.format(
                    sys, DEADLINE_LIST[0], DATARATE_LIST[0], IPR_RATE_LIST[0]
                )
            )

            if sys == "multicontext":
                log_path = param_path.replace("transmission_state_", "").replace(".csv", "_infer.log")
            else:
                log_path = param_path.replace("transmission_state_", "").replace(".csv", ".log")

            if os.path.exists(log_path):
                with open(param_path, 'r') as f:
                    for line in f.readlines():
                        tokens = line.split(",")

                        missrate_percentage = float(tokens[6])*100
                        results[sys].append(missrate_percentage)
            else:
                print(f"File not found: {log_path}, appending NaN as inferrence result did not run due to OOM")
                results[sys].append(np.nan)  # Append NaN if file is missing

            if os.path.exists(mse_error_path):
                with open(mse_error_path, 'r') as f:
                    for line in f.readlines():
                        tokens = line.split(",")

                        ph_error = float(tokens[1])
                        amp_error = float(tokens[0])
                        ph_error_inferonly = float(tokens[3])
                        amp_error_inferonly = float(tokens[2])
                        mse_error_dict[sys].append(ph_error)

            with open(sysstat_path, 'r') as f:
                gpu_util = []
                for i, line in enumerate(f.readlines()):
                    if i == 0:
                        continue  # skip header
                    parts = line.strip().split(',')
                    
                    gpu_util.append(float(parts[4]))  # GPU util is in 5th column
                p30 = np.percentile(gpu_util, 30)
                data = np.array(gpu_util)
                gpuutil_dict[sys].append(data[data>=p30].mean())  # Store the mean GPU utilization above 30th percentile

    # Plotting bar clusters for each model type
    # each bar cluster contains bar for each system
    num_params = len(MODEL_LIST)
    num_systems = len(SYSTEM_NAME_LIST)
    bar_width = 0.8 / num_systems
    x = np.arange(num_params)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax1 = ax.twinx()

    for i, sys in enumerate(SYSTEM_NAME_LIST):
        offsets = x + (i-1) * bar_width
        values = results[sys]
        # plot cross in place of bar if nan
        ax.bar(
            offsets, values, label=SYSTEM_NAME_TO_LEGEND_DICT[sys], width=bar_width, hatch=SYSTEM_NAME_TO_HATCH_DICT[sys]
        )

        for j, val in enumerate(values):
            if np.isnan(val):
                ax.text(
                    offsets[j], 0.05, 'x', ha='center', va='bottom', color='red', fontsize=12
                )

    # ax1.set_ylabel('GPU Util (%)', color='blue', **AXLABEL_KW)
    # ax1.tick_params(axis='y', labelcolor='blue')
    # ax1.set_ylim([60, 120])
    # ax1.set_yticks(np.arange(30, 110, 10))
    # ax1.set_yticklabels(np.arange(30, 110, 10), **YTICK_LABEL_KW)

    # for sys in SYSTEM_NAME_LIST:
    #     gpu_util_list = [gpu_util for gpu_util in gpuutil_dict[sys]]
    #     ax1.plot(x, gpu_util_list, marker=SYSTEM_NAME_TO_MARKER_DICT[sys])
    #     print(sys, gpu_util_list)

    ax1.set_ylabel('Normalized MSE', color='blue', **AXLABEL_KW)
    # ax1.tick_params(axis='y', labelcolor='blue')
    # ax1.set_ylim([60, 120])
    # ax1.set_yticks(np.arange(30, 110, 10))
    # ax1.set_yticklabels(np.arange(30, 110, 10), **YTICK_LABEL_KW)

    for sys in SYSTEM_NAME_LIST:
        mse_error_list = [mse_error/mse_error_dict["multicontext"][i] for i, mse_error in enumerate(mse_error_dict[sys])]
        ax1.plot(x, mse_error_list, marker=SYSTEM_NAME_TO_MARKER_DICT[sys])
        print(sys, mse_error_list)


    ax.set_xticks(np.arange(num_params))
    ax.set_xticklabels([modeltype for modeltype in MODEL_LIST], **YTICK_LABEL_KW)
    ax.set_ylabel("Miss Rate(%)", **AXLABEL_KW)
    ax.set_yticks(np.arange(0, 140, 20))
    ax.set_yticklabels(np.arange(0, 140, 20), **YTICK_LABEL_KW)
    ax.legend(frameon=False, loc="upper left", ncol=2, columnspacing=LEGEND_COLSPACING, prop=LEGEND_PROP)

    fig.savefig("{0}.png".format(args.output_file_basename), dpi=600, bbox_inches="tight")
    fig.savefig("{0}.pdf".format(args.output_file_basename), format="pdf", dpi=600, bbox_inches="tight")
