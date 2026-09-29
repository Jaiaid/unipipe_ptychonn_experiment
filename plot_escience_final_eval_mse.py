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


DATASET_1_DIR = "result_logs_20260514/smalldataset_dratevariation"
DATASET_2_DIR = "result_logs_20260514/largedataset_dratevariation"

if __name__ == "__main__":

    # Plotting
    num_params = len(DATARATE_LIST)
    num_systems = len(SYSTEM_NAME_LIST)
    bar_width = 0.8 / num_systems
    x = np.arange(num_params)

    fig, axs = plt.subplots(2, 2)
    
    # nn error results
    # Dictionary to store average values: {param: [sys1_val, sys2_val, ...]}
    mse_results_dataset1 = {param: [] for param in DATARATE_LIST}
    mse_results_whole_dataset1 = {param: [] for param in DATARATE_LIST}
    mse_results_dataset2 = {param: [[] for _ in SYSTEM_NAME_LIST] for param in DATARATE_LIST}
    mse_results_whole_dataset2 = {param: [[] for _ in SYSTEM_NAME_LIST] for param in DATARATE_LIST}

    # Read values and compute means
    for param in DATARATE_LIST:
        for sys in SYSTEM_NAME_LIST:
            param_path = os.path.join(DATASET_1_DIR, sys, CSV_FILENAME_FMT.format(sys, DEADLINE_LIST[0], param, IPR_RATE_LIST[0]))

            with open(param_path, 'r') as f:
                for line in f.readlines():
                    tokens = line.split(",")

                    ph_error = float(tokens[1])
                    amp_error = float(tokens[0])
                    ph_error_inferonly = float(tokens[3])
                    amp_error_inferonly = float(tokens[2])
                    mse_results_dataset1[param].append(ph_error_inferonly)
                    mse_results_whole_dataset1[param].append(ph_error)

    for param in DATARATE_LIST:
        for i, sys in enumerate(SYSTEM_NAME_LIST):
            param_path = os.path.join(DATASET_2_DIR, sys, CSV_FILENAME_FMT.format(sys, DEADLINE_LIST[0], param, IPR_RATE_LIST[0]))

            with open(param_path, 'r') as f:
                for line in f.readlines():
                    tokens = line.split(",")

                    ph_error = float(tokens[1])
                    amp_error = float(tokens[0])
                    ph_error_inferonly = float(tokens[3])
                    amp_error_inferonly = float(tokens[2])
                    mse_results_dataset2[param][i].append(ph_error_inferonly)
                    mse_results_whole_dataset2[param][i].append(ph_error)
            print(sys, param, mse_results_dataset2[param][i], mse_results_whole_dataset2[param][i])

    # Plotting
    num_params = len(DATARATE_LIST)
    num_systems = len(SYSTEM_NAME_LIST)
    bar_width = 0.8 / num_systems
    x = np.arange(num_params)

    ax = axs[0][0]

    for i, sys in enumerate(SYSTEM_NAME_LIST):
        offsets = x + i * bar_width
        values = [np.mean(mse_results_dataset1[param][i])/np.mean(mse_results_dataset1[param][0]) for param in DATARATE_LIST]
        print("mse ", sys, values)
        ax.bar(
            offsets, values, width=bar_width,
            hatch=SYSTEM_NAME_TO_HATCH_DICT[sys], label=SYSTEM_NAME_TO_LEGEND_DICT[sys]
        )

    ax.set_xticks(x + bar_width * (num_systems - 1) / 2)
    ax.set_xticklabels([param for param in DATARATE_LIST], **YTICK_LABEL_KW)
    ax.set_ylabel("Normalized MSE", **AXLABEL_KW)
    ax.set_xlabel("Data Rate (req./sec.)", **AXLABEL_KW)
    ax.set_title("(a) NN Inference MSE\nDataset 1", fontsize=12, ha='center', y=-0.5, va='top')
    

    ax = axs[0][1]
    for i, sys in enumerate(SYSTEM_NAME_LIST):
        offsets = x + i * bar_width
        values = [np.mean(mse_results_whole_dataset1[param][i])/np.mean(mse_results_whole_dataset1[param][0]) for param in DATARATE_LIST]
        print("re. mse ", sys, values)
        ax.bar(
            offsets, values, width=bar_width,
            hatch=SYSTEM_NAME_TO_HATCH_DICT[sys]
        )

    ax.set_xticks(x + bar_width * (num_systems - 1) / 2)
    ax.set_xticklabels([param for param in DATARATE_LIST], **YTICK_LABEL_KW)
    ax.set_ylabel("Normalized MSE", **AXLABEL_KW)
    ax.set_xlabel("Data Rate (req./sec.)", **AXLABEL_KW)
    ax.set_title("(b) Reconstruction MSE\nDataset 1", fontsize=12, ha='center', y=-0.5, va='top')

    ax = axs[1][0]

    for i, sys in enumerate(SYSTEM_NAME_LIST):
        offsets = x + i * bar_width
        values = [np.mean(mse_results_dataset2[param][i])/np.mean(mse_results_dataset2[param][0]) for param in DATARATE_LIST]
        print("mse ", sys, values)
        ax.bar(
            offsets, values, width=bar_width,
            hatch=SYSTEM_NAME_TO_HATCH_DICT[sys]
        )

    ax.set_xticks(x + bar_width * (num_systems - 1) / 2)
    ax.set_xticklabels([param for param in DATARATE_LIST], **YTICK_LABEL_KW)
    ax.set_ylabel("Normalized MSE", **AXLABEL_KW)
    ax.set_xlabel("Data Rate (req./sec.)", **AXLABEL_KW)
    ax.set_title("(c) NN Inference MSE\nDataset 2", fontsize=12, ha='center', y=-0.5, va='top')
    

    ax = axs[1][1]
    for i, sys in enumerate(SYSTEM_NAME_LIST):
        offsets = x + i * bar_width
        values = [np.mean(mse_results_whole_dataset2[param][i])/np.mean(mse_results_whole_dataset2[param][0]) for param in DATARATE_LIST]
        print("re. mse ", sys, values)
        ax.bar(
            offsets, values, width=bar_width,
            hatch=SYSTEM_NAME_TO_HATCH_DICT[sys]
        )

    ax.set_xticks(x + bar_width * (num_systems - 1) / 2)
    ax.set_xticklabels([param for param in DATARATE_LIST], **YTICK_LABEL_KW)
    ax.set_ylabel("Normalized MSE", **AXLABEL_KW)
    ax.set_xlabel("Data Rate (req./sec.)", **AXLABEL_KW)
    ax.set_title("(d) Reconstruction MSE\nDataset 2", fontsize=12, ha='center', y=-0.5, va='top')

    # fig.legend()
    fig.legend(frameon=False, loc="upper center", ncol=3, prop=LEGEND_PROP, columnspacing=LEGEND_COLSPACING)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.subplots_adjust(wspace=0.4, hspace=0.75)

    fig.savefig("fig_escience_final_eval_mse.png", dpi=600, bbox_inches="tight")
    fig.savefig("fig_escience_final_eval_mse.pdf", format="pdf", dpi=600, bbox_inches="tight")
