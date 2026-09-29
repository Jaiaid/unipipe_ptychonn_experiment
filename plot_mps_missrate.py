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

import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

from plot_parameters import SYSTEM_NAME_TO_LEGEND_DICT, SYSTEM_NAME_TO_HATCH_DICT, CSV_FILENAME_FMT_TRANSMISSION, AXLABEL_KW, YTICK_LABEL_KW, LEGEND_PROP, LEGEND_COLSPACING

IPR_RATE_LIST = [16]
DEADLINE_LIST = [80]
DATARATE_LIST = [1000, 2000, 3000, 4000, 5000] # [5000 ,6000, 7000, 8000, 9000, 10000] # 

SYSTEM_NAME_LIST = SYSTEM_NAME_LIST = ["multicontext_mps_70_30", "multicontext_mps_50_50", "multicontext_mps_30_70", "unipipe_dp"]
LEGEND_PROP = {"size": 12, "weight": "bold"}
LEGEND_COLSPACING = 1
AXLABEL_KW = {"fontsize": 13, "fontweight": "bold"} 
YTICK_LABEL_KW = {"fontsize": 13, "fontweight": "bold" }
SUBCAPTION_FONT_SIZE = 14

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, help="path containing the result directory in approproate format")
    parser.add_argument("--output-file-basename", "-o", type=str, help="output name without extension")
    args = parser.parse_args()

    # Dictionary to store average values: {param: [sys1_val, sys2_val, ...]}
    results = {param: [[] for _ in SYSTEM_NAME_LIST] for param in DATARATE_LIST}

    # Read values and compute means
    for param in DATARATE_LIST:
        for i, sys in enumerate(SYSTEM_NAME_LIST):
            param_path = os.path.join(args.dir, sys, CSV_FILENAME_FMT_TRANSMISSION.format(sys, DEADLINE_LIST[0], param, IPR_RATE_LIST[0]))
            print(f"Reading data from {param_path}")
            with open(param_path, 'r') as f:
                for line in f.readlines():
                    tokens = line.split(",")

                    missrate_percentage = float(tokens[6])*100
                    results[param][i].append(missrate_percentage)

    # Plotting
    num_params = len(DATARATE_LIST)
    num_systems = len(SYSTEM_NAME_LIST)
    bar_width = 0.8 / num_systems
    x = np.arange(num_params)

    fig, ax = plt.subplots(figsize=(4, 2.4))

    for i, sys in enumerate(SYSTEM_NAME_LIST):
        offsets = x + i * bar_width
        values = [np.mean(results[param][i]) for param in DATARATE_LIST]
        yerr = np.array([(values[j] - np.min(results[param][i]), np.max(results[param][i]) - values[j]) for j, param in enumerate(DATARATE_LIST)]).T
        print(f"System: {sys}, Values: {values}, legend: {SYSTEM_NAME_TO_LEGEND_DICT[sys]}, hatch: {SYSTEM_NAME_TO_HATCH_DICT[sys]}, yerr: {yerr}")
        ax.bar(
            offsets, values, width=bar_width,
            label=SYSTEM_NAME_TO_LEGEND_DICT[sys],
            hatch=SYSTEM_NAME_TO_HATCH_DICT[sys],
            yerr=yerr
        )

    ax.set_xticks(x + bar_width * (num_systems - 1) / 2)
    ax.set_xticklabels([param for param in DATARATE_LIST], **AXLABEL_KW)
    ax.set_yticks(np.arange(0, 90, 20))
    ax.set_yticklabels(np.arange(0, 90, 20), **YTICK_LABEL_KW)
    ax.set_ylabel("Miss Rate(%)", **AXLABEL_KW)
    ax.set_ylim([0, 80])
    ax.set_xlabel("Data Rate (req./sec.)", **AXLABEL_KW)
    ax.legend(frameon=False, ncol=1, prop=LEGEND_PROP, columnspacing=LEGEND_COLSPACING)

    fig.savefig("{0}.png".format(args.output_file_basename), dpi=600, bbox_inches="tight")
    fig.savefig("{0}.pdf".format(args.output_file_basename), format="pdf", dpi=600, bbox_inches="tight")
