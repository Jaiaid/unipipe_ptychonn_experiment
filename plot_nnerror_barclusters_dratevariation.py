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
import matplotlib.ticker as ticker
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
    results = {param: [] for param in DATARATE_LIST}
    results_whole = {param: [] for param in DATARATE_LIST}

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
                    results[param].append(ph_error_inferonly)
                    results_whole[param].append(ph_error)

    # Plotting
    num_params = len(DATARATE_LIST)
    num_systems = len(SYSTEM_NAME_LIST)
    bar_width = 0.8 / num_systems
    x = np.arange(num_params)

    fig, ax = plt.subplots(figsize=(4, 2.4))

    for i, sys in enumerate(SYSTEM_NAME_LIST):
        offsets = x + i * bar_width
        values = [results[param][i] for param in DATARATE_LIST]
        ax.bar(
            offsets, values, width=bar_width, 
            label=SYSTEM_NAME_TO_LEGEND_DICT[sys],
            hatch=SYSTEM_NAME_TO_HATCH_DICT[sys]
        )

    ax.set_xticks(x + bar_width * (num_systems - 1) / 2)
    ax.set_xticklabels([param for param in DATARATE_LIST], fontsize=10, fontweight="bold")
    ax.set_yticks(np.arange(0, 1.4, 0.2))
    ax.set_yticklabels(np.arange(0, 1.4, 0.2), fontsize=10, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
    ax.set_ylabel("MSE Error", fontsize=10, fontweight="bold")
    ax.set_xlabel("Data Rate (req./sec.)", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, loc="upper center", ncol=2, columnspacing=0.5, prop={"size": 9, "weight": "bold"})

    fig.savefig("{0}.png".format(args.output_file_basename), dpi=600, bbox_inches="tight")
    fig.savefig("{0}.pdf".format(args.output_file_basename), format="pdf", dpi=600, bbox_inches="tight")


    fig, ax = plt.subplots(figsize=(4, 2.4))

    for i, sys in enumerate(SYSTEM_NAME_LIST):
        offsets = x + i * bar_width
        values = [results_whole[param][i] for param in DATARATE_LIST]
        ax.bar(
            offsets, values, width=bar_width, label=SYSTEM_NAME_TO_LEGEND_DICT[sys],
            hatch=SYSTEM_NAME_TO_HATCH_DICT[sys]
        )

    ax.set_xticks(x + bar_width * (num_systems - 1) / 2)
    ax.set_xticklabels([param for param in DATARATE_LIST], fontsize=10, fontweight="bold")
    ax.set_yticks(np.arange(0, 1.4, 0.2))
    ax.set_yticklabels(np.arange(0, 1.4, 0.2), fontsize=10, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
    ax.set_ylabel("MSE Error", fontsize=10, fontweight="bold")
    ax.set_xlabel("Data Rate (req./sec.)", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, loc="upper center", ncol=2, columnspacing=0.5, prop={"size": 9, "weight": "bold"})

    fig.savefig("{0}_whole.png".format(args.output_file_basename), dpi=600, bbox_inches="tight")
    fig.savefig("{0}_whole.pdf".format(args.output_file_basename), format="pdf", dpi=600, bbox_inches="tight")
