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

from plot_parameters import SYSTEM_NAME_LIST, SYSTEM_NAME_TO_LEGEND_DICT, SYSTEM_NAME_TO_HATCH_DICT, CSV_FILENAME_FMT

IPR_RATE = 16
DEADLINE_LIST = [80, 160, 320]
DATARATE_LIST = [2000]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, help="path containing the result directory in approproate format")
    parser.add_argument("--output-file-basename", "-o", type=str, help="output name without extension")
    args = parser.parse_args()

    # Dictionary to store average values: {param: [sys1_val, sys2_val, ...]}
    results = {param: [] for param in DEADLINE_LIST}
    results_whole = {param: [] for param in DEADLINE_LIST}

    # Read values and compute means
    for param in DEADLINE_LIST:
        for sys in SYSTEM_NAME_LIST:
            param_path = os.path.join(args.dir, sys, CSV_FILENAME_FMT.format(sys, param, DATARATE_LIST[0], IPR_RATE))

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
    num_params = len(DEADLINE_LIST)
    num_systems = len(SYSTEM_NAME_LIST)
    bar_width = 0.8 / num_systems
    x = np.arange(num_params)

    fig, ax = plt.subplots(figsize=(4, 2.25))

    for i, sys in enumerate(SYSTEM_NAME_LIST):
        offsets = x + i * bar_width
        values = [results[param][i] for param in DEADLINE_LIST]
        ax.bar(
            offsets, values, width=bar_width, label=SYSTEM_NAME_TO_LEGEND_DICT[sys],
            hatch=SYSTEM_NAME_TO_HATCH_DICT[sys]
        )

    ax.set_xticks(x + bar_width * (num_systems - 1) / 2)
    ax.set_xticklabels([param for param in DEADLINE_LIST])
    ax.set_ylabel("MSE Error")
    ax.set_xlabel("Deadline (msec.)")
    ax.legend()

    fig.savefig("{0}.png".format(args.output_file_basename), dpi=600, bbox_inches="tight")
    fig.savefig("{0}.pdf".format(args.output_file_basename), format="pdf", dpi=600, bbox_inches="tight")


    fig, ax = plt.subplots(figsize=(4, 2.25))

    for i, sys in enumerate(SYSTEM_NAME_LIST):
        offsets = x + i * bar_width
        values = [results_whole[param][i] for param in DEADLINE_LIST]
        ax.bar(
            offsets, values, width=bar_width, label=SYSTEM_NAME_TO_LEGEND_DICT[sys],
            hatch=SYSTEM_NAME_TO_HATCH_DICT[sys]
        )

    ax.set_xticks(x + bar_width * (num_systems - 1) / 2)
    ax.set_xticklabels([param for param in DEADLINE_LIST])
    ax.set_ylabel("MSE Error")
    ax.set_xlabel("Deadline (msec.)")
    # ax.legend()

    fig.savefig("{0}_whole.png".format(args.output_file_basename), dpi=600, bbox_inches="tight")
    fig.savefig("{0}_whole.pdf".format(args.output_file_basename), format="pdf", dpi=600, bbox_inches="tight")
