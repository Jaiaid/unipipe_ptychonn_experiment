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

from plot_parameters import SYSTEM_NAME_LIST, SYSTEM_NAME_TO_LEGEND_DICT, SYSTEM_NAME_TO_HATCH_DICT, CSV_FILENAME_FMT_TRANSMISSION

IPR_RATE_LIST = [16]
DEADLINE_LIST = [80]
DATARATE_LIST = [3000]
BS_LIST = [16, 32, 64, 128]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, help="path containing the result directory in approproate format")
    parser.add_argument("--output-file-basename", "-o", type=str, help="output name without extension")
    args = parser.parse_args()

    # Dictionary to store average values: {param: [sys1_val, sys2_val, ...]}
    results = {param: [] for param in BS_LIST}

    # Read values and compute means
    for param in BS_LIST:
        for sys in SYSTEM_NAME_LIST:
            param_path = os.path.join(args.dir, "bs{0}".format(param), sys, CSV_FILENAME_FMT_TRANSMISSION.format(sys, DEADLINE_LIST[0], DATARATE_LIST[0], IPR_RATE_LIST[0]))

            with open(param_path, 'r') as f:
                for line in f.readlines():
                    tokens = line.split(",")

                    missrate_percentage = float(tokens[6])*100
                    results[param].append(missrate_percentage)

    # Plotting
    num_params = len(BS_LIST)
    num_systems = len(SYSTEM_NAME_LIST)
    x = np.arange(num_systems)

    fig, ax = plt.subplots(figsize=(4, 2.25))

    for param in BS_LIST:
        values = results[param]
        print(param, values)
        ax.plot(
            x, values, label=param, marker='x', markersize=8, linestyle='--'
        )

    ax.set_xticks(np.arange(num_params+1))
    ax.set_xticklabels([SYSTEM_NAME_TO_LEGEND_DICT[sys] for sys in SYSTEM_NAME_LIST], fontsize=4)
    ax.set_ylabel("Miss Rate(%)")
    ax.legend()

    fig.savefig("{0}.png".format(args.output_file_basename), dpi=600, bbox_inches="tight")
    fig.savefig("{0}.pdf".format(args.output_file_basename), format="pdf", dpi=600, bbox_inches="tight")
