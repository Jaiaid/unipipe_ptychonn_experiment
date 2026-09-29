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
import re

BATCH_LIST = range(1, 65)
NETWORK_LIST = ["resnet18", "resnet50", "ptychonn"]
TOTAL_ITERATION = 230

SYSSTAT_FILENAME_FMT = "{0}_{1}_sysstat.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, help="path containing the result directory in approproate format")
    parser.add_argument("--total-iteration", "-titer", type=int, default=TOTAL_ITERATION, help="total iteration the syste is run")
    parser.add_argument("--output-filepath", "-o", type=str, help="output figure path")
        
    args = parser.parse_args()

    # Dictionary to store average values: {param: [sys1_val, sys2_val, ...]}
    results = {net: {param: [] for param in BATCH_LIST} for net in NETWORK_LIST}
    results_time = {net: {param: [] for param in BATCH_LIST} for net in NETWORK_LIST}
    results_energy = {net: {param: [] for param in BATCH_LIST} for net in NETWORK_LIST}
    results_power = {net: {param: [] for param in BATCH_LIST} for net in NETWORK_LIST}
    # Read values and compute means
    for net in NETWORK_LIST:
        for bs in BATCH_LIST:
            sysstat_path = os.path.join(args.dir, SYSSTAT_FILENAME_FMT.format(net, bs))

            power_profile = []
            energy_profile = []
            time_values = []
            with open(sysstat_path, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    # ignore header line
                    if i == 0:
                        continue

                    tokens = line.split(",")
                    # to filter gpu idle moments
                    gpu_util = float(tokens[4])
                    # if gpu_util < 5:
                    #     continue
                    power_profile.append(float(tokens[-2]))
                    energy_profile.append(float(tokens[-1]))
                    time_values.append(float(tokens[1])/1e9)
            # print(net, bs, len(energy_profile))
            results_energy[net][bs].append((energy_profile[-1]-energy_profile[0])/(args.total_iteration*bs))
            results_power[net][bs].append(power_profile)
            results_time[net][bs].append(time_values)


    fig, ax = plt.subplots(figsize=(4, 2.4))
    
    for i, net in enumerate(NETWORK_LIST):
        # for param in DATARATE_LIST:
        #     print(results_power[param][i], results[param][i], train_data_consumed[param][i])
        ax.plot(
            np.array(BATCH_LIST), [results_energy[net][bs][0] for bs in BATCH_LIST],
            label=net
        )

    # ax.set_yticks(np.arange(0, 90, 20))
    ax.set_ylim([0, 500])
    # ax.set_yticklabels(np.arange(0, 90, 20), **YTICK_LABEL_KW)
    ax.set_ylabel("Energy (mJ/req)")
    ax.set_xlabel("Batch Size")
    ax.legend(frameon=False, ncol=1)

    fig.savefig("{0}.png".format(args.output_filepath), dpi=600, bbox_inches="tight")
    fig.savefig("{0}.pdf".format(args.output_filepath), format="pdf", dpi=600, bbox_inches="tight")
