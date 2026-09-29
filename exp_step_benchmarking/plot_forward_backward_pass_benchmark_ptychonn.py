import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

network_name_to_legend_dict = {"ptychonn": 'PtychONN'}
network_name_list = ["ptychonn"]

forward_pass_data_dict = {}
backward_pass_data_dict = {}

FILE_NAME_FMT = "benchmark_{0}_nn_step.csv"


if __name__ == "__main__":
    # read data from tsv
    # single context and multicontext file header sequences are different
    for network_name in network_name_list:
        filepath = FILE_NAME_FMT.format(network_name)
        forward_pass_data_dict[network_name] = []
        backward_pass_data_dict[network_name] = []

        with open(filepath) as fin:
            for i, line in enumerate(fin.readlines()):
                # header line 
                if i == 0:
                    continue

                # read the data only from batch size line
                tokens = line.split()
                forward_pass_data_dict[network_name].append(float(tokens[5])*1000)
                backward_pass_data_dict[network_name].append(float(tokens[9])*1000)

    # forward pass graph
    fig, ax = plt.subplots(figsize=(4,2.4))
    for network_name in network_name_list:
        ax.plot(forward_pass_data_dict[network_name], label="forward pass")
        ax.plot(backward_pass_data_dict[network_name], label="backward pass")

    ax.set_ylabel('Latency (ms)', **{"fontsize": 10, "fontweight": "bold"})
    ax.set_xlabel("Batch Size", **{"fontsize": 10, "fontweight": "bold"})
    # ax.set_ylim([0, 0.25])
    ax.legend(prop={"size": 9, "weight": "bold"}, frameon=False, loc="upper left")
    ax.tick_params(axis='both', labelsize=10)
    plt.setp(ax.get_xticklabels(), fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")

    fig.savefig("fig_network_forward_backward_pass_benchmark_ptychonn.pdf", format="pdf", dpi=600, bbox_inches="tight")
    fig.savefig("fig_network_forward_backward_pass_benchmark_ptychonn.png", dpi=600, bbox_inches="tight")
