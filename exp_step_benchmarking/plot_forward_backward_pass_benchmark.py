import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

network_name_to_legend_dict = {"resnet18": 'ResNet-18', "resnet50": 'ResNet-50', "resnet101": 'ResNet-101', "mobilenet_v2": 'MobileNet-v2'}
network_name_list = ["resnet18", "resnet50", "resnet101", "mobilenet_v2"]

forward_pass_data_dict = {}
backward_pass_data_dict = {}

FILE_NAME_FMT = "benchmark_{0}_nn_step.csv"


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-idir", "--input-dir", type=str, required=True, help="input directory path containing tsv filepath with specific name format")

    args = arg_parser.parse_args()

    # read data from tsv
    # single context and multicontext file header sequences are different
    for network_name in network_name_list:
        filepath = os.path.join(args.input_dir, FILE_NAME_FMT.format(network_name))
        forward_pass_data_dict[network_name] = []
        backward_pass_data_dict[network_name] = []

        with open(filepath) as fin:
            for i, line in enumerate(fin.readlines()):
                # header line 
                if i == 0:
                    continue

                # read the data only from batch size line
                tokens = line.split()
                forward_pass_data_dict[network_name].append(float(tokens[5]))
                backward_pass_data_dict[network_name].append(float(tokens[9]))

    # forward pass graph
    fig, ax = plt.subplots(figsize=(4,2.4))
    for network_name in network_name_list:
        ax.plot(forward_pass_data_dict[network_name], label=network_name_to_legend_dict[network_name])

    ax.set_ylabel('Execution Time (s)', **{"fontsize": 10, "fontweight": "bold"})
    ax.set_xlabel("Batch Size", **{"fontsize": 10, "fontweight": "bold"})
    ax.set_ylim([0, 0.25])
    ax.legend(prop={"size": 9, "weight": "bold"}, frameon=False, loc="upper left")
    ax.tick_params(axis='both', labelsize=10)
    plt.setp(ax.get_xticklabels(), fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")

    fig.savefig("fig_network_forward_pass_benchmark.pdf", format="pdf", dpi=600, bbox_inches="tight")
    fig.savefig("fig_network_forward_pass_benchmark.png", dpi=600, bbox_inches="tight")

    # backward pass graph
    fig, ax = plt.subplots(figsize=(4,2.4))
    for network_name in network_name_list:
        ax.plot(backward_pass_data_dict[network_name], label=network_name_to_legend_dict[network_name])

    ax.set_ylabel('Execution Time (s)', **{"fontsize": 10, "fontweight": "bold"})
    ax.set_xlabel("Batch Size", **{"fontsize": 10, "fontweight": "bold"})
    ax.tick_params(axis='both', labelsize=10)
    ax.legend(prop={"size": 9, "weight": "bold"}, frameon=False, loc="upper left")
    ax.set_ylim([0, 0.40])
    plt.setp(ax.get_xticklabels(), fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")

    fig.savefig("fig_network_backward_pass_benchmark.pdf", format="pdf", dpi=600, bbox_inches="tight")
    fig.savefig("fig_network_backward_pass_benchmark.png", dpi=600, bbox_inches="tight")
