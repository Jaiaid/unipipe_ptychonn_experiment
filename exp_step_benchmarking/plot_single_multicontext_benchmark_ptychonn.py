"""
We read benchmark data from two tsv data files

We plot the data in a bar chart for four batch sizes. 
For each batch size, we have two bars for single context and multi context inference time.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np


# Two normal bars + stacked bar (two parts)
# for large mini batch size
# val1 = [0.011, 0.036]
# val2 = [0.019, 0.074]
# val2_stack = [0.046-0.02, 0.112-0.073]

# stack1 = [0.011, 0.039]
# stack2 = [0.027, 0.076]

# err1 = [0.0000478, 0.000386]
# err2 = [0.019, 0.019]

singlecontext_inference_data = []
multicontext_inference_data = []
multicontext_inference_invokegap_data = []

singlecontext_forwardpass_data = []
singlecontext_backwardpass_data = []

singlecontext_inference_deviation_data = []
multicontext_inference_deviation_data = []
singlecontext_forwardbackward_deviation_data = []

BS_LIST = [1, 2, 4, 8, 16, 32, 64]


BAR_WIDTH = 0.25


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-istsv1", "--input-singlecontext-tsvpath1", type=str, required=True, help="input tsv filepath with header line containing the data for single context run network1")
    arg_parser.add_argument("-imtsv1", "--input-multicontext-tsvpath1", type=str, required=True, help="input tsv filepath with header line containing the data for multi context run network1")
    arg_parser.add_argument("-o", "--output-plot-filepath", type=str, required=True, help="output pdf diagram filepath")

    args = arg_parser.parse_args()

    # read data from tsv
    # single context and multicontext file header sequences are different
    with open(args.input_singlecontext_tsvpath1) as fin:
        for i, line in enumerate(fin.readlines()):
            # for header line
            if i == 0:
                continue

            tokens = line.split()
            singlecontext_inference_data.append(float(tokens[1]))
            singlecontext_forwardpass_data.append(float(tokens[5]))
            singlecontext_backwardpass_data.append(float(tokens[9]))
            singlecontext_inference_deviation_data.append(float(tokens[3]))
            singlecontext_forwardbackward_deviation_data.append(float(tokens[10])+float(tokens[7]))

    # multicontext has different sequence of data
    with open(args.input_multicontext_tsvpath1) as fin:
        for i, line in enumerate(fin.readlines()):
            # for header line
            if i == 0:
                continue

            tokens = line.split()
            multicontext_inference_data.append(float(tokens[11]))
            multicontext_inference_deviation_data.append(float(tokens[3]))
            multicontext_inference_invokegap_data.append(float(tokens[11])-float(tokens[1]))
                

    fig, ax = plt.subplots(figsize=(4,2.4))
    x = np.arange(len(BS_LIST))

    values = [singlecontext_inference_data[i+1]*1000 for i in range(len(BS_LIST))]
    errors = [singlecontext_inference_deviation_data[i+1]*1000 for i in range(len(BS_LIST))]
    ax.bar(
        x - BAR_WIDTH, values, BAR_WIDTH,
        yerr=[[0]*len(BS_LIST), errors], hatch="o", label='Single-Context', capsize=2.5
    )
    print("single context inference data:", values)

    values = [(multicontext_inference_data[i+1])*1000 for i in range(len(BS_LIST))]
    errors = [multicontext_inference_deviation_data[i+1]*1000 for i in range(len(BS_LIST))]
    ax.bar(
        x, values, BAR_WIDTH,
        yerr=[[0]*len(BS_LIST), errors], hatch="x", label='Multi-Context', capsize=2.5
    )
    print("multi context inference data:", values)

    ax.set_xticks(x)
    ax.set_xticklabels(BS_LIST, **{"fontsize": 10, "fontweight": "bold" })
    ax.set_ylabel('Latency (ms)', **{"fontsize": 10, "fontweight": "bold"})
    # ax.set_ylim([0, 25])
    ax.set_yticks(np.arange(0, 30, 5))
    ax.set_yticklabels(np.arange(0, 30, 5), **{"fontsize": 10, "fontweight": "bold"})
    ax.legend(prop={"size": 9, "weight": "bold"}, frameon=False, loc="upper left", ncol=2)
    ax.set_xlabel("Batch Size", **{"fontsize": 10, "fontweight": "bold"})

    fig.savefig(args.output_plot_filepath, format="pdf", dpi=600, bbox_inches="tight")
    fig.savefig(args.output_plot_filepath[:-4]+".png", dpi=600, bbox_inches="tight")
