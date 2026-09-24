import argparse
import matplotlib.pyplot as plt
import numpy as np

groups = ['ResNet-18', 'ResNet-50'] #, 'ResNet-101', 'MobileNet-v2']
x = np.arange(len(groups))

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

BAR_WIDTH = 0.25


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-istsv1", "--input-singlecontext-tsvpath1", type=str, required=True, help="input tsv filepath with header line containing the data for single context run network1")
    arg_parser.add_argument("-imtsv1", "--input-multicontext-tsvpath1", type=str, required=True, help="input tsv filepath with header line containing the data for multi context run network1")
    arg_parser.add_argument("-istsv2", "--input-singlecontext-tsvpath2", type=str, required=True, help="input tsv filepath with header line containing the data for single context run network2")
    arg_parser.add_argument("-imtsv2", "--input-multicontext-tsvpath2", type=str, required=True, help="input tsv filepath with header line containing the data for multi context run network2")
    arg_parser.add_argument("-bs", "--batch-size", type=int, required=True, help="for which batch size the diagram will be drawn")
    arg_parser.add_argument("-o", "--output-plot-filepath", type=str, required=True, help="output pdf diagram filepath")

    args = arg_parser.parse_args()

    # read data from tsv
    # single context and multicontext file header sequences are different
    with open(args.input_singlecontext_tsvpath1) as fin:
        for i, line in enumerate(fin.readlines()):
            if i == 0:
                continue

            # read the data only from batch size line
            if i == args.batch_size:
                tokens = line.split()
                singlecontext_inference_data.append(float(tokens[1]))
                singlecontext_forwardpass_data.append(float(tokens[5]))
                singlecontext_backwardpass_data.append(float(tokens[9]))
                singlecontext_inference_deviation_data.append(float(tokens[3]))
                singlecontext_forwardbackward_deviation_data.append(float(tokens[10])+float(tokens[7]))
                break

    with open(args.input_singlecontext_tsvpath2) as fin:
        for i, line in enumerate(fin.readlines()):
            if i == 0:
                continue

            # read the data only from batch size line
            if i == args.batch_size:
                tokens = line.split()
                singlecontext_inference_data.append(float(tokens[1]))
                singlecontext_forwardpass_data.append(float(tokens[5]))
                singlecontext_backwardpass_data.append(float(tokens[9]))
                singlecontext_inference_deviation_data.append(float(tokens[3]))
                singlecontext_forwardbackward_deviation_data.append(float(tokens[10])+float(tokens[7]))
                break

    # multicontext has different sequence of data
    with open(args.input_multicontext_tsvpath1) as fin:
        for i, line in enumerate(fin.readlines()):
            if i == 0:
                continue

            # read the data only from batch size line
            if i == args.batch_size:
                tokens = line.split()
                multicontext_inference_data.append(float(tokens[1]))
                multicontext_inference_deviation_data.append(float(tokens[3]))
                multicontext_inference_invokegap_data.append(float(tokens[11])-float(tokens[1]))
                break
    

    # multicontext has different sequence of data
    with open(args.input_multicontext_tsvpath2) as fin:
        for i, line in enumerate(fin.readlines()):
            if i == 0:
                continue

            # read the data only from batch size line
            if i == args.batch_size:
                tokens = line.split()
                multicontext_inference_data.append(float(tokens[1]))
                multicontext_inference_deviation_data.append(float(tokens[3]))
                multicontext_inference_invokegap_data.append(float(tokens[11])-float(tokens[1]))
                break


    fig, ax = plt.subplots(figsize=(4,2.4))
    ax.bar(
        x - BAR_WIDTH, singlecontext_inference_data, BAR_WIDTH,
        yerr=singlecontext_inference_deviation_data, hatch="o", label='Single-Context Inference', capsize=2.5
    )
    ax.bar(
        x, multicontext_inference_data, BAR_WIDTH,
        hatch="/", label='Multi-Context Inference'
    )

    ax.bar(
        x, multicontext_inference_invokegap_data, BAR_WIDTH, bottom=multicontext_inference_data,
        yerr=multicontext_inference_deviation_data, hatch="x", label='Multi-Context Invoke Gap', capsize=2.5
    )

    ax.bar(
        x + BAR_WIDTH, singlecontext_forwardpass_data, BAR_WIDTH, hatch="\\",
        label='Single-Context Forward'
    )

    ax.bar(
        x + BAR_WIDTH, singlecontext_backwardpass_data, BAR_WIDTH,
        yerr=singlecontext_forwardbackward_deviation_data, hatch="+",
        bottom=singlecontext_forwardpass_data, label='Single-Context Backward', capsize=2.5
    )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, **{"fontsize": 10, "fontweight": "bold" })
    ax.set_ylabel('Execution Time (s)', **{"fontsize": 10, "fontweight": "bold"})
    ax.set_ylim([0, 0.25])
    ax.set_yticks(np.arange(0, 0.25+0.01, 0.05))
    ax.set_yticklabels([f"{x:.2f}" for x in np.arange(0, 0.25+0.01, 0.05)], **{"fontsize": 10, "fontweight": "bold"})
    ax.legend(prop={"size": 9, "weight": "bold"}, frameon=False, loc="upper left")

    fig.savefig(args.output_plot_filepath, format="pdf", dpi=600, bbox_inches="tight")
    fig.savefig(args.output_plot_filepath+".png", dpi=600, bbox_inches="tight")
