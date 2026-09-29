import os
import numpy as np
import matplotlib.pyplot as plot
import argparse
import re

import plotprop

# compared system
SYSTEM_TO_COLORMARKER_DICT = {"pretrained": ("tab:orange", "/"), "worst_case": ("r", "\\"), "multicontext": ("g", "o"), "unipipe": ("b", "+")}

# parameter describing configurations
DUR_LIST = [86, 172]
DATARATE_LIST = [50, 100]
REQALLOW_LIST = [100, 500]


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dir", "-dir", type=str, help="root path containing folder for each system with each configuration sysstat")

    args = argparser.parse_args()

    fig1, ax1 = plot.subplots(figsize=(4, 3))

    label = []
    ticklabel = []
    data_dict = {}

    rootpath = args.dir

    for system_idx, system in enumerate(plotprop.COMPARED_SYSTEMS):
        # to handle some log file naming detail
        peak_usage_list = [0]
        for dur in DUR_LIST:
            for datarate in DATARATE_LIST:
                for accumallow in REQALLOW_LIST:

                    basename = "{0}_sysstat_3_{1}_{2}_{3}.csv".format(system, dur, datarate, accumallow) 
                    csv_filepath = os.path.join((os.path.join(rootpath, system)), basename)

                    max_usage = 0
                    try:
                        with open(csv_filepath) as fin:
                            for idx, line in enumerate(fin.readlines()):
                                # ignore header line
                                if idx == 0:
                                    continue

                                tokens = line.split(',')
                                usage = float(tokens[4])
                                if usage > max_usage:
                                    max_usage = usage
                    except FileNotFoundError:
                        print("{0} not found, putting max usage to 0".format(csv_filepath))

                    peak_usage_list.append(max_usage)

        ax1.bar(
            x=np.arange(1, len(peak_usage_list)*2-1, 2) + system_idx*0.3 - 0.45,
            height=peak_usage_list[1:],width=0.3, label=system,
            hatch=SYSTEM_TO_COLORMARKER_DICT[system][1], color=SYSTEM_TO_COLORMARKER_DICT[system][0])

    ax1.set_xticks(np.arange(1, len(peak_usage_list)*2-1, 2))
    ax1.set_xticklabels([str(i) for i in range(1, len(peak_usage_list))])
    ax1.set_ylabel("Peak GPU Util(%)")
    ax1.set_ylim([0, 130])
    ax1.legend([plotprop.COMPARED_SYSTEMS_LEGEND_DICT[system] for system in plotprop.COMPARED_SYSTEMS], ncol=2)

    fig1.savefig("figure_gpuutil_summary.pdf", format="pdf", bbox_inches='tight')
