import os
import numpy as np
import matplotlib.pyplot as plot
import argparse
import re

import plotprop

COMPARED_SYSTEM_PHERROR_REGEX_DICT = {"pretrained": r"\[(\d+\.\d+)\]\s+PH\. ERROR,([\d\.,]+)", "worst_case": r"\[(\d+\.\d+)\]\s+PH\. ERROR,([\d\.,]+)",
                              "multicontext": r"\[(\d+\.\d+)\]\s+PH\. ERROR,([\d\.,]+)", "unipipe": r"\[(\d+\.\d+)\]\s+PH\. ERROR,([\d\.,]+)"}
COMPARED_SYSTEM_MISSRATE_REGEX_DICT = {"pretrained": r"\[\d+\.\d+\]\s+MISSRATE,\[([\d\.\s,]+)\]", "worst_case": r"\[\d+\.\d+\]\s+MISSRATE,\[([\d\.\s,]+)\]",
                              "multicontext": r"\[\d+\.\d+\]\s+MISSRATE,\[([\d\.\s,]+)\]", "unipipe": r"\[\d+\.\d+\]\s+MISSRATE,\[([\d\.\s,]+)\]"}

SYSTEM_TO_COLORMARKER_DICT = {"pretrained": ("tab:orange", "*"), "worst_case": ("r", "+"), "multicontext": ("g", "o"), "unipipe": ("b", "x")}


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dir", "-d", type=str, help="directory containing subfolder named by system which contains log files")
    argparser.add_argument("--duration", "-dur", type=int, help="which interval duration to create the throughput for")
    argparser.add_argument("--accumallow", "-accum", type=int, help="deadline determining parameter")
    argparser.add_argument("--datarate", "-rate", type=int, help="data rate (per sec.)")

    args = argparser.parse_args()

    # data dictionary to collect all data
    data_dict = {}

    fig1, ax1 = plot.subplots(figsize=(4, 3))

    for system_idx, system in enumerate(plotprop.COMPARED_SYSTEMS):
        data_dict[system] = {}
        interval_time_list = []
        data_dirpath = os.path.join(args.dir, system)
        
        label = []
        ticklabel = []

        interval_count = 5
        interval_duration = args.duration
        datarate = args.datarate
        accumallow = args.accumallow

        if system == "multicontext":
            filepath = os.path.join(
                data_dirpath, 
                "{0}_{1}_{2}_{3}_{4}_infer.log".format(
                    system, interval_count, interval_duration, datarate, args.accumallow)
            )
        else:
            filepath = os.path.join(
                data_dirpath, 
                "{0}_{1}_{2}_{3}_{4}.log".format(
                    system, interval_count, interval_duration, datarate, args.accumallow)
            )

        ticklabel.append(system)
        label.append(system)
        cur_label = label[-1]

        with open(filepath) as fin:
            for line in fin.readlines():
                pattern = COMPARED_SYSTEM_PHERROR_REGEX_DICT[system]
                match = re.search(pattern, line)
                if match:
                    timestamp = match.group(1)  # Extract the timestamp
                    numbers = match.group(2)    # Extract the numbers as a string
                    pherror_list = [float(num) for num in numbers.split(',')]  # C

                pattern = COMPARED_SYSTEM_MISSRATE_REGEX_DICT[system]
                match = re.search(pattern, line)
                if match:
                    numbers_str = match.group(1)  # Extract the numbers as a single string
                    missrate_list = [float(num) for num in numbers_str.split(',')] 

        data_dict[system] = [pherror_list, missrate_list]

    # for first interval data
    ax1.bar(x = np.arange(len(plotprop.COMPARED_SYSTEMS)) - 0.125, height=[data_dict[system][0][0] for system in plotprop.COMPARED_SYSTEMS], width=0.25, label="Interval 1", hatch="/")
    ax1.bar(x = np.arange(len(plotprop.COMPARED_SYSTEMS)) + 0.125, height=[data_dict[system][0][3] for system in plotprop.COMPARED_SYSTEMS], width=0.25, label="Interval 4", hatch="o")

    ax1.set_ylabel("MSE Error")
    ax1.set_xlabel("")
    ax1.set_xticklabels([""] + plotprop.COMPARED_SYSTEMS)

    # for miss rate
    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(plotprop.COMPARED_SYSTEMS)) - 0.125, [data_dict[system][1][0]*100 for system in plotprop.COMPARED_SYSTEMS], label="Interval 1", marker="x", mec="k", mfc="k")
    ax2.plot(np.arange(len(plotprop.COMPARED_SYSTEMS)) + 0.125, [data_dict[system][1][3]*100 for system in plotprop.COMPARED_SYSTEMS], label="Interval 4", marker="+", mec="k", mfc="k")
    ax2.set_ylabel("Miss Rate (%)")

    ax1.legend([plotprop.COMPARED_SYSTEMS_LEGEND_DICT[system] for system in plotprop.COMPARED_SYSTEMS])

    fig1.savefig("figure_inferquality_{0}_{1}_{2}.pdf".format(args.duration, args.datarate, args.accumallow),
                 format="pdf", bbox_inches='tight')
