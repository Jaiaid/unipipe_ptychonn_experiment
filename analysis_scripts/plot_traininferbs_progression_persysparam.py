import os
import numpy as np
import matplotlib.pyplot as plot
import argparse
import re


COMPARED_SYSTEMS = ["worst_case", "multicontext", "unipipe"]
COMPARED_SYSTEM_REGEX_DICT = {"worst_case": r"\[(\d+\.\d+)\]\s+INFERENCE BATCH SIZE,(\d+)", "multicontext": r"\[(\d+\.\d+)\]\s+INFERENCE BATCH SIZE,(\d+)", "unipipe": r"\[(\d+\.\d+)\]\s+UNIPIPE TRAIN,\s*INFER BS,(\d+),(\d+)"}
SYSTEM_TO_COLORMARKER_DICT = {"worst_case": ("r", "+"), "multicontext": ("g", "o"), "unipipe": ("b", "x")}


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--log", "-log", type=str, help="log file path containing log")

    args = argparser.parse_args()

    fig1, ax1 = plot.subplots(figsize=(20, 5))

    label = []
    ticklabel = []

    filepath = args.log

    basename = filepath.split("/")[-1]
    tokens = basename.split(".")
    tokens = tokens[0].split("_")
    system = tokens[0]
    # to handle some log file naming detail
    if system == "worst":
        system = "worst_case"

    if system == "multicontext":
        interval_count = int(tokens[-5])
        interval_duration = int(tokens[-4])
        datarate = int(tokens[-3])
        deadline_msec = int((1000/datarate) * int(tokens[-2]))
    else:
        interval_count = int(tokens[-4])
        interval_duration = int(tokens[-3])
        datarate = int(tokens[-2])
        deadline_msec = int((1000/datarate) * int(tokens[-1]))

    ticklabel.append(system)
    label.append(system)
    cur_label = label[-1]

    inference_iteration_gap_timelist = []
    inference_times = []
    with open(filepath) as fin:
        pattern = COMPARED_SYSTEM_REGEX_DICT[system]
        for line in fin.readlines():
            match = re.search(pattern, line)
            if match:
                # print(line)
                timestamp = float(match.group(1))     # Extracts the timestamp (e.g., 1727860372.151061)
                if system == "unipipe":
                    batch_size = int(match.group(3))
                else:
                    batch_size = int(match.group(2))
                inference_iteration_gap_timelist.append(batch_size)
                inference_times.append(timestamp)

    zero_time = inference_times[0]
    for i in range(len(inference_times)):
        inference_times[i] = inference_times[i] - zero_time

    xticklabels = []
    xticks = []
    variant_idx = 0
    # # print(amperror_data_list, amperror_var_data_list)

    ax1.set_ylabel("Batch Size")
    ax1.set_xlabel("")
    ax1.plot(inference_times, inference_iteration_gap_timelist, marker='x')
    # ax1.set_xticks(xticks)
    # ax1.set_xticklabels(xticklabels, size=9, rotation=90)
    ax1.legend([system])
    # ax1.set_ylim([0, 10])
    fig1.savefig("inferbs_progression_{0}_{1}_{2}_{3}.png".format(system, interval_duration, datarate, deadline_msec), bbox_inches='tight')
