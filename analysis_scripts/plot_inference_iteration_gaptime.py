import os
import numpy as np
import matplotlib.pyplot as plot
import argparse
import re

import plotprop

COMPARED_SYSTEM_REGEX_DICT = {"pretrained": r"\[(\d+\.\d+)\]\s+INFERENCE BATCH SIZE,(\d+)", "worst_case": r"\[(\d+\.\d+)\]\s+INFERENCE BATCH SIZE,(\d+)", "multicontext": r"\[(\d+\.\d+)\]\s+INFERENCE BATCH SIZE,(\d+)", "unipipe": r"\[(\d+\.\d+)\]\s+UNIPIPE TRAIN,\s*INFER BS,(\d+),(\d+)"}
INTERVAL_START_REGEX_DICT = {"pretrained": r"\[(\d+\.\d+)\]\s+INTERVAL START (\d+)", "worst_case": r"\[(\d+\.\d+)\]\s+INTERVAL START (\d+)", "multicontext": r"\[(\d+\.\d+)\]\s+MULTICONTEXT INFER DATASTREAM START", "unipipe": r"\[(\d+\.\d+)\]\s+INTERVAL START (\d+)"}
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
    global_data_ara = []

    fig1, ax1 = plot.subplots(figsize=(4, 3))

    for system_idx, system in enumerate(plotprop.COMPARED_SYSTEMS):
        data_dict[system] = {}
        interval_time_list = []
        data_dirpath = os.path.join(args.dir, system)
        
        label = []
        ticklabel = []

        interval_count = 3
        interval_duration = args.duration
        datarate = args.datarate
        accumallow = args.accumallow

        data_dict[system][interval_duration] = {}
        data_dict[system][interval_duration][datarate] = {}
        data_dict[system][interval_duration][datarate][accumallow] = []

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
        data_dict[system][interval_duration] = {}
        if datarate not in data_dict[system][interval_duration]:
            data_dict[system][interval_duration][datarate] = {}
        # to store each interval data separately
        if accumallow not in data_dict[system][interval_duration][datarate]:
            data_dict[system][interval_duration][datarate][accumallow] = []

        ticklabel.append(system)
        label.append(system)
        cur_label = label[-1]

        inference_iteration_gap_timelist = []
        interval_start_timelist = []
        batch_sizelist = []
        with open(filepath) as fin:
            for line in fin.readlines():
                # print(line)
                pattern = INTERVAL_START_REGEX_DICT[system]
                match = re.search(pattern, line)
                if match:
                    timestamp = float(match.group(1))     # Extracts the timestamp (e.g., 1727860372.151061)
                    interval_start_timelist.append([timestamp])
                    inference_iteration_gap_timelist.append([])
                    batch_sizelist.append([])

                pattern = COMPARED_SYSTEM_REGEX_DICT[system]
                match = re.search(pattern, line)
                if match:
                    timestamp = float(match.group(1))     # Extracts the timestamp (e.g., 1727860372.151061)
                    batch_size = int(match.group(2))
                    inference_iteration_gap_timelist[-1].append(timestamp)
                    batch_sizelist[-1].append(batch_size)

        # print(system, interval_duration, datarate, accumallow)
        # print(inference_iteration_gap_timelist)
        # preprocess to make it difference rather than abosolute value
        data_ara = []
        for i in range(len(inference_iteration_gap_timelist)):
            for j in range(1, len(inference_iteration_gap_timelist[i])):
                data_ara.append((inference_iteration_gap_timelist[i][j] - inference_iteration_gap_timelist[i][j-1])/batch_sizelist[i][j-1])
        # print(data_ara)
        
        if len(data_ara) < 1:
            data_ara = [interval_duration]
            print(system, interval_duration, datarate, accumallow)
        global_data_ara.append(data_ara)

    ax1.boxplot(global_data_ara, notch=True)

    ax1.set_ylabel("Iteration Timegap (sec.)")
    ax1.set_xlabel("")
    # to remove outlier
    ax1.set_ylim([0, 0.1])
    ax1.set_xticklabels([plotprop.COMPARED_SYSTEMS_LEGEND_DICT[system] for system in plotprop.COMPARED_SYSTEMS])
    fig1.savefig("figure_infergap_{0}_{1}_{2}.pdf".format(args.duration, args.datarate, args.accumallow),
                 format="pdf", bbox_inches='tight')
