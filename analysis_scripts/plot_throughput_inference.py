import os
import numpy as np
import matplotlib.pyplot as plot
import argparse
import re

import plotprop

INTERVAL_START_REGEX_DICT = {"pretrained": r"\[(\d+\.\d+)\]\s+INTERVAL START (\d+)", "worst_case": r"\[(\d+\.\d+)\]\s+INTERVAL START (\d+)", "multicontext": r"\[(\d+\.\d+)\]\s+MULTICONTEXT INFER DATASTREAM START", "unipipe": r"\[(\d+\.\d+)\]\s+INTERVAL START (\d+)"}
COMPARED_SYSTEM_REGEX_DICT = {"pretrained": r"\[(\d+\.\d+)\]\s+INFERENCE BATCH SIZE,(\d+)", "worst_case": r"\[(\d+\.\d+)\]\s+INFERENCE BATCH SIZE,(\d+)", "multicontext": r"\[(\d+\.\d+)\]\s+INFERENCE BATCH SIZE,(\d+)", "unipipe": r"\[(\d+\.\d+)\]\s+UNIPIPE TRAIN,\s*INFER BS,(\d+),(\d+)"}
SYSTEM_TO_COLORMARKER_DICT = {"pretrained": ("tab:orange", "*"), "worst_case": ("r", "+"), "multicontext": ("g", "o"), "unipipe": ("b", "x"), "unipipe_gd": ("y", "*")}


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

        data_dict[system][interval_duration] = {}
        data_dict[system][interval_duration][datarate] = {}
        data_dict[system][interval_duration][datarate][args.accumallow] = []

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

        inference_iteration_gap_timelist = []
        with open(filepath) as fin:
            interval_no = 0
            for line in fin.readlines():
                match = re.search(INTERVAL_START_REGEX_DICT[system], line)
                if match:
                    interval_no += 1
                    # take the timestamp
                    interval_time_list.append(float(match.group(1)))

                if interval_no < 2:
                    continue

                match = re.search(COMPARED_SYSTEM_REGEX_DICT[system], line)
                if match:
                    timestamp = float(match.group(1))     # Extracts the timestamp (e.g., 1727860372.151061)
                    if system == "unipipe":
                        batch_size = int(match.group(3))
                    else:
                        batch_size = int(match.group(2))

                    data_dict[system][interval_duration][datarate][args.accumallow].append(batch_size)
                    inference_iteration_gap_timelist.append(timestamp)

            throughput_ara = []
            for i in range(1, len(inference_iteration_gap_timelist)):
                throughput_ara.append(
                    data_dict[system][interval_duration][datarate][args.accumallow][i]/\
                    (inference_iteration_gap_timelist[i] - inference_iteration_gap_timelist[i-1])
                )
            for i in range(1, len(inference_iteration_gap_timelist)):
                inference_iteration_gap_timelist[i] = inference_iteration_gap_timelist[i] - interval_time_list[1]
    
            ax1.plot(
                inference_iteration_gap_timelist[1:],
                throughput_ara,
                label=system, marker=SYSTEM_TO_COLORMARKER_DICT[system][1],
                color=SYSTEM_TO_COLORMARKER_DICT[system][0]
            )
            # print(system, interval_duration, datarate, deadline_msec)
            # break

    # # print(amperror_data_list, amperror_var_data_list)

    ax1.set_ylabel("Infer. Throughput(image/sec.)")
    ax1.set_xlabel("Time (sec.)")
    ax1.legend([plotprop.COMPARED_SYSTEMS_LEGEND_DICT[system] for system in plotprop.COMPARED_SYSTEMS])
    # ax1.set_xlabel("")
    # ax1.set_xticks(xticks)
    # ax1.set_xticklabels(xticklabels, size=9, rotation=90)
    # ax1.legend(COMPARED_SYSTEMS)
    # ax1.set_ylim([0, 10])
    fig1.savefig("figure_inference_throughput_{0}_{1}_{2}.pdf".format(args.duration, args.datarate, args.accumallow),
                 format="pdf", bbox_inches='tight')

