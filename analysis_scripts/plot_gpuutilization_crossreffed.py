import os
import numpy as np
import matplotlib.pyplot as plot
import argparse
import re

import plotprop

FORWARDPASS_TIME_REGEX_DICT = {"pretrained": r"\[(\d+\.\d+)\]\s+INFERENCE BATCH SIZE,(\d+)", "worst_case": r"\[(\d+\.\d+)\]\s+INFERENCE BATCH SIZE,(\d+)",
                               "unipipe": r"\[(\d+\.\d+)\]\s+FORWARD PASS TOOK\(sec\.\),([\d\.]+)", "multicontext": r"\[(\d+\.\d+)\]\s+MISSED INFER FORWARD LATENCY,(\d+)"}
SYSSTAT_TIMECENTISEC_DATALIST_DICT = {}

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--runlog", "-rlog", type=str, help="log file path containing run event log")
    argparser.add_argument("--sysstat-csvlog", "-statcsv", type=str, help="csv file containing sysstat")

    args = argparser.parse_args()

    fig1, ax1 = plot.subplots(figsize=(4, 3))

    label = []
    ticklabel = []

    filepath = args.runlog

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
        accumallow=int(tokens[-2])
        deadline_msec = int((1000/datarate) * int(tokens[-2]))
    else:
        interval_count = int(tokens[-4])
        interval_duration = int(tokens[-3])
        datarate = int(tokens[-2])
        accumallow=int(tokens[-1])
        deadline_msec = int((1000/datarate) * int(tokens[-1]))

    ticklabel.append(system)
    label.append(system)
    cur_label = label[-1]

    timelist = []
    with open(filepath) as fin:
        for line in fin.readlines():
            match = re.search(FORWARDPASS_TIME_REGEX_DICT[system], line)
            if match:
                timestamp = float(match.group(1))     # Extracts the timestamp (e.g., 1727860372.151061)
                timelist.append(int(round(timestamp/10))) # take time stamp at 10^-2sec. precision

    # print(timelist)
    with open(args.sysstat_csvlog) as fin:
        for idx, line in enumerate(fin.readlines()):
            # ignore header line
            if idx == 0:
                continue
            
            tokens = line.split(',')
            # at 10^-1 precision
            # the time is at nanosecond
            timestamp = int(round(float(tokens[1])/1e10))
            if timestamp not in SYSSTAT_TIMECENTISEC_DATALIST_DICT:
                SYSSTAT_TIMECENTISEC_DATALIST_DICT[timestamp] = []

            SYSSTAT_TIMECENTISEC_DATALIST_DICT[timestamp].append(float(tokens[4]))
    
    # print(SYSSTAT_TIMECENTISEC_DATALIST_DICT)

    matched_timeval = []
    gpu_util_mean = []
    for timeval in timelist:
        if timeval in SYSSTAT_TIMECENTISEC_DATALIST_DICT:
            gpu_util_mean.append(
                sum(SYSSTAT_TIMECENTISEC_DATALIST_DICT[timeval])/len(SYSSTAT_TIMECENTISEC_DATALIST_DICT[timeval])
            )
            matched_timeval.append(timeval)
            # print(timeval)

    # positioned to zero
    for i in range(len(matched_timeval)):
        matched_timeval[i] -= timelist[0]

    xticklabels = []
    xticks = []
    variant_idx = 0
    # # print(amperror_data_list, amperror_var_data_list)

    ax1.set_ylabel("GPU Util(%)")
    ax1.set_xlabel("Time (x10 sec.)")
    ax1.plot(matched_timeval, gpu_util_mean, marker='x', label="gpu utilization (%)")
    ax1.legend([plotprop.COMPARED_SYSTEMS_LEGEND_DICT[system] for system in plotprop.COMPARED_SYSTEMS])

    fig1.savefig("figure_gpuutil_curve_{0}_{1}_{2}_{3}.pdf".format(system, interval_duration, datarate, accumallow),
                 format="pdf", bbox_inches='tight')
