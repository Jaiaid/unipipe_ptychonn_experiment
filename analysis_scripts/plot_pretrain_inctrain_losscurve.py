import os
import numpy as np
import matplotlib.pyplot as plot
import argparse
import re

PRETRAIN_EPOCH = 10
TRAINLOSS_REGEX = r"\[(\d+\.\d+)\]\s+TRAINING LOSS AT EPOCH,(\d+),([\d\.]+),([\d\.]+),([\d\.]+)"
INTERVAL_START_REGEX = r"\[(\d+\.\d+)\]\s+INTERVAL START (\d+)"
                    

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

    trainloss_timelist = []
    interval_start_timelist = []
    loss_val = []
    amploss_val = []
    phloss_val = []
    with open(filepath) as fin:
        for line in fin.readlines():
            match = re.search(TRAINLOSS_REGEX, line)
            if match:
                timestamp = float(match.group(1))     # Extracts the timestamp (e.g., 1727860372.151061)
                epoch = int(match.group(2))
                totalloss = float(match.group(3))
                amploss = float(match.group(4))
                phloss = float(match.group(5))

                trainloss_timelist.append(timestamp)
                loss_val.append(totalloss)
                amploss_val.append(amploss)
                phloss_val.append(phloss)

            match = re.search(INTERVAL_START_REGEX, line)
            if match:
                timestamp = float(match.group(1))     # Extracts the timestamp (e.g., 1727860372.151061)
                interval_no = int(match.group(2))

                interval_start_timelist.append(timestamp)

    zero_time = interval_start_timelist[1]
    for i in range(len(trainloss_timelist)):
        trainloss_timelist[i] = trainloss_timelist[i] - zero_time
    for i in range(len(interval_start_timelist)):
        interval_start_timelist[i] -= zero_time

    xticklabels = []
    xticks = []
    variant_idx = 0
    # # print(amperror_data_list, amperror_var_data_list)

    ax1.set_ylabel("Training Loss")
    ax1.set_xlabel("time(s)")
    ax1.plot(trainloss_timelist, loss_val, marker='o', label="total loss")
    # ax1.plot(trainloss_timelist[PRETRAIN_EPOCH:], amploss_val[PRETRAIN_EPOCH:], marker='x', label="amp. loss")
    # ax1.plot(trainloss_timelist[PRETRAIN_EPOCH:], phloss_val[PRETRAIN_EPOCH:], marker='+', label="ph loss")
    for i in range(1, len(interval_start_timelist)):
        ax1.axvline(interval_start_timelist[i], label='interval boundary')
    # ax1.set_xticks(xticks)
    # ax1.set_xticklabels(xticklabels, size=9, rotation=90)
    ax1.legend()
    # ax1.set_ylim([0, 10])
    fig1.savefig("figure_trainloss_curve_{0}_{1}_{2}_{3}.png".format(system, interval_duration, datarate, deadline_msec), bbox_inches='tight')
