import os
import numpy as np
import matplotlib.pyplot as plot
import argparse
import re

TRAINLOSS_REGEX = r"\[(\d+\.\d+)\]\s+ITER_COUNT, TRAIN LOSS,(\d+),([\d.eE+-]+),([\d.eE+-]+),([\d.eE+-]+)"
INTERVAL_START_REGEX = r"\[(\d+\.\d+)\]\s+INTERVAL START (\d+)"
TRAIN_START_REGEX = r"\[(\d+\.\d+)\]\s+MULTICONTEXT TRAIN BEGIN" 

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--ulog", "-ulog", type=str, help="log file path containing unipipe log")
    argparser.add_argument("--mlog", "-mlog", type=str, help="log file path containing multicontext log")

    args = argparser.parse_args()

    fig1, ax1 = plot.subplots(figsize=(20, 5))

    label = []
    ticklabel = []

    for filepath in [args.ulog, args.mlog]:
        # find out system name and others
        filebname = os.path.basename(filepath)
        tokens = filebname.split("_")
        system = tokens[0]

        timelist = []
        interval_start_timelist = []
        train_block_start_timelist = []
        loss_val = []
        amploss_val = []
        phloss_val = []
        trainloss_val = []
        trainamploss_val = []
        trainphloss_val = []
        with open(filepath) as fin:
            for line in fin.readlines():
                match = re.search(TRAINLOSS_REGEX, line)
                if match:
                    timestamp = float(match.group(1))     # Extracts the timestamp (e.g., 1727860372.151061)
                    totalloss = float(match.group(3))
                    amploss = float(match.group(4))
                    phloss = float(match.group(5))

                    trainloss_val.append(totalloss)
                    trainamploss_val.append(amploss)
                    trainphloss_val.append(phloss)
                    timelist.append(timestamp)

                match = re.search(INTERVAL_START_REGEX, line)
                if match:
                    timestamp = float(match.group(1))     # Extracts the timestamp (e.g., 1727860372.151061)
                    interval_start_timelist.append(timestamp)

                match = re.search(TRAIN_START_REGEX, line)
                if match:
                    timestamp = float(match.group(1))     # Extracts the timestamp (e.g., 1727860372.151061)
                    train_block_start_timelist.append(timestamp)

        zero_time = interval_start_timelist[0]
        for i in range(len(timelist)):
            timelist[i] = timelist[i] - zero_time
        for i in range(len(train_block_start_timelist)):
            train_block_start_timelist[i] -= zero_time
        for i in range(len(interval_start_timelist)):
            interval_start_timelist[i] -= zero_time
        xticklabels = []
        xticks = []
        variant_idx = 0
        # # print(amperror_data_list, amperror_var_data_list)

        ax1.set_ylabel("Loss")
        ax1.set_xlabel("Time (s)")
        print(interval_start_timelist)
        print(min(trainloss_val))
        ax1.plot(timelist, trainloss_val, marker='o', label=system)
        for i, t in enumerate(train_block_start_timelist):
            ax1.axvline(t)
    
    # ax1.set_xticks(xticks)
    # ax1.set_xticklabels(xticklabels, size=9, rotation=90)
    ax1.legend()
    # ax1.set_ylim([0, 10])
    fig1.savefig("figure_trainloss_multivsuni.pdf".format(),
                 format="pdf", bbox_inches='tight')
