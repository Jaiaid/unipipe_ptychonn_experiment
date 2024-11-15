import os
import numpy as np
import matplotlib.pyplot as plot
import argparse
import re


COMPARED_SYSTEMS = ["worst_case", "multicontext", "unipipe", "unipipe_gd"]
COMPARED_SYSTEM_REGEX_DICT = {"worst_case": r"\[(\d+\.\d+)\]\s+INFERENCE BATCH SIZE,(\d+)", "multicontext": r"\[(\d+\.\d+)\]\s+INFERENCE BATCH SIZE,(\d+)", "unipipe": r"\[(\d+\.\d+)\]\s+UNIPIPE TRAIN,\s*INFER BS,(\d+),(\d+)", "unipipe_gd": r"\[(\d+\.\d+)\]\s+UNIPIPE TRAIN,\s*INFER BS,(\d+),(\d+)"}
SYSTEM_TO_COLORMARKER_DICT = {"worst_case": ("r", "+"), "multicontext": ("g", "o"), "unipipe": ("b", "x"), "unipipe_gd": ("y", "*")}


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dir", "-d", type=str, help="directory containing subfolder named by system which contains log files")

    args = argparser.parse_args()

    # data dictionary to collect all data
    data_dict = {}

    fig1, ax1 = plot.subplots(figsize=(20, 5))

    for system_idx, system in enumerate(COMPARED_SYSTEMS):
        data_dict[system] = {}
        data_dirpath = os.path.join(args.dir, system)
        
        label = []
        ticklabel = []

        for filename in os.listdir(data_dirpath):
            if system == "multicontext":
                if filename[-9:] != "infer.log":
                    continue
            else:
                if filename[-4:] != ".log":
                    continue

            tokens = filename.split(".")
            tokens = tokens[0].split("_")

            if system == "multicontext":
                interval_count = int(tokens[-5])
                interval_duration = int(tokens[-4])
                datarate = int(tokens[-3])
                deadline_msec = (1000/datarate) * int(tokens[-2])
            else:
                interval_count = int(tokens[-4])
                interval_duration = int(tokens[-3])
                datarate = int(tokens[-2])
                deadline_msec = (1000/datarate) * int(tokens[-1])

            if interval_duration not in data_dict[system]:
                data_dict[system][interval_duration] = {}

            ticklabel.append(system)
            label.append(system)
            cur_label = label[-1]

            data_filepath = os.path.join(data_dirpath, filename)
            inference_iteration_gap_timelist = []
            with open(data_filepath) as fin:
                pattern = COMPARED_SYSTEM_REGEX_DICT[system]
                for line in fin.readlines():
                    match = re.search(pattern, line)
                    if match:
                        timestamp = float(match.group(1))     # Extracts the timestamp (e.g., 1727860372.151061)
                        if system == "unipipe_gd" or system == "unipipe":
                            batch_size = int(match.group(3))
                        else:
                            batch_size = int(match.group(2))
                        inference_iteration_gap_timelist.append(batch_size)

                
            
            if len(inference_iteration_gap_timelist) == 0:
                mean = 100
                stddev = 0
                print(system, deadline_msec, datarate, interval_duration)
            else:
                # there are five intervals, there maybe significant gap between interval, to reduce error chance
                # take the first interval data
                inference_iteration_gap_timelist = inference_iteration_gap_timelist[:len(inference_iteration_gap_timelist)//40]

                # calc mean and std.dev
                nara = np.array(inference_iteration_gap_timelist[1:], dtype=np.float64)
                mean = nara.mean()
                stddev = nara.std()
            
            if datarate not in data_dict[system][interval_duration]:
                data_dict[system][interval_duration][datarate] = {}
            data_dict[system][interval_duration][datarate][deadline_msec] = [mean, stddev] 


    xticklabels = []
    xticks = []
    variant_idx = 0
    interval_duration_list = sorted(list(data_dict["unipipe"].keys()))
    for interval_duration in interval_duration_list:
        datarate_list = sorted(list(data_dict["unipipe"][interval_duration].keys()))
        ax1.axvline(variant_idx * 2 - 1, label='_nolegend_')
        for datarate in datarate_list:
            deadline_list = sorted(list(data_dict["unipipe"][interval_duration][datarate].keys()))

            for deadline in deadline_list:
                for system in COMPARED_SYSTEMS:
                    try:
                        ax1.scatter(
                            [variant_idx*2],
                            data_dict[system][interval_duration][datarate][deadline][0],
                            label=system, color=SYSTEM_TO_COLORMARKER_DICT[system][0], marker=SYSTEM_TO_COLORMARKER_DICT[system][1]
                        )
                    except Exception as e:
                        print(system, interval_duration, datarate, deadline)
                
                xticks.append(variant_idx*2)
                xticklabels.append("{0} {1} {2:.0f}".format(interval_duration, datarate, deadline))
                variant_idx += 1
        

    # # print(amperror_data_list, amperror_var_data_list)

    ax1.set_ylabel("Mean Batch Size")
    ax1.set_xlabel("")
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels, size=9, rotation=90)
    ax1.legend(COMPARED_SYSTEMS)
    ax1.set_ylim([0, 10])
    fig1.savefig("meaninferbs_vs_datarate_and_deadline.png", bbox_inches='tight')

