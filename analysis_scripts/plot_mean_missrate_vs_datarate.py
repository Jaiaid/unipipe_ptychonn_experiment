import os
import numpy as np
import traceback
import matplotlib.pyplot as plot
import argparse


COMPARED_SYSTEMS = ["pretrained", "worst_case", "multicontext", "unipipe"]
SYSTEM_TO_COLORMARKER_DICT = {"pretrained": ("c", "*"), "worst_case": ("r", "+"), "multicontext": ("g", "o"), "unipipe": ("b", "x")}


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dir", "-d", type=str, help="directory containing subfolder named by system which contains log files")

    args = argparser.parse_args()

    # data dictionary to collect all data
    data_dict = {}
    evaluated_systems = []

    fig1, ax1 = plot.subplots(figsize=(20, 5))

    for system_idx, system in enumerate(COMPARED_SYSTEMS):
        data_dict[system] = {}
        data_dirpath = os.path.join(args.dir, system)
        if not os.path.exists(data_dirpath):
            continue
        else:
            evaluated_systems.append(system)
        
        label = []
        ticklabel = []

        for filename in os.listdir(data_dirpath):
            if filename[-4:] != ".csv":
                continue

            tokens = filename.split(".")
            tokens = tokens[0].split("_")

            # 
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
            with open(data_filepath) as fin:
                # first reading it into a list
                # to calculate both average and variance
                missrate = []
                line_count = 0
                for line in fin.readlines():
                    run_missrate = []
                    tokens = line.split(",")
                    
                    run_missrate = [float(token) for token in tokens[-8:-4]]
                    missrate.append(np.mean(np.array(run_missrate))*100)
                    line_count += 1

                if datarate not in data_dict[system][interval_duration]:
                    data_dict[system][interval_duration][datarate] = {}
                data_dict[system][interval_duration][datarate][deadline_msec] = [missrate] 
    
    handle_dict = {}
    xticklabels = []
    xticks = []
    variant_idx = 0
    interval_duration_list = sorted(list(data_dict["unipipe"].keys()))
    for interval_duration in interval_duration_list:
        datarate_list = sorted(list(data_dict["unipipe"][interval_duration].keys()))

        for datarate in datarate_list:
            deadline_list = sorted(list(data_dict["unipipe"][interval_duration][datarate].keys()))

            for deadline in deadline_list:
                for sysidx, system in enumerate(COMPARED_SYSTEMS):
                    try:
                        handle_dict[system] = ax1.bar(
                            [variant_idx*2 - sysidx * 0.25],
                            data_dict[system][interval_duration][datarate][deadline][0],
                            label=system, color=SYSTEM_TO_COLORMARKER_DICT[system][0], width=0.25
                        )
                    except Exception as e:
                        print(traceback.format_exc())
                        print(system, interval_duration, datarate, deadline)
                
                xticks.append(variant_idx*2)
                xticklabels.append("{0} {1} {2:.0f}".format(interval_duration, datarate, deadline))
                variant_idx += 1
        

    # # print(amperror_data_list, amperror_var_data_list)
    print(handle_dict)
    ax1.set_ylabel("Mean Missrate (%)")
    ax1.set_xlabel("")
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels, size=9, rotation=90)
    ax1.legend(handles=[handle_dict[system] for system in evaluated_systems], labels=COMPARED_SYSTEMS)
    fig1.savefig("meanmissrate_vs_datarate_and_deadline.png", bbox_inches='tight')

