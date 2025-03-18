import os
import numpy as np
import traceback
import matplotlib.pyplot as plot
import argparse

import plotprop


SYSTEM_TO_COLORMARKER_DICT = {"pretrained": ("c", "*"), "multicontext": ("r", "+"), "unipipe": ("b", "x")}


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dir", "-d", type=str, help="directory containing subfolder named by system which contains log files")

    args = argparser.parse_args()

    # data dictionary to collect all data
    data_dict = {}
    evaluated_systems = []

    fig1, ax1 = plot.subplots(figsize=(20, 5))

    for system_idx, system in enumerate(plotprop.COMPARED_SYSTEMS):
        data_dict[system] = {}
        data_dirpath = os.path.join(args.dir, system)
        if not os.path.exists(data_dirpath):
            continue
        else:
            evaluated_systems.append(system)
        
        label = []
        ticklabel = []

        for filename in os.listdir(data_dirpath):
            if "transmission_state" not in filename:
                continue

            tokens = filename.split(".")
            tokens = tokens[0].split("_")

            # 
            if system == "multicontext":
                interval_count = int(tokens[-4])
                deadline = int(tokens[-3])
                datarate = int(tokens[-2])
                iprt = int(tokens[-1])
            else:
                interval_count = int(tokens[-4])
                deadline = int(tokens[-3])
                datarate = int(tokens[-2])
                iprt = int(tokens[-1])
            # if deadline_msec not in [5 , 10]:
            #     continue

            if deadline not in data_dict[system]:
                data_dict[system][deadline] = {}

            ticklabel.append(system)
            label.append(system)
            cur_label = label[-1]

            data_filepath = os.path.join(data_dirpath, filename)
            line_count = 0
            with open(data_filepath) as fin:
                # first reading it into a list
                # to calculate both average and variance
                missrate = 100
                line_count = 0
                for line in fin.readlines():
                    tokens = line.split(",")
                    print(tokens, data_filepath)
                    consumed = int(tokens[5])
                    missed = int(tokens[6])
                    missrate = missed * 100/(consumed + missed)
                    line_count += 1

                assert line_count == 1, "transmission state has more than single line, CHECK {0}".format(data_filepath)

                if datarate not in data_dict[system][deadline]:
                    data_dict[system][deadline][datarate] = {}
                data_dict[system][deadline][datarate][iprt] = [missrate] 
    
    handle_dict = {}
    xticklabels = []
    xticks = []
    variant_idx = 0
    deadline_list = sorted(list(data_dict["unipipe"].keys()))
    for deadline in deadline_list:
        datarate_list = sorted(list(data_dict["unipipe"][deadline].keys()))

        for datarate in datarate_list:
            iprt_list = sorted(list(data_dict["unipipe"][deadline][datarate].keys()))

            for iprt in iprt_list:
                for sysidx, system in enumerate(plotprop.COMPARED_SYSTEMS):
                    try:
                        handle_dict[system] = ax1.bar(
                            [variant_idx*2 - sysidx * 0.25],
                            data_dict[system][deadline][datarate][iprt][0],
                            label=system, color=SYSTEM_TO_COLORMARKER_DICT[system][0], width=0.25
                        )
                    except Exception as e:
                        print(traceback.format_exc())
                        # print(system, deadline, datarate, deadline)
                
                xticks.append(variant_idx*2)
                xticklabels.append("{0} {1} {2:.0f}".format(deadline, datarate, iprt))
                variant_idx += 1
        

    # # print(amperror_data_list, amperror_var_data_list)
    print(handle_dict)
    ax1.set_ylabel("Mean Missrate (%)")
    ax1.set_xlabel("")
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels, size=9, rotation=90)
    ax1.legend(handles=[handle_dict[system] for system in evaluated_systems],
               labels=[plotprop.COMPARED_SYSTEMS_LEGEND_DICT[system] for system in evaluated_systems])
    fig1.savefig("meanmissrate_vs_datarate_and_deadline.pdf", format="pdf", bbox_inches='tight')

