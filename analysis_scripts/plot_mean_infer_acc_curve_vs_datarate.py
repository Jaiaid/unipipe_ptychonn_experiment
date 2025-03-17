import os
import numpy as np
import matplotlib.pyplot as plot
import traceback
import argparse

import plotprop

SYSTEM_TO_COLORMARKER_DICT = {"pretrained": ("c", "*"), "worst_case": ("r", "+"), "multicontext": ("g", "o"), "unipipe": ("b", "x")}

# INTERVAL_TIMELIST = [4, 5]
# INTERVAL_TIMELIST = [4, 5, 6, 7]

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dir", "-d", type=str, help="directory containing subfolder named by system which contains log files")

    args = argparser.parse_args()

    # data dictionary to collect all data
    data_dict = {}
    evaluated_systems = []

    fig1, ax1 = plot.subplots(figsize=(20, 5))
    fig2, ax2 = plot.subplots(figsize=(20, 5))

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
            if filename[-4:] != ".csv" or "sysstat" in filename:
                continue

            tokens = filename.split(".")
            tokens = tokens[0].split("_")

            if system == "multicontext":
                interval_count = int(tokens[-4])
                interval_duration = int(tokens[-3])
                datarate = int(tokens[-2])
                deadline_msec = int(tokens[-1])
            else:
                interval_count = int(tokens[-4])
                interval_duration = int(tokens[-3])
                datarate = int(tokens[-2])
                deadline_msec = int(tokens[-1])
            if deadline_msec not in [100, 500]:
                continue

            print(filename,tokens)

            if interval_duration not in data_dict[system]:
                data_dict[system][interval_duration] = {}

            ticklabel.append(system)
            label.append(system)
            cur_label = label[-1]

            data_filepath = os.path.join(data_dirpath, filename)
            with open(data_filepath) as fin:
                # first reading it into a list
                # to calculate both average and variance
                amp_error = []
                ph_error = []
                line_count = 0
                for line in fin.readlines():
                    run_amp_error = []
                    run_ph_error = []
                    tokens = line.split(",")
                    
                    run_amp_error = [float(token) for token in tokens[2:2+interval_count-1]]
                    # print(run_amp_error)
                    run_ph_error = [float(token) for token in tokens[2+interval_count-1:2+2*interval_count-2]]
                    amp_error.append(np.mean(np.array(run_amp_error)))
                    ph_error.append(np.mean(np.array(run_ph_error)))
                    line_count += 1

                if datarate not in data_dict[system][interval_duration]:
                    data_dict[system][interval_duration][datarate] = {}
                data_dict[system][interval_duration][datarate][deadline_msec] = [amp_error, ph_error] 
    
    handle_dict1 = {}
    handle_dict2 = {}
    xticklabels = []
    xticks = []
    variant_idx = 0
    print(data_dict)
    interval_duration_list = sorted(list(data_dict["unipipe"].keys()))
    for interval_duration in interval_duration_list:
        datarate_list = sorted(list(data_dict["unipipe"][interval_duration].keys()))
        ax1.axvline(variant_idx * 2 - 1, label='_nolegend_')
        ax2.axvline(variant_idx * 2 - 1, label='_nolegend_')
        for datarate in datarate_list:
            deadline_list = sorted(list(data_dict["unipipe"][interval_duration][datarate].keys()))

            for deadline in deadline_list:
                for sysidx, system in enumerate(plotprop.COMPARED_SYSTEMS):
                    try:
                        print(system)
                        handle_dict1[system] = ax1.bar(
                            [variant_idx*2 - sysidx * 0.25],
                            data_dict[system][interval_duration][datarate][deadline][0],
                            label=system, color=SYSTEM_TO_COLORMARKER_DICT[system][0], width=0.25
                        )

                        handle_dict2[system] = ax2.bar(
                            [variant_idx*2 - sysidx * 0.25],
                            data_dict[system][interval_duration][datarate][deadline][1],
                            label=system, color=SYSTEM_TO_COLORMARKER_DICT[system][0], width=0.25
                        )

                    except Exception as e:
                        print(traceback.format_exc())
                        print(system, interval_duration, datarate, deadline)
                
                xticks.append(variant_idx*2)
                xticklabels.append("{0} {1} {2:.0f}".format(interval_duration, datarate, deadline))
                variant_idx += 1

    # # print(amperror_data_list, amperror_var_data_list)

    ax1.set_ylabel("Mean Amp. MSE")
    ax1.set_xlabel("")
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels, size=9, rotation=90)
    ax1.legend(handles=[handle_dict1[system] for system in evaluated_systems],
               labels=[plotprop.COMPARED_SYSTEMS_LEGEND_DICT[system] for system in evaluated_systems])
    fig1.savefig("meanamp_error_vs_datarate_and_deadline.pdf", format="pdf", bbox_inches='tight')

    ax2.set_ylabel("Mean Phase MSE")
    ax2.set_xlabel("")
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels, size=9, rotation=90)
    ax2.legend(handles=[handle_dict2[system] for system in evaluated_systems],
               labels=[plotprop.COMPARED_SYSTEMS_LEGEND_DICT[system] for system in evaluated_systems])
    fig2.savefig("meanph_error_vs_datarate_and_deadline.pdf", format="pdf", bbox_inches='tight')
