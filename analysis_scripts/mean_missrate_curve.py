import os
import numpy as np
import matplotlib.pyplot as plot

DATA_DIR = "../result_logs"
COMPARED_SYSTEMS = ["multicontext", "unipipe", "worst_case"]
WIDTH = 0.2
INTERVAL_COUNT = 5
INTERVAL_TIMELIST = list(range(4, 17))
# INTERVAL_TIMELIST = [4, 5]
# INTERVAL_TIMELIST = [4, 5, 6, 7]

if __name__ == "__main__":
    fig1, ax1 = plot.subplots()
    fig2, ax2 = plot.subplots()

    for system_idx, system in enumerate(COMPARED_SYSTEMS):
        missrate_data_list = []
        missrate_data_var_list = [[], []]
        missrate_datastreamer_data_list = []
        missrate_datastreamer_data_var_list = [[], []]
        label = []
        ticklabel = []

        data_dirpath = os.path.join(DATA_DIR, system)
        
        for timeval in INTERVAL_TIMELIST:
            label.append(system)
            ticklabel.append(timeval)
            cur_label = label[-1]

            filename = "{0}_{1}_{2}.csv".format(system, INTERVAL_COUNT, timeval)
            data_filepath = os.path.join(data_dirpath, filename)

            with open(data_filepath) as fin:
                # first reading it into a list
                # to calculate both average and variance
                missrate = []
                missrate_datastreamer = []
                line_count = 0
                for line in fin.readlines():
                    tokens = line.split(",")
                    
                    run_missrates = [float(token) for token in tokens[-8:-4]]
                    run_missrates_datastreamer = [float(token) for token in tokens[-3:]]
                    # print(run_missrates)
                    missrate.append(np.mean(np.array(run_missrates)))
                    missrate_datastreamer.append(np.mean(np.array(run_missrates_datastreamer)))
                    line_count += 1

                # from [1:] to avoid considering mean error
                missrate_data_list.append(np.mean(np.array(missrate))*100)
                missrate_datastreamer_data_list.append(np.mean(np.array(missrate_datastreamer))*100)
                        
                range_missrate = np.array(missrate) - missrate_data_list[-1]
                range_missrate_datastreamer = np.array(missrate_datastreamer) - missrate_datastreamer_data_list[-1]
                # print(range_missrate, np.array(missrate_data_list), missrate_data_list[-1])

                missrate_data_var_list[0].append(abs(np.min(range_missrate)))
                missrate_data_var_list[1].append(np.max(range_missrate))

                missrate_datastreamer_data_var_list[0].append(abs(np.min(range_missrate_datastreamer)))
                missrate_datastreamer_data_var_list[1].append(np.max(range_missrate_datastreamer))
        # print(missrate_data_list, missrate_data_var_list)
        ax1.set_xticks(np.arange(len(missrate_data_list)))
        ax2.set_xticks(np.arange(len(missrate_datastreamer_data_list)))
        ax1.bar(x=np.arange(len(missrate_data_list)) + WIDTH * system_idx, width=WIDTH, height=missrate_data_list, label=label, yerr=missrate_data_var_list)
        ax2.bar(x=np.arange(len(missrate_datastreamer_data_list)) + WIDTH * system_idx, width=WIDTH, height=missrate_datastreamer_data_list, label=label, yerr=missrate_datastreamer_data_var_list)

    
    ax1.set_ylabel("Mean Miss Rate(%)")
    ax1.set_xlabel("Interval Duration (sec.)")
    ax1.set_xticklabels(ticklabel, rotation=90, size=7)
    ax1.legend(COMPARED_SYSTEMS)
    fig1.savefig("meanmissrate_interval_and_infservesystem.png")
    
    ax2.set_ylabel("Mean Miss Rate Datastream Overhead (%)")
    ax2.set_xlabel("Interval Duration (sec.)")
    ax2.set_xticklabels(ticklabel, rotation=90, size=7)
    ax2.legend(COMPARED_SYSTEMS)
    fig2.savefig("meanmissrate_datastream_interval_and_infservesystem.png")
