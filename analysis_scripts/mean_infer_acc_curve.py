import os
import numpy as np
import matplotlib.pyplot as plot

DATA_DIR = "../result_logs"
COMPARED_SYSTEMS = ["multicontext", "unipipe", "worst_case"]

INTERVAL_COUNT = 5
WIDTH = 0.2
INTERVAL_TIMELIST = list(range(4, 17))
# INTERVAL_TIMELIST = [4, 5]
# INTERVAL_TIMELIST = [4, 5, 6, 7]

if __name__ == "__main__":
    fig1, ax1 = plot.subplots()
    fig2, ax2 = plot.subplots()

    for system_idx, system in enumerate(COMPARED_SYSTEMS):
        data_dirpath = os.path.join(DATA_DIR, system)
        
        label = []
        ticklabel = []
        amperror_data_list = []
        pherror_data_list = []
        amperror_var_data_list = [[], []]
        pherror_var_data_list = [[], []]

        for timeval in INTERVAL_TIMELIST:
            ticklabel.append(timeval)
            label.append(system)
            cur_label = label[-1]

            filename = "{0}_{1}_{2}.csv".format(system, INTERVAL_COUNT, timeval)
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
                    # if system in ["unipipe", "worst_case"]:
                    #     run_amp_error = [float(token) for token in tokens[2:2+INTERVAL_COUNT-1]]
                    #     run_ph_error = [float(token) for token in tokens[2+INTERVAL_COUNT:2+2*INTERVAL_COUNT]]
                    #     print(run_amp_error)
                    #     amp_error.append(np.mean(np.array(run_amp_error[1:])))
                    #     ph_error.append(np.mean(np.array(run_ph_error[1:])))
                    # elif system in ["multicontext", "worst_case"]:
                    #     run_amp_error = [float(token) for token in tokens[2:2+INTERVAL_COUNT-1]]
                    #     run_ph_error = [float(token) for token in tokens[2+INTERVAL_COUNT-1:2+2*INTERVAL_COUNT-2]]
                    #     print(run_amp_error)
                    #     amp_error.append(np.mean(np.array(run_amp_error)))
                    #     ph_error.append(np.mean(np.array(run_ph_error)))
                    run_amp_error = [float(token) for token in tokens[2:2+INTERVAL_COUNT-1]]
                    # print(run_amp_error)
                    run_ph_error = [float(token) for token in tokens[2+INTERVAL_COUNT-1:2+2*INTERVAL_COUNT-2]]
                    amp_error.append(np.mean(np.array(run_amp_error)))
                    ph_error.append(np.mean(np.array(run_ph_error)))
                    line_count += 1

                    # print(np.array(run_amp_error[1:]), np.mean(np.array(run_amp_error[1:])),np.array(run_ph_error[1:]), np.mean(np.array(run_ph_error[1:])))
                # from [1:] to avoid considering mean error
                # if system in ["unipipe", "worst_case"]:
                #     amperror_data_list.append(np.mean(np.array(amp_error[1:])))
                #     pherror_data_list.append(np.mean(np.array(ph_error[1:])))
                # elif system == "multicontext":
                amperror_data_list.append(np.mean(np.array(amp_error)))
                pherror_data_list.append(np.mean(np.array(ph_error)))
                        
                range_amp_error = np.array(amp_error) - amperror_data_list[-1]
                range_ph_error = np.array(ph_error) - pherror_data_list[-1]

                amperror_var_data_list[0].append(-np.min(range_amp_error))
                amperror_var_data_list[1].append(np.max(range_amp_error))
                pherror_var_data_list[0].append(-np.min(range_ph_error))
                pherror_var_data_list[1].append(np.max(range_ph_error))

        # print(ticklabel)
        ax1.set_xticks(np.arange(1, len(amperror_data_list)+1))
        ax2.set_xticks(np.arange(1, len(amperror_data_list)+1))
        ax1.bar(x=np.arange(1, len(amperror_data_list)+1) + WIDTH * system_idx, width=WIDTH, height=amperror_data_list, label=label, yerr=amperror_var_data_list)
        ax2.bar(x=np.arange(1, len(pherror_data_list)+1) + WIDTH * system_idx, width=WIDTH, height=pherror_data_list, label=label, yerr=pherror_var_data_list)
    
    # print(amperror_data_list, amperror_var_data_list)
    
    ax1.set_ylabel("Mean Amp. MSE")
    ax1.set_xlabel("Interval Duration (sec.)")
    ax1.set_xticklabels(ticklabel, size=7)
    ax1.legend(COMPARED_SYSTEMS)
    fig1.savefig("meanamp_error_interval_and_infservesystem.png")

    ax2.set_ylabel("Mean Phase MSE")
    ax2.set_xlabel("Interval Duration (sec.)")
    ax2.set_xticklabels(ticklabel, size=7)
    ax2.legend(COMPARED_SYSTEMS)
    fig2.savefig("meanph_error_interval_and_infservesystem.png")
