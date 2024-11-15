import os
import re
import math
import numpy as np
import matplotlib.pyplot as plot

DATA_DIR = "../result_logs/unipipe_trainbatch_mod_missrate_change"
COMPARED_SYSTEMS_LABEL_DICT = {"without_algo": "constant train BS", "with_algo":"dynamic train BS"}

if __name__ == "__main__":
    # Assuming invoke_times contains the extracted time values
    fig1, ax1 = plot.subplots()
    fig2, ax2 = plot.subplots()

    zero_missed_dict = {}

    for system_idx, system in enumerate(COMPARED_SYSTEMS_LABEL_DICT):
        filepath = os.path.join(DATA_DIR, "{0}.log".format(system))
        missed_counts = []
        zero_missed_dict[system] = 0
        with open(filepath) as fin:
            for line in fin.readlines():
                match = re.search(r"^Missed Inference Size:\s*([\d\.]+)", line)
                if match:
                    missed_counts.append(math.ceil(float(match.group(1))))
                    if missed_counts[-1] == 0:
                        zero_missed_dict[system] += 1
                # match = re.search(r"^GPU Inference Batch Size:\s*(\d+)", line)
                # if match:
                #     invoke_bs.append(float(match.group(1)))
                #     total_served += invoke_bs[-1]

            # Create a y-values list with a constant value, e.g., y=1
            print(len(missed_counts))
            ax1.scatter(list(range(len(missed_counts))), missed_counts, label=COMPARED_SYSTEMS_LABEL_DICT[system])
    
    ax1.set_xlabel('Request Collection Event')
    ax1.set_ylabel('Missed Inference Sample Count')
    ax1.set_ylim([0, 20])
    ax1.set_title('Missed Inference time For 3 Runs')
    ax1.grid(True)
    ax1.legend()
    fig1.savefig("missrate_effect_of_trainbatch_modification.png")
    print(zero_missed_dict)

    # ax1.set_xlabel('GPU Invoke Time (s)')
    # ax2.set_xlabel('GPU Infer Batch Size')
    # ax2.set_title('Scatter Plot of GPU Infer Batch Size For 3 Runs')
    # ax2.grid(True)
    # ax2.legend()
    # fig2.savefig("gpu_invokebs_differentbatch_system.png")