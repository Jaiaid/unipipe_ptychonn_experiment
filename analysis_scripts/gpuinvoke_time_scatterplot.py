import os
import re
import numpy as np
import matplotlib.pyplot as plot

DATA_DIR = "../result_logs/multicontext/20240730_3run_infertimes"
COMPARED_SYSTEMS = ["single_batch", "dyn_batch"]
INTERVAL_DURATION = 4
INTERVAL_COUNT = 5

if __name__ == "__main__":
    # Assuming invoke_times contains the extracted time values
    fig1, ax1 = plot.subplots()
    fig2, ax2 = plot.subplots()

    for system_idx, system in enumerate(COMPARED_SYSTEMS):
        filepath = os.path.join(DATA_DIR, "multicontext_{0}_{1}_{2}_infer.log".format(INTERVAL_COUNT, INTERVAL_DURATION, system))
        invoke_times = []
        invoke_bs = []
        total_served = 0
        with open(filepath) as fin:
            for line in fin.readlines():
                match = re.search(r"^GPU Invoke\s+Time:\s*([\d\.]+)", line)
                if match:
                    invoke_times.append(float(match.group(1)))
                match = re.search(r"^GPU Inference Batch Size:\s*(\d+)", line)
                if match:
                    invoke_bs.append(float(match.group(1)))
                    total_served += invoke_bs[-1]

            # Create a y-values list with a constant value, e.g., y=1
            invoke_times = np.array(invoke_times)
            invoke_times -= np.min(invoke_times)
            print(len(invoke_bs), total_served)
            ax1.scatter(invoke_times, invoke_bs, marker='+', label=system)
    
    ax1.set_xlabel('GPU Invoke Time (s)')
    ax1.set_ylabel('Batch Size')
    ax1.set_title('Scatter Plot of GPU Invoke BS and time For 3 Runs')
    ax1.grid(True)
    ax1.legend()
    fig1.savefig("gpu_invoketime_differentbatch_system.png")

    # ax1.set_xlabel('GPU Invoke Time (s)')
    # ax2.set_xlabel('GPU Infer Batch Size')
    # ax2.set_title('Scatter Plot of GPU Infer Batch Size For 3 Runs')
    # ax2.grid(True)
    # ax2.legend()
    # fig2.savefig("gpu_invokebs_differentbatch_system.png")