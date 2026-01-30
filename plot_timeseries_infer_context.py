"""
This script will follow similar structure to plot_missrate_barclusters_dratevariation.py
but will plot the time breakdown of different activities.

How it will get the data?
It will read from same basename format but log file
Using regex it will extract the inference invocation gap values and plot them.
The line of interest in the log file looks like: [1765484226.5689094] FORWARD PASS TOOK(sec.),0.0014719963073730469

It will plot bar clusters for different systems for only one datarate

Write the code for me copilot, please
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import argparse

from plot_parameters import SYSTEM_NAME_LIST, SYSTEM_NAME_TO_LEGEND_DICT, SYSTEM_NAME_TO_HATCH_DICT, CSV_FILENAME_FMT, SYSSTAT_FILENAME_FMT

IPR_RATE_LIST = [16]
DEADLINE_LIST = [80]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, help="path containing the result directory in approproate format")
    parser.add_argument("--datarate", "-r", type=float, help="which datarate to plot")
    parser.add_argument("--output-file-basename", "-o", type=str, help="output name without extension")
    args = parser.parse_args()

    # Dictionary to store average values: {param: [sys1_val, sys2_val, ...]}
    io_lat_list = {sys: [] for sys in SYSTEM_NAME_LIST}
    fwpass_lat_list = {sys: [] for sys in SYSTEM_NAME_LIST}
    bwpass_lat_list = {sys: [] for sys in SYSTEM_NAME_LIST}
    chkpt_lat_list = {sys: [] for sys in SYSTEM_NAME_LIST}
    ctxswitch_lat_list = {sys: [] for sys in SYSTEM_NAME_LIST}
    inferbs_list = {sys: [] for sys in SYSTEM_NAME_LIST}
    gpuutil_list = {sys: [] for sys in SYSTEM_NAME_LIST}

    # Read values and compute means
    param = int(args.datarate)

    for sys in SYSTEM_NAME_LIST:
        # for log file multicontext is named differently
        if sys == "multicontext":
            log_path = os.path.join(args.dir, sys, CSV_FILENAME_FMT.format(sys, DEADLINE_LIST[0], param, IPR_RATE_LIST[0]).replace('.csv', '_infer.log'))
        else:
            log_path = os.path.join(args.dir, sys, CSV_FILENAME_FMT.format(sys, DEADLINE_LIST[0], param, IPR_RATE_LIST[0]).replace('.csv', '.log'))
        
        sysstat_path = os.path.join(args.dir, sys, SYSSTAT_FILENAME_FMT.format(sys, DEADLINE_LIST[0], param, IPR_RATE_LIST[0]))


        with open(log_path, 'r') as f:
            total_consumed = 0
            fwpass_lat = 0
            bwpass_lat = 0
            io_lat = 0
            chkpt_lat = 0
            ctxswitch_lat = 0
            iteration_count = 0
            start_timestamp = None
            end_timestamp = None
            
            for line in f.readlines():
                match = re.search(r'INFER READ LATENCY,([\d.eE+-]+),(\d+)', line)
                if match:
                    io_lat = float(match.group(1))
                    inferbs = int(match.group(2))

                match = re.search(r'BACKWARD TAKES\(sec.\),([\d.eE+-]+)', line)
                if match:
                    bwpass_lat = float(match.group(1))

                match = re.search(r'FORWARD PASS TOOK\(sec.\),([\d.eE+-]+)', line)
                if match:
                    fwpass_lat = float(match.group(1))

                match = re.search(r'MODEL LOAD TAKES,([\d.eE+-]+)', line)
                if match:
                    chkpt_lat = float(match.group(1))

                match = re.search(r'INFER GAP,([\d.eE+-]+)', line)
                if match:
                    ctxswitch_lat = float(match.group(1))

                match = re.search(r'ITERATION TAKES\(sec.\),([\d.eE+-]+)', line)
                if match:
                    io_lat_list[sys].append(io_lat)
                    fwpass_lat_list[sys].append(fwpass_lat)
                    bwpass_lat_list[sys].append(bwpass_lat)
                    chkpt_lat_list[sys].append(chkpt_lat)
                    inferbs_list[sys].append(inferbs)
                    ctxswitch_lat_list[sys].append(float(match.group(1)) - fwpass_lat - bwpass_lat - chkpt_lat)
                    iteration_count += 1

        with open(sysstat_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    continue  # skip header
                parts = line.strip().split(',')
                
                gpu_util = float(parts[4])  # GPU util is in 5th column
                gpuutil_list[sys].append(gpu_util)

    # Plotting
    num_systems = len(SYSTEM_NAME_LIST)
    bar_width = 0.5
    x = np.arange(num_systems)
    
    fig, ax = plt.subplots(figsize=(4, 2.25))
    ax1 = ax.twinx()
    
    # plot stacked bars and in different axis draw line plot for per sample fw pass time
    # we take each type of latency and stack on top of each other for each system
    # there will be stack for each system every stack will have mean latency of each type
    # in this sequence: io, fwpass, bwpass, chkpt, ctxswitch 
    io_mean = np.array([np.mean(io_lat_list[sys])*1000 for sys in SYSTEM_NAME_LIST])  # convert to ms
    fwpass_mean = np.array([np.mean(fwpass_lat_list[sys])*1000 for sys in SYSTEM_NAME_LIST])
    bwpass_mean = np.array([np.mean(bwpass_lat_list[sys])*1000 for sys in SYSTEM_NAME_LIST])
    chkpt_mean = np.array([np.mean(chkpt_lat_list[sys])*1000 for sys in SYSTEM_NAME_LIST])
    ctxswitch_mean = np.array([np.mean(ctxswitch_lat_list[sys])*1000 for sys in SYSTEM_NAME_LIST])
    
    # ax.bar(x, io_mean, width=bar_width, label='IO', bottom=0, color='lightblue', edgecolor='black')
    ax.bar(x, fwpass_mean, width=bar_width, label='Forward Pass', bottom=0, color='lightgreen', edgecolor='black')
    ax.bar(x, bwpass_mean, width=bar_width, label='Backward Pass', bottom=fwpass_mean, color='lightcoral', edgecolor='black')
    ax.bar(x, chkpt_mean, width=bar_width, label='Checkpoint', bottom=fwpass_mean+bwpass_mean, color='lightsalmon', edgecolor='black')
    # ax.bar(x, ctxswitch_mean, width=bar_width, label='Context Switch', bottom=fwpass_mean+bwpass_mean+chkpt_mean, color='lightgray', edgecolor='black')

    ax.set_ylabel('Avg. Latency (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels([SYSTEM_NAME_TO_LEGEND_DICT[sys] for sys in SYSTEM_NAME_LIST], fontsize=4)
    ax.legend(frameon=False, fontsize=6, loc='upper left')

    # ax1.set_ylabel('Per Sample FW. (s/req.)', color='blue')
    # ax1.tick_params(axis='y', labelcolor='blue')
    # infer_thpt = [(np.mean(fwpass_lat_list[sys]))/np.mean(inferbs_list[sys]) for sys in SYSTEM_NAME_LIST]
    # ax1.plot(x, infer_thpt, color='blue', marker='o')

    # ax1.set_ylabel('Infer Batch Size (s/req.)', color='blue')
    # ax1.tick_params(axis='y', labelcolor='blue')
    # infer_bs = [np.mean(inferbs_list[sys]) for sys in SYSTEM_NAME_LIST]
    # ax1.plot(x, infer_bs, color='blue', marker='o')
    ax1.set_ylabel('GPU Util (%)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    gpu_util = []
    for sys in SYSTEM_NAME_LIST:
        p90 = np.percentile(gpuutil_list[sys], 90)
        data = np.array(gpuutil_list[sys])
        gpu_util.append(data[data<=p90].mean())
    ax1.plot(x, gpu_util, color='blue', marker='o')

    percentile_90_all = [np.percentile(fwpass_lat_list[sys]+bwpass_lat_list[sys]+chkpt_lat_list[sys], 90)*1000 for sys in SYSTEM_NAME_LIST]
    ax.scatter(x, percentile_90_all, color='black', marker='x', label='P90 Total Latency')
    ax.legend(frameon=False, fontsize=6, loc='upper left')

    percentile_90_fw = [np.percentile(fwpass_lat_list[sys], 90)*1000 for sys in SYSTEM_NAME_LIST]
    ax.scatter(x, percentile_90_fw, color='red', marker='+', label='P90 FW Latency')
    ax.legend(frameon=False, fontsize=6, loc='upper left')
    output_path = f"{args.output_file_basename}_{args.datarate}reqpersec.png"
    fig.savefig(output_path, dpi=600, bbox_inches="tight")

    # print([np.any(bwpass_lat_list[sys])<0 for sys in SYSTEM_NAME_LIST])
    # print([np.any(chkpt_lat_list[sys])<0 for sys in SYSTEM_NAME_LIST])
    # print([np.any(fwpass_lat_list[sys])<0 for sys in SYSTEM_NAME_LIST])
    # print([np.any(ctxswitch_lat_list[sys])<0 for sys in SYSTEM_NAME_LIST])
    # print([np.any(io_lat_list[sys])<0 for sys in SYSTEM_NAME_LIST])
    