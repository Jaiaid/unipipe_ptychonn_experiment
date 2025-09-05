"""
CHATGPT initiated code

Provide two log files as cmd line arguments
"""

import re
import matplotlib.pyplot as plot
import argparse


def parse_log(file_path):
    PATTERN_BEGIN = r"^\[(\d+\.\d+)\].*BEGIN$"
    PATTERN_MODELLOAD_EVENT = r"^\[(\d+\.\d+)\] MODEL LOAD TAKES,([\d\.eE+-]+)$"
    PATTERN1 = r"^\[(\d+\.\d+)\] UNIPIPE TRAIN, INFER BS,\d+,(\d+)$"
    PATTERN2 = r"^\[(\d+\.\d+)\] MULTICONTEXT INFER BS,(\d+)$"

    zero_timestamp = 0
    zero_time_found = False
    results = []
    model_load_events = []
    with open(file_path, 'r') as f:
        for line in f:
            # do this search until zero time found
            if not zero_time_found:
                match = re.search(PATTERN_BEGIN, line)
                if match:
                    zero_timestamp = float(match.group(1))
                    zero_time_found = True

            match = re.search(PATTERN_MODELLOAD_EVENT, line)
            if match:
                timestamp = float(match.group(1))
                dur = float(match.group(2))
                model_load_events.append([timestamp-dur, dur])

            # Match lines with a timestamp and at least one trailing number
            match = re.search(PATTERN1, line)
            if match:
                timestamp = float(match.group(1))
                last_number = int(match.group(2))
                if last_number != 0:
                # print(last_number, line)
                    results.append([timestamp, last_number])
            
            match = re.search(PATTERN2, line)
            if match:
                timestamp = float(match.group(1))
                last_number = int(match.group(2))
                if last_number != 0:
                # print(last_number, line)
                    results.append([timestamp, last_number])

    mean = 0
    for i, val in enumerate(results):
        results[i][0] = results[i][0] - zero_timestamp
        mean += results[i][1]
    mean /= len(results)

    mean_timediff = 0
    for i in range(1, len(results)):
        mean_timediff += results[i][0] - results[i-1][0]
    mean_timediff /=  (len(results) - 1)
    
    for i in range(len(model_load_events)):
        model_load_events[i][0] = model_load_events[i][0] - zero_timestamp

    return results, mean, mean_timediff, model_load_events


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-multicontext", "-log1", type=str, help="path to log file with multicontext run info")
    parser.add_argument("--log-unipipe", "-log2", type=str, help="path to log file with unipipe run info")
    parser.add_argument("--outfile-bname", "-outfbname", type=str, help="file name without extension")
    args = parser.parse_args()

    log1_results, log1_results_mean, log1_results_timediff_mean, model_load_events = parse_log(args.log_multicontext)
    log2_results, log2_results_mean, log2_results_timediff_mean, _ = parse_log(args.log_unipipe)

    fig, ax = plot.subplots(figsize=(4, 2.25))
    ax.plot([res[0] for res in log1_results], [res[1] for res in log1_results], label="Multicontext", marker="x", markersize=1)
    ax.plot([res[0] for res in log2_results], [res[1] for res in log2_results], label="Unipipe", marker="o", markersize=1)
    ax.scatter([res[0] for res in model_load_events], [1 for res in model_load_events], label="Model Load", marker="+")
    print(log1_results_mean, log2_results_mean, log1_results_timediff_mean, log2_results_timediff_mean)
    # min_time = min([res[0] for res in log1_results])
    # avg_difference1 = 
    # ax.set_xlim([min_time, min_time+5])

    ax.legend()
    fig.savefig(args.outfile_bname+".png", bbox_inches="tight", dpi=600)
