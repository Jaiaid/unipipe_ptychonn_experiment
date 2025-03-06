

import numpy as np
import os
import time
import argparse

import ipc
import parameters


def get_gtdata() -> np.ndarray:
    ground_truth_data = np.load(parameters.REAL_SPACE_PATH)
    ground_truth_amp = np.abs(ground_truth_data)
    ground_truth_ph = np.angle(ground_truth_data)

    # we will generate the data array by reading each line from ground truth amp. and phase data
    for i in range(parameters.DIFFRLINE):
        if i == 0:
            Y_I = ground_truth_amp[i, :].reshape(-1,parameters.H,parameters.W)
            Y_phi = ground_truth_ph[i, :].reshape(-1,parameters.H,parameters.W)
        else:
            Y_I = np.vstack((Y_I, ground_truth_amp[i, :].reshape(-1,parameters.H,parameters.W)))
            Y_phi = np.vstack((Y_phi, ground_truth_ph[i, :].reshape(-1,parameters.H,parameters.W)))

    return Y_I, Y_phi


# PERFORMANCE MODEL 1
# Phase Retrieval generation time length without any skipping
# Assumption:
# 1/phase_retrieval_genrate < deadline_sec
def estimate_T_IPR(
        phase_retrieval_genrate: float, acquisition_rate: float,
        deadline_sec: float):
    return phase_retrieval_genrate * deadline_sec / (acquisition_rate - phase_retrieval_genrate)


# the synchronization of transmission and producing like following
# consumer signal finish initiation, keeps waiting for producer to start transmission
# producer waits for finish consumer initiation, then signal transmission start

# signal producer to indicate finish of initiation
# then it will wait for transmission start
# this is part of mechanism to synchronize start of transmission and processing
def signal_producer():
    ipc.create_shm_marker(parameters.SHM_MARKER_IPR_INIT_FINISH)


# blocking function to wait for producer to start transmission
# this is part of mechanism to synchronize start of transmission and processing
def producer_transmit_wait():
    while not ipc.exist_shm(parameters.SHM_MARKER_TRANSMIT_START):
        pass


def cleanup():
    ipc.remove_shm(parameters.SHM_MARKER_IPR_INIT_FINISH)


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    # following arguments will be used to estimate 
    # when the mock process should skip 
    # how much to skip when generating ground truth
    parser.add_argument("--acquisition-rate", "-ar", type=float, help="at which rate (Hz/s^-1) new data will be coming")
    parser.add_argument("--generation-rate", "-gr", type=float, help="at which rate (Hz/s^-1) new ground truth will be generated")
    parser.add_argument("--deadline-msec", "-d", type=float, help="after how many millisecond a data file in shm will be removed, also determines interval length")
    parser.add_argument("--interval-count", "-icount", type=int, help="how many interval to run for")
    parser.add_argument("--interval-one-oracle", "-i", action="store_true", help="first interval all ground truth data will be made available")

    args = parser.parse_args()

    # ground truth data will be 161x161 (parameters.DIFFRLINE X parameters.DIFFRLINE) 
    # for each probe point in a 161x161 probe field we will have ground truth of 64x64 (parameters.H X parameters.W)
    gt_data_i, gt_data_ph = get_gtdata()
    total_data = gt_data_i.shape[0] * gt_data_i.shape[1]

    # generation state
    current_generate_idx = 0
    current_interval = 0

    total_generated = 0
    total_missed = 0
    retry_attempt = 0
    deadline_sec = args.deadline_msec/1000

    # perf. model estimated property
    time_stretch_continuous_data_process = estimate_T_IPR(
        phase_retrieval_genrate=args.generation_rate,
        acquisition_rate=args.acquisition_rate,
        deadline_sec=deadline_sec)
    skip_data_idx = round(deadline_sec * (args.acquisition_rate - args.generation_rate))

    # signal finish of initiation
    signal_producer()
    print("starting retrieval", time.time())
    # blockingwait until data streaming start
    producer_transmit_wait()
    print("data capture start", time.time())

    start_timestamp = time.time()
    # to give producer time to put first data
    time.sleep(1/args.acquisition_rate)
    for cur_interval in range(args.interval_count):
        cur_folder = parameters.SHM_MARKER_FMT_GTGENERATION_FOLDER.format(cur_interval)
        ipc.create_shm_folder(cur_folder)

        current_interval_start_timestamp = time.time()
        current_timestamp = time.time()
        while current_timestamp - current_interval_start_timestamp < deadline_sec:
            # to indicate consumption tp transmit process the data is deleted
            try:
                ipc.remove_shm(parameters.SHM_DATA_DIFFR_NAMEFMT.format(current_generate_idx))
                # retry_attempt = 0
            except FileNotFoundError as e:

                # current_generate_idx += 1
                print(current_generate_idx, list(os.listdir("/dev/shm/"))[-12:-1])
                current_timestamp = time.time()
                total_missed += 1
                # retry_attempt = 0
                # time.sleep(1/args.acquisition_rate)
                # print(e)
                continue

            # time gap to wait for the generation
            # it will be a busy loop
            data_process_interval_start_timestamp = current_timestamp
            while current_timestamp - data_process_interval_start_timestamp < 1/args.generation_rate:
                current_timestamp = time.time()

            # create the data in shared memory space /dev/shm
            ipc.create_shm_data(
                os.path.join(cur_folder, parameters.SHM_DATA_GEN_AMP_NAMEFMT.format(current_generate_idx)),
                gt_data_i[current_generate_idx])
            ipc.create_shm_data(
                os.path.join(cur_folder, parameters.SHM_DATA_GEN_PHASE_NAMEFMT.format(current_generate_idx)),
                gt_data_ph[current_generate_idx])
            total_generated += 1

            # increase generation idx
            current_generate_idx += 1
            current_timestamp = time.time()

        ipc.create_shm_marker(parameters.SHM_MARKER_FMT_INTERVAL_END.format(cur_interval))
        
        print("skipping to {0} by jumping {1}".format(current_generate_idx + skip_data_idx - 1, skip_data_idx - 1))
        current_generate_idx += skip_data_idx - 1

    print("==================================IPR Mock Status=================================")
    print("Data rate: {0}Hz".format(args.acquisition_rate))
    print("Deadline: {0}ms".format(args.deadline_msec))
    print("Generation rate: {0}Hz".format(args.generation_rate))
    print("Total Generated: {0}".format(total_generated))
    print("Total Missed: {0}".format(total_missed))
    total_time = time.time() - start_timestamp
    print("Total Time: {0}s".format(total_time))

    # output the transmission related data into a csv
    with open("ipr_generation_state.csv", "w") as fout:
        # rate,deadline_msec,total,consumed,missed,transmission time, total time
        fout.write(
            "{0},{1},{2},{3},{4},{5}\n".format(
                args.acquisition_rate, args.deadline_msec, args.generation_rate,
                total_generated, total_missed, total_time
            )
        )

    # cleanup
    cleanup()
