"""
Producer for Ptychographic Data Acquisition Simulation

A datastreamer which streams data at parameter rate
By streaming we mean create .raw file for each diffraction data

Data is deleted at parameter deadline

If data is not found when deleting it is assumed to be consumed
If consumer do not signal of their presence it will not start producing
Thereofore, conusmer of the data has to do the followings
1. Signal presence by creating entry at /dev/shm
2. Read and delete the data
"""

import numpy as np
import time
import argparse
import h5py
import os

from skimage.transform import resize

from ptychonn import ipc
from ptychonn import parameters
from ptychonn import dataset


# the synchronization of transmission and producing like following
# consumer signal finish initiation, keeps waiting for producer to start transmission
# producer waits for finish consumer initiation, then signal transmission start

# blocking function to wait for consumer to start initiation
# this is part of mechanism to synchronize start of transmission and processing
def consumer_init_wait():
    # wait for IPR process to finish initiation
    while not ipc.exist_shm(parameters.SHM_MARKER_IPR_INIT_FINISH):
        pass
    # wait for ML process to finish initiation
    while not ipc.exist_shm(parameters.SHM_MARKER_ML_INIT_FINISH):
        pass


def signal_consumer():
    ipc.create_shm_marker(parameters.SHM_MARKER_TRANSMIT_START)


def cleanup(args):
    if not args.no_sync:
        ipc.remove_shm(parameters.SHM_MARKER_TRANSMIT_START)


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--rate", "-r", type=float, help="at which rate (Hz/s^-1) new data will be created")
    parser.add_argument("--deadline-msec", "-dmsec", type=float, help="after how many millisecond a data file in shm will be removed")
    parser.add_argument("--no-sync", "-nsync", action="store_true", help="no synchronization with consumer, needed if run independently")
    parser.add_argument("--skip-line-pretrained", "-skipline", type=int, help="how many data to skip as model is pretrained on it")
    parser.add_argument("--debug-log", "-debug", action="store_true", help="debug message print")
    parser.add_argument("--large-dataset", "-largedataset", action="store_true", help="if larger dataset will be ysed")

    args = parser.parse_args()

    # diffr data will be 64x64 for each probe point in a 161x161 probe field
    if not args.large_dataset:
        diffr_data = dataset.get_diffrdata(skip_line=args.skip_line_pretrained)
    else:
        diffr_data = dataset.get_diffrdata_large(skip_line=args.skip_line_pretrained)
    
    total_image_count = diffr_data.shape[0] * diffr_data.shape[1]

    # transmission state
    current_transmit_idx = 0
    # if consumer delete it we consider it consumed
    # followings are also needed for proper cleanup
    consumed = 0
    missed = 0
    consumed_nonpretrained = 0
    missed_nonpretrained = 0

    next_delete_idx = 0
    deadline_sec = args.deadline_msec/1000
    deadline_list = []
    transmission_list = []

    # wait for consumer to finish initiation
    print("waiting for consumer to join")
    if not args.no_sync:
        consumer_init_wait()
    # indicate start of activity
    signal_consumer()


    start_timestamp = time.time()
    print("starting transmission", start_timestamp)
    data_interval_start_timestamp = start_timestamp
    for i in range(diffr_data.shape[0]):
        for j in range(diffr_data.shape[1]):
            current_timestamp = time.time()
            # keep deleting data if deadline over
            first_delete_idx = next_delete_idx
            initial_missed_count = missed
            while next_delete_idx < len(deadline_list) and deadline_list[next_delete_idx] < current_timestamp:
                try:
                    ipc.remove_shm(parameters.SHM_DATA_DIFFR_NAMEFMT.format(next_delete_idx))
                    missed += 1
                    if next_delete_idx > total_image_count/5:
                        missed_nonpretrained += 1
                except FileNotFoundError as e:
                    consumed += 1
                    if next_delete_idx > total_image_count/5:
                        consumed_nonpretrained += 1
                next_delete_idx += 1
                current_timestamp = time.time()
            if args.debug_log and missed != initial_missed_count:
                print(time.time(), "Deleted {0} samples from {1} to {2}".format(missed - initial_missed_count, first_delete_idx, next_delete_idx - 1))

            # we assume deadline >> interval between two data samples
            # therefore, waiting for new data to arrive will not cause deadline to be over significantly
            while current_timestamp - data_interval_start_timestamp < 1/args.rate:
                current_timestamp = time.time()
            # create the data in shared memory space /dev/shm
            ipc.create_shm_data(parameters.SHM_DATA_DIFFR_NAMEFMT.format(current_transmit_idx), diffr_data[i,j])
            if args.debug_log:
                print(time.time(), "Created sample {0}".format(current_transmit_idx))

            # append to deadline list
            deadline_list.append(current_timestamp + deadline_sec)
            # increase transmit idx
            current_transmit_idx += 1

            # set new data interval start timestamp
            data_interval_start_timestamp = current_timestamp

    # wait until data are consumed or deadline over
    current_timestamp = time.time()
    transmission_end_time = current_timestamp
    first_delete_idx = next_delete_idx
    initial_missed_count = missed
    missed_after_evaluation_time = 0
    while consumed + missed < current_transmit_idx:
        current_timestamp = time.time()
        # keep deleting data if deadline over
        while next_delete_idx < len(deadline_list) and deadline_list[next_delete_idx] < current_timestamp:
            try:
                ipc.remove_shm(parameters.SHM_DATA_DIFFR_NAMEFMT.format(next_delete_idx))
                missed += 1
                missed_after_evaluation_time += 1
                if next_delete_idx > total_image_count/5:
                    missed_nonpretrained += 1
            except FileNotFoundError:
                consumed += 1
                if next_delete_idx > total_image_count/5:
                    consumed_nonpretrained += 1
            next_delete_idx += 1
            current_timestamp = time.time()
    if args.debug_log and missed != initial_missed_count:
        print(time.time(), "Deleted {0} samples from {1} to {2}".format(missed - initial_missed_count, first_delete_idx, next_delete_idx - 1))

    print("==================================Data Streamer Status=================================")
    print("Data rate: {0}Hz".format(args.rate))
    print("Deadline: {0}ms".format(args.deadline_msec))
    print("Total Transmitted Data: {0}".format(current_transmit_idx))
    print("Consumed: {0}".format(consumed))
    print("Missed: {0}".format(missed))
    print("Missed After Evaluation Time: {0}".format(missed_after_evaluation_time))
    print("Total Transmission Time: {0}s".format(transmission_end_time - start_timestamp))
    total_time = time.time() - start_timestamp
    print("Total Time: {0}s".format(total_time))

    # output the transmission related data into a csv
    with open("transmission_state.csv", "w") as fout:
        # rate,deadline_msec,total,consumed,missed,transmission time, total time
        fout.write(
            "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}\n".format(
                args.rate, args.deadline_msec, current_transmit_idx, consumed, missed, missed_after_evaluation_time,
                (missed-missed_after_evaluation_time)/current_transmit_idx, consumed_nonpretrained, missed_nonpretrained,
                transmission_end_time - start_timestamp, total_time
            )
        )

    # transmission is not ongoing and all deadline are finished
    # args needed to determine if in no sync mode 
    try:
        cleanup(args)
    except Exception as e:
        print("Exception during cleanup:", e)
    print("Data streamer finished cleanup and exit")
