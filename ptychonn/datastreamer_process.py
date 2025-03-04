import numpy as np
import time
import argparse

import ipc
import parameters

from skimage.transform import resize

def get_diffrdata() -> np.ndarray:
    diffr_data = np.load(parameters.DATA_DIFFR_PATH)["arr_0"]

    diffr_data_red = np.zeros((diffr_data.shape[0],diffr_data.shape[1],64,64), float)
    for i in range(1, diffr_data.shape[0]):
        for j in range(diffr_data.shape[1]):
            diffr_data_red[i,j] = resize(diffr_data[i,j,32:-32,32:-32],(64,64),preserve_range=True, anti_aliasing=True)
            diffr_data_red[i,j] = np.where(diffr_data_red[i,j]<3,0,diffr_data_red[i,j])

    return diffr_data


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--rate", "-r", type=float, help="at which rate (Hz/s^-1) new data will be created")
    parser.add_argument("--deadline-msec", "-dmsec", type=float, help="after how many millisecond a data file in shm will be removed, determine the ring buffer length")

    args = parser.parse_args()

    # diffr data will be 64x64 for each probe point in a 161x161 probe field
    diffr_data = get_diffrdata()
    total_image_count = diffr_data.shape[0] * diffr_data.shape[1]

    # transmission state
    current_transmit_idx = 0
    # if consumer delete it we consider it consumed
    # followings are also needed for proper cleanup
    consumed = 0
    missed = 0

    next_delete_idx = 0
    deadline_sec = args.deadline_msec/1000
    deadline_list = []
    transmission_list = []

    start_timestamp = time.time()
    data_interval_start_timestamp = start_timestamp
    for i in range(diffr_data.shape[0]):
        for j in range(diffr_data.shape[1]):
            current_timestamp = time.time()
            # keep deleting data if deadline over
            while next_delete_idx < len(deadline_list) and deadline_list[next_delete_idx] < current_timestamp:
                try:
                    ipc.remove_shm_marker("{0}.raw".format(next_delete_idx))
                    missed += 1
                except FileNotFoundError as e:
                    consumed += 1
                next_delete_idx += 1
                current_timestamp = time.time()

            # we assume deadline >> interval between two data samples
            # therefore, waiting for new data to arrive will not cause deadline to be over significantly
            while current_timestamp - data_interval_start_timestamp < 1/args.rate:
                current_timestamp = time.time()
            # create the data in shared memory space /dev/shm
            ipc.create_shm_data("{0}.raw".format(current_transmit_idx), diffr_data[i,j])
            # append to deadline list
            deadline_list.append(current_timestamp + deadline_sec)
            # increase transmit idx
            current_transmit_idx += 1

            # set new data interval start timestamp
            data_interval_start_timestamp = current_timestamp

    # wait until data are consumed or deadline over
    current_timestamp = time.time()
    transmission_end_time = current_timestamp
    while consumed + missed < current_transmit_idx:
        current_timestamp = time.time()
        # keep deleting data if deadline over
        while next_delete_idx < len(deadline_list) and deadline_list[next_delete_idx] < current_timestamp:
            try:
                ipc.remove_shm_marker("{0}.raw".format(next_delete_idx))
                missed += 1
            except FileNotFoundError:
                consumed += 1
            next_delete_idx += 1
            current_timestamp = time.time()

    print("==================================Data Streamer Status=================================")
    print("Data rate: {0}Hz".format(args.rate))
    print("Deadline: {0}ms".format(args.deadline_msec))
    print("Total Transmitted Data: {0}".format(current_transmit_idx))
    print("Consumed: {0}".format(consumed))
    print("Missed: {0}".format(missed))
    print("Total Transmission Time: {0}s".format(transmission_end_time - start_timestamp))
    total_time = time.time() - start_timestamp
    print("Total Time: {0}s".format(total_time))

    # output the transmission related data into a csv
    with open("transmission_state.csv", "w") as fout:
        # rate,deadline_msec,total,consumed,missed,transmission time, total time
        fout.write(
            "{0},{1},{2},{3},{4},{5},{6}\n".format(
                args.rate, args.deadline_msec, current_transmit_idx, consumed, missed,
                transmission_end_time - start_timestamp, total_time
            )
        )
