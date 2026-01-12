import argparse
import os
import time
import ptychonn.parameters
import ptychonn.ipc

"""
This code verifies if the datastreamer shared memory based streaming shows the streamed and stale files 
with appropriate timestamps from consumer process.

We check for one particular file in the streamed and stale directories to see if the timestamps are updated as expected.

* We keep looking if file idx 10000 exists
* If it exists we keep looking when it first appeared in streamed
* Then we keep looking when it first appeared in stale by checking when it stop existing in the streamed directory
"""

FILE_INDEX_TO_CHECK = 20500


def verify_shm_streaming(start_time, stream_rate, deadline_ms):
    expected_time_to_exist = start_time + ((FILE_INDEX_TO_CHECK+1) / stream_rate)
    expected_time_to_stale = expected_time_to_exist + (deadline_ms / 1000.0)

    target_file_path = os.path.join("/dev/shm/", ptychonn.parameters.SHM_DATA_DIFFR_NAMEFMT.format(FILE_INDEX_TO_CHECK))

    print(f"Looking for {target_file_path} in streamed directory...")

    # Wait for the file to appear in the streamed directory
    while not os.path.exists(target_file_path):
        pass

    streamed_timestamp = time.time()
    print(f"File {target_file_path} appeared in streamed directory at timestamp {streamed_timestamp}")
    print("Difference with expected time to exist: {0} seconds".format(streamed_timestamp - expected_time_to_exist))
    print(f"Waiting for {target_file_path} to move to stale directory...")

    # Wait for the file to disappear from the streamed directory
    while os.path.exists(target_file_path):
        pass

    stale_timestamp = time.time()
    print(f"File {target_file_path} appeared in stale directory at timestamp {stale_timestamp}")
    print("Difference with expected time to stale: {0} seconds".format(stale_timestamp - expected_time_to_stale))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify SHM streaming for PtychoNN")
    parser.add_argument("--file_index", type=int, default=FILE_INDEX_TO_CHECK,
                        help="Index of the file to check in SHM streaming")
    parser.add_argument("--data_rate", "-drate", type=float, help="Data rate in files per second for expected timing verification")
    parser.add_argument("--deadline", "-dl", type=float, help="Deadline in milliseconds for expected timing verification")

    args = parser.parse_args()

    # wait for some time to let the streamer start
    start_time, total_runtime = ptychonn.ipc.producer_transmit_wait()
    
    # start_time = time.time() # + 0.00005  # adjust for small delay
    print(f"Detected streamer start marker at {start_time}, starting verification...")

    verify_shm_streaming(start_time, stream_rate=args.data_rate, deadline_ms=args.deadline)
