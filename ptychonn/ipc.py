import os
import numpy as np

def create_shm_marker(name:str):
    fd = open(os.path.join("/dev/shm", name), "w")
    fd.close()

def create_shm_folder(dirname:str):
    os.mkdir(path=os.path.join("/dev/shm", dirname))

def create_shm_data(name:str, data:np.ndarray):
    # https://numpy.org/doc/2.1/reference/generated/numpy.ndarray.tofile.html
    # We assume this is used to serialize data among processes in same system
    # and both processes know in which format data will come 
    # otherwise it will not work
    data.tofile(os.path.join("/dev/shm", name))


def read_shm_data(name:str, dtype=np.float32) -> np.ndarray:
    # https://numpy.org/doc/2.1/reference/generated/numpy.fromfile.html#numpy-fromfile
    # We assume this is used to serialize data among processes in same system
    # and both processes know in which format data will come 
    # otherwise it will not work
    try:
        return np.fromfile(os.path.join("/dev/shm", name), dtype=dtype)
    except FileNotFoundError as e:
        return None

def exist_shm(rel_path:str):
    return os.path.exists(os.path.join("/dev/shm", rel_path))

def move_shm(rel_path:str, new_rel_dirpath:str):
    basename = os.path.basename(rel_path)
    return os.rename(
        os.path.join("/dev/shm", rel_path),
        os.path.join(os.path.join("/dev/shm", new_rel_dirpath), basename))

def remove_shm(rel_path:str):
    return os.remove(os.path.join("/dev/shm", rel_path))


if __name__=="__main__":
    # to benchmark stuff
    import time
    REPEAT_COUNT = 100000
    BYTE_SIZE_PER_FILE = 64*64*4
    # create shm data, 1000000 data of 64,64 sized np.float32 numpy array
    data = np.asarray(np.random.rand(64,64), dtype=np.float32)

    start_time = time.time()
    for i in range(REPEAT_COUNT):
        create_shm_data(str(i), data)
    time_taken = time.time()-start_time
    print("Creating {0} file of size {1}Byte takes {2}s, on average {3}MB/s".format(
        REPEAT_COUNT, BYTE_SIZE_PER_FILE, time_taken, REPEAT_COUNT*BYTE_SIZE_PER_FILE/(time_taken*1e6)))

    start_time = time.time()
    for i in range(REPEAT_COUNT):
        tmp=read_shm_data(str(i))
    time_taken = time.time()-start_time
    print("Reading {0} file of size {1}Byte takes {2}s, on average {3}MB/s".format(
        REPEAT_COUNT, BYTE_SIZE_PER_FILE, time_taken, REPEAT_COUNT*BYTE_SIZE_PER_FILE/(time_taken*1e6)))
    
    # start_time = time.time()
    # for i in range(REPEAT_COUNT):
    #     move_shm(str(i), str(i)+"_moved")
    # time_taken = time.time()-start_time
    # print("Moving {0} file of size {1}Byte takes {2}s, on average {3}B/s".format(
    #     REPEAT_COUNT, BYTE_SIZE_PER_FILE, time_taken, REPEAT_COUNT*BYTE_SIZE_PER_FILE/time_taken))

    start_time = time.time()
    for i in range(REPEAT_COUNT):
        remove_shm(str(i))
    time_taken = time.time()-start_time
    print("Removing {0} file of size {1}Byte takes {2}s, on average {3}MB/s".format(
        REPEAT_COUNT, BYTE_SIZE_PER_FILE, time_taken, REPEAT_COUNT*BYTE_SIZE_PER_FILE/(time_taken*1e6)))
