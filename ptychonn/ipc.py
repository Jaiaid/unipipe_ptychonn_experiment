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


def read_shm_data(name:str, data:np.ndarray):
    # https://numpy.org/doc/2.1/reference/generated/numpy.fromfile.html#numpy-fromfile
    # We assume this is used to serialize data among processes in same system
    # and both processes know in which format data will come 
    # otherwise it will not work
    try:
        data.tofile(os.path.join("/dev/shm", name))
    except FileNotFoundError as e:
        return 0

    return 1

def exist_shm(rel_path:str):
    return os.path.exists(os.path.join("/dev/shm", rel_path))

def remove_shm(rel_path:str):
    return os.remove(os.path.join("/dev/shm", rel_path))