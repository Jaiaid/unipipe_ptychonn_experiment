import os

def create_shm_marker(name:str):
    fd = open(os.path.join("/dev/shm", name), "w")
    fd.close()