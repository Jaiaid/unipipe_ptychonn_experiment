import os
import time

class FastLogger:
    SHMFILE_NAME_FORMAT = "{0}_fastlog.log"
    DELIMETER = ","

    def __init__(self):
        self.pid = os.getpid()
        self.mfid = open(os.path.join("/dev/shm/", FastLogger.SHMFILE_NAME_FORMAT.format(self.pid)), "w")
        self.delimeter = FastLogger.DELIMETER

    def set_delimeter(self, delimeter:str):
        self.delimeter = delimeter

    def log(self, *args, end='\n', logtime=True):
        args = [str(arg) for arg in args]
        if logtime:
            line = "[{0}] {1}{2}".format(time.time(), self.delimeter.join(args), end)
        else:
            line = "{0}{1}".format(self.delimeter.join(args), end)
        self.mfid.write(line)

    def persist(self, filepath: str):
        self.mfid.close()

        pfid = open(filepath, "w")
        with open(os.path.join("/dev/shm/", FastLogger.SHMFILE_NAME_FORMAT.format(self.pid))) as fin:
            for line in fin.readlines():
                pfid.write("{0}".format(line))
        pfid.close()
        
        os.remove(os.path.join("/dev/shm/", FastLogger.SHMFILE_NAME_FORMAT.format(self.pid)))
