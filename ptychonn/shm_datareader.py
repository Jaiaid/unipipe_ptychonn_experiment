import numpy as np
import os

from . import parameters
from . import ipc


class SHMInferDataReader():
    """
        Holds state of data which is meant to be read as inference 
        Data is stored in SHM for our custom producer consumer scenario

        It holds following states to maintain call between read
        .cur_readidx : next read will be attempted from /dev/shm/<cur_readidx>.raw

        Assumption:
        1. Data source is infinite
        2. Best attempt read, it is not guarenteed it will be able to read bs size data
        3. At initiation will always try to read data from 0 of bs size, consumer should 
           call .reposition() explitictly to avoid that. Otherwise, multiple read call 
           may return nothing until bs indices is passed and current existing idx is reached 
    """
    def __init__(self):
        self.cur_readidx = 0
        pass

    def read(self, bs) -> tuple[np.ndarray, int]:
        consumed = 0
        missed = 0
        ara = None
        for i in range(bs):
            try:
                # read it and add to batch
                if consumed == 0:
                    ara = ipc.read_shm_data(
                        parameters.SHM_DATA_DIFFR_NAMEFMT.format(self.cur_readidx)
                    ).reshape(1, 1, parameters.H, parameters.W)
                else:
                    ara = np.vstack(
                        (
                            ara, ipc.read_shm_data(
                                parameters.SHM_DATA_DIFFR_NAMEFMT.format(self.cur_readidx)
                            ).reshape(1, 1, parameters.H, parameters.W)
                        ) 
                    )

                # inference will be done only once
                # so delete
                ipc.remove_shm(parameters.SHM_DATA_DIFFR_NAMEFMT.format(self.cur_readidx))
                consumed += 1 
            except Exception as e:
                missed += 1
            self.cur_readidx += 1

        return ara, consumed, missed
    
    def reposition(self):
        """
            This method is special one to move the read head to current next available data 
            When it needs to move far 
            For example, Consumer may have some processing to incur delay meanwhile other 
            consumers have consumed some data
            
            For efficiency reason, it should be better called by consumer
        """
        for filename in os.listdir("/dev/shm"):
            if ".raw" in filename:
                self.cur_readidx = int(filename.split(".")[0])


class SHMTrainDataReader():
    """
        Holds state of data which is meant to be read as training
        Data is stored in SHM for our custom producer consumer scenario
        Data is stored in multiple folders, so unlike inference data, here needs to
        keep state of which data folder will be read

        It holds following states to maintain call between read
        .cur_ipriteration : which iteration of ground truth generation will be consumed now
            determine another internal attribute .cur_datafoldername 's value
        .cur_datafoldername : next read will be attempted from /dev/shm/<self.cur_datafoldername> folder
        .cur_readidx : 
            next read will be attempted from /dev/shm/<self.cur_datafoldername>/<cur_readidx>.raw
            and from /dev/shm/<self.cur_datafoldername>/<cur_readidx>.rawgti
            and from /dev/shm/<self.cur_datafoldername>/<cur_readidx>.rawgtph
        .cur_readidx_begin : to set .cur_readidx value when reset is called,
            useful for multiepoch training

        Assumption:
        1. Data source is finite and in which folder determined by cur_ipriteration attribute
           Consumer is expected to set it explicitly
        2. Best attempt read, it is not guarenteed it will be able to read bs size data
        3. At initiation will always try to read data from 0 of bs size, consumer should 
           call .reposition() explitictly to avoid that. Otherwise, multiple read call 
           may return nothing until bs indices is passed and current existing idx is reached 
    """
    def __init__(self):
        self.cur_readidx = 0
        # stored so in a epoch rotation can be done for a chunk of data
        self.cur_readidx_begin = 0
        self.cur_ipriteration = 0
        self.cur_datafoldername = parameters.SHM_MARKER_FMT_GTGENERATION_FOLDER.format(self.cur_ipriteration)
        pass

    def set_curipriteration(self, cur_ipriteration):
        self.cur_ipriteration = cur_ipriteration

    def reset(self):
        self.cur_readidx = self.cur_readidx_begin

    def read(self, bs) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        consumed = 0
        ara1 = ara2 = ara3 = None
        for i in range(bs):
            try:
                if consumed == 0:
                    ara1 = ipc.read_shm_data(
                        os.path.join(
                            self.cur_datafoldername,
                            parameters.SHM_DATA_DIFFR_NAMEFMT.format(self.cur_readidx)
                        )
                    ).reshape(1, 1, parameters.H, parameters.W)
                    ara2 = ipc.read_shm_data(
                        os.path.join(
                            self.cur_datafoldername,
                            parameters.SHM_DATA_GEN_AMP_NAMEFMT.format(self.cur_readidx)
                        )
                    ).reshape(1, 1, parameters.H, parameters.W)
                    ara3 = ipc.read_shm_data(
                        os.path.join(
                            self.cur_datafoldername,
                            parameters.SHM_DATA_GEN_PHASE_NAMEFMT.format(self.cur_readidx)
                        )
                    ).reshape(1, 1, parameters.H, parameters.W)
                else:
                    ara1 = np.vstack(
                        (
                            ara1,
                            ipc.read_shm_data(
                                os.path.join(
                                    self.cur_datafoldername,
                                    parameters.SHM_DATA_DIFFR_NAMEFMT.format(self.cur_readidx)
                                )
                            ).reshape(1, 1, parameters.H, parameters.W)
                        ) 
                    )
                    ara2 = np.vstack(
                        (
                            ara2, ipc.read_shm_data(
                                os.path.join(
                                    self.cur_datafoldername,
                                    parameters.SHM_DATA_GEN_AMP_NAMEFMT.format(self.cur_readidx)
                                )
                            ).reshape(1, 1, parameters.H, parameters.W)
                        ) 
                    )
                    ara3 = np.vstack(
                        (
                            ara3, ipc.read_shm_data(
                                os.path.join(
                                    self.cur_datafoldername,
                                    parameters.SHM_DATA_GEN_PHASE_NAMEFMT.format(self.cur_readidx)
                                )
                            ).reshape(1, 1, parameters.H, parameters.W)
                        ) 
                    )
                consumed += 1
            except Exception as e:
                pass
            
            self.cur_readidx += 1

        return ara1, ara2, ara3, consumed
    
    def reposition(self):
        """
            This method is special one to move the read head to available ground truth 
            When it needs to move far or needs changing data folder
            
            For efficiency reason, it should be better called by consumer
        """
        for filename in sorted(
                os.listdir(
                    os.path.join(
                        "/dev/shm",
                        parameters.SHM_MARKER_FMT_GTGENERATION_FOLDER.format(self.cur_ipriteration)
                    )
                )
            ):
            if ".rawgti" in filename:
                self.cur_readidx = int(filename.split(".")[0])
                self.cur_readidx_begin = self.cur_readidx