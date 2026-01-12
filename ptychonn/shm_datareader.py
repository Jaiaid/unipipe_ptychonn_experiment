import math
import time
import os
import numpy as np

from typing import Tuple, List

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
    def __init__(self, bs=64, dryrun_mode=False, start_timestamp=None, datarate=None, deadline_sec=None, stream_alive_time=None):
        self.cur_readidx = 0
        self.len = 0
        self.batch_size = bs
        self.dataara = np.asarray(np.random.rand(bs,1,64,64),dtype=np.float32)
        self.dryrun_mode = dryrun_mode
        self.start_timestamp = start_timestamp
        self.last_read_timestamp = start_timestamp
        self.stream_alive_time = stream_alive_time
        self.datarate = datarate
        self.deadline_sec = deadline_sec
        if self.deadline_sec is not None:
            self.max_available_bs = int(self.deadline_sec * self.datarate)
        else:
            self.max_available_bs = None

    def set_len(self, len):
        self.len = len

    def __len__(self) -> int:
        return self.len

    def read(self, bs, logger=None, blocking_call=False) -> Tuple[np.ndarray, int, int, List[int]]:
        consumed = 0
        missed = 0
        file_notfound_exceptions = 0
        exceptions = 0
        dataidx_list = []
        ara = None

        self.reposition()

        if self.max_available_bs is not None:
            bs = min (bs, self.max_available_bs)

        available_bs = min(int(math.floor(self.datarate * (time.time() - self.last_read_timestamp))), self.max_available_bs)
        if blocking_call:
            # print(available_bs, bs, self.last_read_timestamp, self.start_timestamp, self.stream_alive_time)
            # busy wait until enough data is available or time limit is reached
            while available_bs < bs and time.time() - self.start_timestamp < self.stream_alive_time:
                available_bs = int(math.floor(self.datarate * (time.time() - self.last_read_timestamp)))

        if logger is not None:
            logger.log("EXPECTED AVAILABLE BS FOR READ", available_bs, bs, self.cur_readidx, self.last_read_timestamp)
        # to handle initial condition
        # as inference probe is always behind at the beginning it is possible read idx set at negative
        # the dataset size should also be set 0 but that check is not done here
        # dataset size was added later, inference shm reader supposed to read blindly from current probe/idx position
        if self.cur_readidx >= 0 or self.dryrun_mode:
            for i in range(bs):
                try:
                    if not self.dryrun_mode:
                        # read it and add to batch
                        tmp_data = ipc.read_shm_data(
                            parameters.SHM_DATA_DIFFR_NAMEFMT.format(self.cur_readidx)
                        )

                        if tmp_data is not None:
                            if consumed == 0:
                                ara = tmp_data.reshape(1, 1, parameters.H, parameters.W)
                            else:
                                ara = np.vstack(
                                    (
                                        ara, tmp_data.reshape(1, 1, parameters.H, parameters.W)
                                    )
                                )
                        
                            dataidx_list.append(self.cur_readidx)
                            consumed += 1

                            # inference will be done only once
                            # so delete
                            ipc.remove_shm(
                                parameters.SHM_DATA_DIFFR_NAMEFMT.format(self.cur_readidx)
                            )

                            self.cur_readidx += 1
                except FileNotFoundError as e:
                    # print(e)
                    missed += 1
                    file_notfound_exceptions += 1
                    if not self.dryrun_mode:
                        if logger is not None:
                            logger.log("SHM INFER DATA READER FILE NOT FOUND EXCEPTION AT READ IDX, REPOSITIONING", self.cur_readidx)  
                        self.reposition()
                except Exception as e:
                    # print(e)
                    exceptions += 1
                    if not self.dryrun_mode:
                        if logger is not None:
                            logger.log("SHM INFER DATA READER EXCEPTION AT READ IDX, REPOSITIONING", self.cur_readidx)  
                        self.reposition()


        if ara is not None and ara.shape[0] > 0:
            logger.log("SHM INFER DATA READER READ", ara.shape, self.cur_readidx, file_notfound_exceptions, exceptions, self.last_read_timestamp)
            self.last_read_timestamp = self.start_timestamp + ((self.cur_readidx-1) / self.datarate)
        
        if self.dryrun_mode:
            return self.dataara[:consumed,], consumed, missed, list(range(self.cur_readidx - bs, self.cur_readidx - bs + consumed))
        
        return ara, consumed, missed, dataidx_list
    
    def reposition(self):
        """
            This method is special one to move the read head to current next available data 
            When it needs to move far 
            For example, Consumer may have some processing to incur delay meanwhile other 
            consumers have consumed some data
            
            For efficiency reason, it should be better called by consumer
        """
        # for filename in sorted(os.listdir("/dev/shm")):
        #     if ".raw" in filename:
        #         self.cur_readidx = int(filename.split(".")[0])
        #         return
        if self.start_timestamp is not None and self.datarate is not None:
            # adjust read idx according to current time
            expected_idx = max(0, math.floor((time.time() - self.start_timestamp - self.deadline_sec*9/10) * self.datarate))
            if self.cur_readidx < expected_idx:
                # print("Repositioned infer read idx to ", expected_idx+1, " from expected idx ", self.cur_readidx, time.time())
                self.cur_readidx = expected_idx


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
    def __init__(self, bs=64):
        self.cur_readidx = 0
        # stored so in a epoch rotation can be done for a chunk of data
        self.cur_readidx_begin = 0
        self.cur_readidx_end = 0
        self.cur_ipriteration = 0
        self.cur_datafoldername = parameters.SHM_MARKER_FMT_GTGENERATION_FOLDER.format(self.cur_ipriteration)
        self.dryrun_mode = True
        self.dataara1 = np.asarray(np.random.rand(bs,1,64,64),dtype=np.float32)
        self.dataara2 = np.asarray(np.random.rand(bs,1,64,64),dtype=np.float32)
        self.dataara3 = np.asarray(np.random.rand(bs,1,64,64),dtype=np.float32)
        pass

    def __len__(self) -> int:
        """
        Returned value is useful only after 
        .set_curipriteration() and .reposition() is called once
        """
        return self.cur_readidx_end - self.cur_readidx_begin + 1

    def set_curipriteration(self, cur_ipriteration):
        self.cur_ipriteration = cur_ipriteration
        self.cur_datafoldername = parameters.SHM_MARKER_FMT_GTGENERATION_FOLDER.format(self.cur_ipriteration)

    def set_len(self, begin, end):
        self.cur_readidx_begin = begin 
        self.cur_readidx_end = end
        self.cur_readidx = self.cur_readidx_begin
 
    def reset(self):
        self.cur_readidx = self.cur_readidx_begin

    def read(self, bs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        consumed = 0
        ara1 = ara2 = ara3 = None
        for i in range(bs):
            if self.dryrun_mode:
                pass
            else:
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
                            ), dtype=np.float32
                        ).reshape(1, 1, parameters.H, parameters.W)
                        ara3 = ipc.read_shm_data(
                            os.path.join(
                                self.cur_datafoldername,
                                parameters.SHM_DATA_GEN_PHASE_NAMEFMT.format(self.cur_readidx)
                            ), dtype=np.float32
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
                                    ), dtype=np.float32
                                ).reshape(1, 1, parameters.H, parameters.W)
                            ) 
                        )
                        ara3 = np.vstack(
                            (
                                ara3, ipc.read_shm_data(
                                    os.path.join(
                                        self.cur_datafoldername,
                                        parameters.SHM_DATA_GEN_PHASE_NAMEFMT.format(self.cur_readidx)
                                    ), dtype=np.float32
                                ).reshape(1, 1, parameters.H, parameters.W)
                            ) 
                        )
                    consumed += 1
                except Exception as e:
                    # print(e, os.path.join(
                    #     self.cur_datafoldername,
                    #     parameters.SHM_DATA_GEN_PHASE_NAMEFMT.format(self.cur_readidx)
                    # ))
                    # print(
                    #     os.path.join(
                    #             self.cur_datafoldername,
                    #             parameters.SHM_DATA_GEN_PHASE_NAMEFMT.format(self.cur_readidx)
                    #         ),
                    #     os.path.exists(
                    #         os.path.join(
                    #             self.cur_datafoldername,
                    #             parameters.SHM_DATA_GEN_PHASE_NAMEFMT.format(self.cur_readidx)
                    #         )
                    #     )
                    # )
                    pass
            
            self.cur_readidx += 1
            if self.cur_readidx > self.cur_readidx_end:
                self.cur_readidx = self.cur_readidx_begin

        if self.dryrun_mode:
            return self.dataara1[:bs,], self.dataara2[:bs,], self.dataara3[:bs,], bs
        return ara1, ara2, ara3, consumed
    
    def get_dataidxlist(self) -> List[int]:
        return list(range(self.cur_readidx_begin, self.cur_readidx_end + 1))

    def reposition(self):
        """
            This method is special one to move the read head to available ground truth 
            When it needs to move far or needs changing data folder
            
            For efficiency reason, it should be better called by consumer
        """

        self.cur_datafoldername = parameters.SHM_MARKER_FMT_GTGENERATION_FOLDER.format(self.cur_ipriteration)

        sorted_filelist = sorted(
            list(os.listdir(
                os.path.join(
                    "/dev/shm",
                    self.cur_datafoldername
                ))
            )
        )

        for filename in sorted_filelist:
            if ".rawgti" in filename:
                self.cur_readidx = int(filename.split(".")[0])
                self.cur_readidx_begin = self.cur_readidx
                break
        
        for i in range(len(sorted_filelist) - 1, -1, -1):
            if ".rawgti" in sorted_filelist[i]:
                self.cur_readidx_end = int(sorted_filelist[i].split(".")[0])
                break

        # print(self.cur_datafoldername, self.cur_readidx_begin, self.cur_readidx_end)
