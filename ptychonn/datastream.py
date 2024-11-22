import time
import math
import torch
import numpy as np

import logfast
import logfast.fastlogger


class DataStream:
    def __init__(self, datarate, deadline_sec, dataset: torch.utils.data.Dataset) -> None:
        self.index = 0
        self.dataset = dataset
        self.dataset_copy2 = self.dataset
        # print(type(self.dataset[0:2]),type(self.dataset[0:2][0]),self.dataset[0:2][0].shape)
        # exit(0)
        self.start_time = 0
        self.last_query_time = None
        self.datarate = datarate
        self.deadline_sec = deadline_sec
        self.accumulated = 0

        self.transmitted = 0
        self.missed = 0

        self.avg_delay = 0
        pass

    def start_stream(self):
        self.start_time = time.time()
        self.last_query_time = self.start_time

    def get_deadlinesec(self):
        return self.deadline_sec
    
    def get_datarate(self):
        return self.datarate

    def extract(self, bs=None, logger:logfast.fastlogger=None):
        # first calculate how much time has passed
        # we can use that to understand how much data has accumulated
        if self.last_query_time is not None:
            time_passed = time.time() - self.last_query_time
        else:
            time_passed = 1/self.datarate

        if logger is not None:
            logger.log("EXTRACT CALLED")

        # if time passed from last extraction > deadline, we have some stale requests
        if time_passed > self.deadline_sec:
            # we are assuming the deadline countdown starts afte whole request accumulates
            # therefore, missed count will be truncated, 0.5 missed half of the request is missed not complete
            missed = int((time_passed - self.deadline_sec) * self.datarate)

            # bring the last query time forward to the point where the last non stale request started to arrive
            # similarly change the time passed
            self.last_query_time += missed / self.datarate
            time_passed -= missed / self.datarate

            # remove the missed data from accumulated
            self.accumulated -= missed
        else:
            missed = 0

        # calculate how much non stale data has accumulated
        self.accumulated += self.datarate * time_passed

        # take at least one reading to ensure we have updated the last query time
        time_mark = time.time()
        # we will start to count how much data has accumulated from this point
        self.last_query_time = time_mark

        # entering in this loop means not enough data has accumulated to be extracted
        threshold_bs_size = 1 if bs is None else bs
        while self.accumulated < threshold_bs_size:
            time_mark = time.time()
            self.accumulated += (time_mark - self.last_query_time) * self.datarate
            self.last_query_time = time_mark
        
        # no particular batch size requested means greedy selection
        to_serve = int(self.accumulated) if bs is None else bs

        self.index = (self.index + missed) % len(self.dataset)
        # print(self.index, missed, to_serve, time_passed)
        final_index = (self.index + to_serve) % len(self.dataset)
        # TODO:
        # think about more precise deadline list
        # current assumption is that the last accumulated has completed arrival on last_query_time
        # which may not be the case, it may be in self.last_query_time - [0, 1/self.datarate)
        deadlines = [(self.last_query_time - i/self.datarate + self.deadline_sec) for i in range(to_serve - 1, -1, -1)]
        if final_index < self.index and final_index != 0:
            data = (torch.vstack((self.dataset[self.index:][0], self.dataset_copy2[:final_index][0])),
                    torch.vstack((self.dataset[self.index:][1], self.dataset_copy2[:final_index][1])),
                    torch.vstack((self.dataset[self.index:][2], self.dataset_copy2[:final_index][2])))
        else:
            final_index = self.index + to_serve
            # print(len(self.dataset[self.index:final_index]), self.dataset[self.index:final_index][0].shape)
            data = (self.dataset[self.index:final_index][0], self.dataset[self.index:final_index][1], self.dataset[self.index:final_index][2])

        # if not completely accumulated we don't consider it as transmitted
        self.transmitted += int(self.accumulated) + missed
        self.missed += missed
        self.index = final_index
        # update accumulated data for next extraction
        # to_serve amount will be given to callee, hence the subtraction
        self.accumulated -= to_serve
        # print(self.accumulated, self.transmitted)
        # print(len(data), data[0][0].shape)
        return data, deadlines, missed
    
    def get_perf(self):
        # print((time.time() - self.start_time) * self.datarate)
        return self.transmitted, self.transmitted - self.missed, self.missed