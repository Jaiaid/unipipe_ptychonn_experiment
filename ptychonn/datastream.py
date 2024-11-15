import time
import math
import torch
import numpy as np


class DataStream:
    def __init__(self, datarate, deadline_sec, dataset: torch.utils.data.Dataset) -> None:
        self.index = 0
        self.dataset = dataset
        self.dataset_copy2 = self.dataset
        # print(type(self.dataset[0:2]),type(self.dataset[0:2][0]),self.dataset[0:2][0].shape)
        # exit(0)
        self.last_query_time = None
        self.datarate = datarate
        self.deadline_sec = deadline_sec
        self.accumulated = 0

        self.transmitted = 0
        self.missed = 0

        self.avg_delay = 0
        pass

    def start_stream(self):
        self.last_query_time = time.time()

    def get_deadlinesec(self):
        return self.deadline_sec
    
    def get_datarate(self):
        return self.datarate

    def extract(self, bs=None):
        if self.last_query_time is not None:
            time_passed = time.time() - self.last_query_time
        else:
            time_passed = 1/self.datarate

        # if time passed from last extraction > deadline, we have some stale requests
        if time_passed > self.deadline_sec:
            # we are assuming the deadline countdown starts afte whole request accumulates
            # therefore, missed count will be truncated, 0.5 missed half of the request is missed not complete
            missed = int((time_passed - self.deadline_sec) * self.datarate)
        else:
            missed = 0

        self.accumulated = int(self.datarate * time_passed) + self.accumulated - missed
        if bs is None:
            while self.accumulated == 0:
                time_passed = time.time() - self.last_query_time
                self.accumulated = int(self.datarate * time_passed)
        else:
            while self.accumulated < bs:
                time_passed = time.time() - self.last_query_time
                self.accumulated = int(self.datarate * time_passed)
        # we have decided how much has passed, therefore mark the time here, not later
        self.last_query_time = time.time()

        # means greedy selection
        if bs is None:
            to_serve = self.accumulated
        else:
            to_serve = bs
        # if to_serve > 5:
        #     print(accumulation, missed, time_passed, self.datarate)
        
        # to_serve = 1
        # missed = accumulation - to_serve

        # missed = int((time_passed - self.deadline_sec) * self.datarate)
        # print(accumulation, missed, time_passed)
        # print(time_passed, accumulation, missed, to_serve)

        self.index = (self.index + missed) % len(self.dataset)
        # print(self.index, missed, to_serve, time_passed)
        final_index = self.index + to_serve

        # TODO:
        # think about more precise deadline list
        # current assumption is that the last accumulated has completed arrival on last_query_time
        # which may not be the case, it may be in self.last_query_time - [0, 1/self.datarate)
        deadlines = [(self.last_query_time - i/self.datarate + self.deadline_sec) for i in range(to_serve - 1, -1, -1)]
        if final_index < self.index:
            data = (torch.hstack(self.dataset[self.index:final_index][0], self.dataset_copy2[:final_index][0]),
                    torch.hstack(self.dataset[self.index:final_index][1], self.dataset_copy2[:final_index][1]),
                    torch.hstack(self.dataset[self.index:final_index][2], self.dataset_copy2[:final_index][2]))
        else:
            # print(len(self.dataset[self.index:final_index]), self.dataset[self.index:final_index][0].shape)
            data = (self.dataset[self.index:final_index][0], self.dataset[self.index:final_index][1], self.dataset[self.index:final_index][2])
        # print(len(data), data[0].shape)

        self.transmitted += self.accumulated
        self.missed += missed
        self.index = final_index
        # update accumulated data for next extraction
        self.accumulated -= to_serve
        # print(len(data), data[0][0].shape)
        return data, deadlines, missed
    
    def get_perf(self):
        return self.transmitted, self.transmitted - self.missed, self.missed