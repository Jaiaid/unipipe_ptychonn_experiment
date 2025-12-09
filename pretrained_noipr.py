"""
 We assume only difference with unipipe is that no model training

 Objectives:
 1. Collect lost inference rate at each interval
 2. Collect mean inference accuracy, assume missed inference as 0 accuracy
"""

import argparse
import os
import math
import random
import copy
import time
import traceback
import torch
import numpy as np
import torch.utils
import torch.utils.data
import sklearn.metrics

import ptychonn.perf_model
import ptychonn.model
import ptychonn.dataset
import ptychonn.parameters
import ptychonn.ipc
import ptychonn.process_funcs
import ptychonn.error_calculation
import ptychonn.datastream
import ptychonn.shm_datareader

# for logging
import logfast.fastlogger


def pretrained_inferonly_process(
        model, teststream:ptychonn.shm_datareader.SHMInferDataReader,
        inferdatalist_fileobj, ipriteration_no,
        logger:logfast.fastlogger.FastLogger, datarate:float, time_limit=None):

    logger.log("PRETRAINED BEGIN")

    start_time = time.time()
    
    # to store training related metrics
    total_consumed = 0
    total_missed = 0
    total_iter_count = 0
    # this is not needed I kept it from the beginning that's why not want to remove
    metrics = {}
    # got from profile data
    inferbs = 64
    inference_iter_count = 0
    iteration_start_time = time.time()
    
    while time.time() - start_time < time_limit and total_consumed < len(teststream):
        logger.log("PRETRAINED ITERATION START", total_iter_count)
        # first take from test
        infer_count = 0

        try:
            infer_batch, consumed, missed, inferidxlist = teststream.read(
                bs=min(inferbs, len(teststream) - total_consumed))

            if infer_batch is not None:
                infer_count = infer_batch.shape[0]
                inference_iter_count += 1
                total_missed += missed
                total_consumed += infer_count
        except Exception as e:
            print(e)
            continue


        forward_pass_arrival_time = time.time()
        # move the infer data to GPU
        ft_images = torch.tensor(infer_batch).to("cuda")

        logger.log("PRETRAINED INFER BS", infer_count)
        # to keep track how many infer request missed due to forward pass latency
        pred_amps, pred_phs = model(ft_images) #Forward pass
        forward_pass_done_time = time.time()

        if infer_count > 0:
            pred_amps_cpu_np = pred_amps.cpu().detach().numpy()
            pred_ph_cpu_np = pred_phs.cpu().detach().numpy()
            # print(pred_amps.shape, pred_phs.shape)
            for i in range(infer_count):
                inferdatalist_fileobj.write("{0},{1}\n".format(inferidxlist[i], ipriteration_no))
                ptychonn.ipc.create_shm_data(
                    os.path.join(
                        ptychonn.parameters.SHM_MARKER_NNRES_FOLDER,
                        ptychonn.parameters.SHM_MARKER_NNRES_PHASE_NAMEFMT.format(inferidxlist[i])
                    ),
                    pred_ph_cpu_np[i]
                )
                ptychonn.ipc.create_shm_data(
                    os.path.join(
                        ptychonn.parameters.SHM_MARKER_NNRES_FOLDER,
                        ptychonn.parameters.SHM_MARKER_NNRES_AMP_NAMEFMT.format(inferidxlist[i])
                    ),
                    pred_amps_cpu_np[i]
                )

        # update total missed count
        logger.log("FORWARD PASS TOOK(sec.)", forward_pass_done_time - forward_pass_arrival_time)
        total_iter_count += 1
        # busy wait to ensure enough data accumulated
        # while ptychonn.parameters.INFERENCE_BATCH_SIZE/datarate > time.time() - iteration_start_time:
        #     pass
        logger.log("ITERATION TIME", time.time() - iteration_start_time)
        iteration_start_time = time.time()

    if total_consumed >= len(infer_datareader):
        logger.log("ALL INFER DATA CONSUMED")

    logger.log("TOTAL CONSUMED", total_consumed)

    return metrics, total_consumed

# signal producer to indicate finish of initiation
# then it will wait for transmission start
# this is part of mechanism to synchronize start of transmission and processing
def signal_producer():
    # need to signal for phase retrieval init finish
    # to keep the illusion that it is still two consumer one producer workflow
    ptychonn.ipc.create_shm_marker(ptychonn.parameters.SHM_MARKER_IPR_INIT_FINISH)
    ptychonn.ipc.create_shm_marker(ptychonn.parameters.SHM_MARKER_ML_INIT_FINISH)

# blocking function to wait for producer to start transmission
# this is part of mechanism to synchronize start of transmission and processing
def producer_transmit_wait():
    while not ptychonn.ipc.exist_shm(ptychonn.parameters.SHM_MARKER_TRANSMIT_START):
        pass
    
    # for timestamp sync and stream length
    with open("/dev/shm/{0}".format(ptychonn.parameters.SHM_MARKER_TIMESTAMP_SYNC), "r") as fd:
        marker_content = fd.read()
        start_timestamp = float(marker_content.split("\n")[0])
        stream_length = float(marker_content.split("\n")[1])

    return start_timestamp, stream_length


if __name__ == "__main__":
    # for reproducability
    # https://discuss.pytorch.org/t/training-reproducibility-problem/37143/3
    # https://vandurajan91.medium.com/random-seeds-and-reproducible-results-in-pytorch-211620301eba
    random.seed(ptychonn.parameters.SEED)
    torch.manual_seed(ptychonn.parameters.SEED)
    torch.cuda.manual_seed(ptychonn.parameters.SEED)
    torch.cuda.manual_seed_all(ptychonn.parameters.SEED)
    np.random.seed(ptychonn.parameters.SEED)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    # define arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--interval-duration", "-idur", type=int, required=True, help="length of interval in seconds")
    arg_parser.add_argument("--interval-count", "-icount", type=int, required=True, help="number of interval")
    arg_parser.add_argument("--datarate", "-drate", type=int, required=True, help="request/datasample per second")
    arg_parser.add_argument("--deadline", "-dead", type=int, required=True, help="each request deadline after arrival in millisecond")
    arg_parser.add_argument("--constant-bs", "-constbs", action="store_true", help="if constant batch size will be used")
    arg_parser.add_argument("--inferbs", "-inferbs", type=int, default=ptychonn.parameters.INFERENCE_BATCH_SIZE,  help="if constant inference batch size will be used what will be the value")
    arg_parser.add_argument("--trainbs", "-trainbs", type=int, default=ptychonn.parameters.TRAIN_BATCH_SIZE, help="if constant train batch size will be used what will be the value")
    arg_parser.add_argument("--ipr-throughput", "-iprt", type=float, default=None, help="IPR process throughput")
    arg_parser.add_argument("--csvlog-file", "-csvlog", type=str, required=True, help="name of csv log file")
    arg_parser.add_argument("--skip-line-pretrained", "-skipline", type=int, help="how many data to skip as model is pretrained on it")
    arg_parser.add_argument("--large-dataset", "-largedataset", action="store_true", help="if larger dataset will be ysed")
    arg_parser.add_argument("--model-type", "-type", type=str, choices=["1.25M", "5M", "10M", "20M", "100M", "200M"], help="which model to choose", default="1.25M")
    
    # get the arguments
    args = arg_parser.parse_args()

    # init the model
    model = ptychonn.model.get_model(type_name=args.model_type)
    _, _, _, nn_uf, nn_ub = ptychonn.model.benchmark_model(model)
    # other variants are just for performance test
    if args.model_type == "1.25M":
        if os.path.exists(os.path.join("pretrained_model", "pretrained_bestmodel.pth")):
            model = torch.load(os.path.join("pretrained_model", "pretrained_bestmodel.pth"), weights_only=False)
        else:
            print("Pretrained Model Not Found...Exiting")
            exit()
    # GPU environment is assumend
    model.to("cuda")

    # initiate the logger
    logger = logfast.fastlogger.FastLogger()
    
    # make a result directory where generated images will be stored
    ptychonn.ipc.create_shm_folder(ptychonn.parameters.SHM_MARKER_NNRES_FOLDER)

    # initiate the file name to log down which data got consumed for what
    inferdatalist_file = open(
        "/dev/shm/inferdatalist_pretrained_{0}_{1}_{2}_{3}.csv".format(
            args.interval_count, args.interval_duration, args.datarate, int(args.ipr_throughput)), "w") 

    # init the data reader
    infer_datareader = ptychonn.shm_datareader.SHMInferDataReader()
    # infer_datareader.set_len(args.datarate * args.interval_duration)

    # warmup run
    warmup_start_time = time.time()
    # put unipipe traininfer for one ipriteration data here
    # metrics, consumed = pretrained_inferonly_process(
    #     model, infer_datareader,
    #     inferdatalist_fileobj=inferdatalist_file, ipriteration_no=0, 
    #     logger=logger, datarate=args.datarate, time_limit=args.deadline/1000)
    logger.log("Warmup Run took {0}s".format(time.time() - warmup_start_time))
    
    # training state controller variable initiation
    cur_ipriteration = -1
    deadline_sec = args.deadline / 1000
    # estimate ipriteration time limit from perf. model
    # not neeeded to iterate as no ground truth generation involved
    # we are still doing it just to reuse code from pretrained with unipipe scheduling
    # we will just keep trainsize to 0
    ipriter_time_limit = ptychonn.perf_model.estimate_T_IPR_pretrained(
        phase_retrieval_genrate=args.ipr_throughput, acquisition_rate=args.datarate,
        deadline_sec=deadline_sec, nn_uf=nn_uf, nn_ub=0
    )

    if args.large_dataset:
        total_runtime = args.interval_count * args.interval_duration
    else:
        # first interval data is used to pretrain the model
        total_runtime = args.interval_count * args.interval_duration

    # signal producer that done, needed if initiation become expensive
    ptychonn.ipc.signal_producer_from_ML_surrogate()
    # wait to synchronize time calculation with produce process
    start_time, total_runtime = ptychonn.ipc.producer_transmit_wait()
    current_time = start_time
    cur_interval_start_time = current_time

    # to give producer time to put first data
    time.sleep(1/args.datarate)

    print("pretrained consumption start ", time.time())
    while current_time - start_time < total_runtime:
        current_time = time.time()


        ipriter_time_start = time.time()
        cur_ipriteration += 1
        logger.log("IPR ITERATION START", cur_ipriteration)

        # update the current inference idx and training data idx
        # the files are named in such a way that
        # t1 = time.time()
        # infer_datareader.reposition()
        # print("first infer data selection takes {0}s".format(time.time() - t1))

        # args.ipr_throughput * deadline_sec / (args.datarate - args.ipr_throughput)
        trainsize = 0
        infersize = int(math.floor(ipriter_time_limit * args.datarate)) # same as args.ipr_throughput * deadline_sec

        # for inference location on datastream repositioning
        train_readidx_curpos = (cur_ipriteration-1)*(trainsize + infersize)
        logger.log("INFER DATAREADER STATUS", infer_datareader.cur_readidx, train_readidx_curpos, train_readidx_curpos - infersize + 1, infersize)
        infer_datareader.cur_readidx = train_readidx_curpos + trainsize
        # set the reader length for the unipipe call
        # to handle initial boundary condition
        infer_datareader.set_len(infersize if infer_datareader.cur_readidx >= 0 else 0)
        
        # put unipipe traininfer for one ipriteration data here
        metrics, consumed = pretrained_inferonly_process(
            model, infer_datareader,
            inferdatalist_fileobj=inferdatalist_file, ipriteration_no=cur_ipriteration, 
            logger=logger, datarate=args.datarate, time_limit=ipriter_time_limit)
        
        # log how much ipr iteration matches with unipipe iteration

        # busy wait until time is passed
        while time.time() - ipriter_time_start < ipriter_time_limit:
            pass


    # postmortem of data, calculate error
    amp_error, ph_error, nn_amp_error, nn_ph_error = ptychonn.error_calculation.postsimulation_error_calc(
        skip_line=args.skip_line_pretrained, large_dataset=args.large_dataset
    )

    with open(args.csvlog_file, "w") as fout:
        # amp error, ph error, nn amp error, nn ph error
        fout.write("{0},{1},{2},{3}\n".format(amp_error, ph_error, nn_amp_error, nn_ph_error))

    inferdatalist_file.close()
    logger.persist(args.csvlog_file[:-4] + ".log")
