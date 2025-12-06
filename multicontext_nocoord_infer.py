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

import ptychonn.perf_model
import ptychonn.model
import ptychonn.dataset
import ptychonn.parameters
import ptychonn.ipc
import ptychonn.process_funcs
import ptychonn.error_calculation
import ptychonn.datastream
import ptychonn.shm_datareader

# for multicontext specific paramters
import multicontext_parameters
# for logging
import logfast.fastlogger


# for checkpoint overhead experiment
model_load_spenttime_list = []


def multicontext_inferonly_process(
        model, teststream:ptychonn.shm_datareader.SHMInferDataReader,
        inferdatalist_fileobj, ipriteration_no,
        logger:logfast.fastlogger.FastLogger, datarate:float, time_limit=None, chkpt_dir="model_multicontext"):

    logger.log("MULTICONTEXT INFER BEGIN")

    start_time = time.time()
    
    # to store training related metrics
    total_consumed = 0
    total_iter_count = 0
    # this is not needed I kept it from the beginning that's why not want to remove
    metrics = {}
    inferbs = ptychonn.parameters.INFERENCE_BATCH_SIZE
    # from profiled data, tuned for latency
    inferbs = 64
    # next model indicates if the model which is being trained in separate context is loaded 
    # for completeion of <next_model> no. checkpoint
    next_model = 0
    iteration_start_time = time.time()
    
    while time.time() - start_time < time_limit and total_consumed < len(teststream):
        logger.log("MULTICONTEXT ITERATION START", total_iter_count)
        # first take from test
        infer_count = 0

        try:
            infer_batch, consumed, missed, inferidxlist = teststream.read(bs=min(inferbs, len(teststream) - total_consumed))
            if infer_batch is None:
                continue
            infer_count = infer_batch.shape[0]
            total_consumed += infer_count
            total_iter_count += 1
            logger.log("MULTICONTEXT INFER BS", infer_count)
        except Exception as e:
            print(e)
            continue

        forward_pass_arrival_time = time.time()
        # move the infer data to GPU
        ft_images = torch.tensor(infer_batch).to("cuda")
        # to keep track how many infer request missed due to forward pass latency
        pred_amps, pred_phs = model(ft_images) #Forward pass
        forward_pass_done_time = time.time()

        pred_amps_cpu_np = pred_amps.cpu().detach().numpy()
        pred_ph_cpu_np = pred_phs.cpu().detach().numpy()
        # print(pred_amps.shape, pred_phs.shape)
        if infer_count != len(inferidxlist):
            print(infer_count, len(inferidxlist))
        for i in range(len(inferidxlist)):
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

        # check if an updated model is there
        # if it is there load it
        model_load_time = time.time()
        if ptychonn.ipc.exist_shm(
            os.path.join(chkpt_dir, multicontext_parameters.MULTICONTEXT_SHM_MARKER_IPRITER_END.format(next_model))):
            cur_model_dir = os.path.join("/dev/shm", chkpt_dir)

            logger.log("MODEL UPDATE TO", chkpt_dir, next_model)
            # print("inference process is swapping model, ", os.path.join(cur_model_dir, multicontext_parameters.MULTICONTEXT_IPRITER_MODELNAME_FMT.format(next_model)))
            model = torch.load(
                os.path.join(
                    cur_model_dir,
                    multicontext_parameters.MULTICONTEXT_IPRITER_MODELNAME_FMT.format(next_model)
                ), weights_only=False
            )
            model.to("cuda")
            taken_time = time.time() - model_load_time
            logger.log("MODEL LOAD TAKES", taken_time)
            model_load_spenttime_list.append(taken_time)
            next_model += 1

        logger.log("ITERATION TIME", time.time() - iteration_start_time)
        iteration_start_time = time.time()

    logger.log("TOTAL CONSUMED", total_consumed)

    return metrics, total_consumed


# blocking function to wait for producer to start transmission
# this is part of mechanism to synchronize start of transmission and processing
def producer_transmit_wait():
    while not ptychonn.ipc.exist_shm(ptychonn.parameters.SHM_MARKER_TRANSMIT_START):
        pass


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
    arg_parser.add_argument("--allckpttest", "-ckpttest", action="store_true", help="if all checkpoints will be saved and tested with inference data")
    arg_parser.add_argument("--csvlog-file", "-csvlog", type=str, required=True, help="name of csv log file")
    arg_parser.add_argument("--skip-line-pretrained", "-skipline", type=int, help="how many data to skip as model is pretrained on it")
    arg_parser.add_argument("--large-dataset", "-largedataset", action="store_true", help="if larger dataset will be ysed")
    arg_parser.add_argument("--model-type", "-type", type=str, choices=["1.25M", "5M", "10M", "20M", "100M", "200M"], help="which model to choose", default="1.25M")
    
    # get the arguments
    args = arg_parser.parse_args()

    # initiate the logger
    logger = logfast.fastlogger.FastLogger()
    
    # make a result directory where generated images will be stored
    ptychonn.ipc.create_shm_folder(ptychonn.parameters.SHM_MARKER_NNRES_FOLDER)

    # initiate the file name to log down which data got consumed for what
    inferdatalist_file = open(
        "/dev/shm/inferdatalist_multicontext_{0}_{1}_{2}_{3}.csv".format(
            args.interval_count, args.interval_duration, args.datarate, int(args.ipr_throughput)), "w") 

    # init the model
    model = ptychonn.model.get_model(type_name=args.model_type)
    _, _, _, nn_uf, nn_ub = ptychonn.model.benchmark_model(model)
    # from profiling data
    # forward pass tuned for latency
    # backward pass tuned for throughput
    nn_uf = 0.00027
    nn_ub = 0.00027
    # other variants are just for performance test
    if args.model_type == "1.25M":
        if os.path.exists(os.path.join("pretrained_model", "pretrained_bestmodel.pth")):
            model = torch.load(os.path.join("pretrained_model", "pretrained_bestmodel.pth"), weights_only=False)
        else:
            print("Pretrained Model Not Found...Exiting")
            exit()
    # assumed GPU environment
    model.to("cuda")

    # init the data reader
    infer_datareader = ptychonn.shm_datareader.SHMInferDataReader()
    infer_datareader.set_len(args.datarate * args.interval_duration)

    # warmup run
    warmup_start_time = time.time()
    metrics, consumed = multicontext_inferonly_process(
            model, infer_datareader, 
            logger=logger, datarate=args.datarate, time_limit=args.deadline/1000,
            inferdatalist_fileobj=inferdatalist_file, ipriteration_no=0,
            chkpt_dir=multicontext_parameters.MULTICONTEXT_IPRITER_MODEL_DIRNAME_FMT.format(0))
    logger.log("Warmup Run took {0}s".format(time.time() - warmup_start_time))

    # wait to synchronize time calculation with produce process
    producer_transmit_wait()
    print("multicontext infer consumption start ", time.time())
    
    # training state controller variable initiation
    start_time = time.time()
    cur_ipriteration = -1
    cur_interval = 1
    current_time = start_time
    cur_interval_start_time = current_time
    deadline_sec = args.deadline / 1000

    # estimate ipriteration time limit from perf. model
    # for coordination with ground truth data generation
    # although we are not training here, to make things fair with unipipe
    # we have to generate some ground truth data
    ipriter_time_limit = ptychonn.perf_model.estimate_T_IPR(
        phase_retrieval_genrate=args.ipr_throughput, deadline_sec=deadline_sec,
        acquisition_rate=args.datarate, nn_uf=nn_uf, nn_ub=nn_ub
    )
    
    if args.large_dataset:
        total_runtime = args.interval_count * args.interval_duration
    else:
        # first interval data is used to pretrain the model
        total_runtime = args.interval_count * args.interval_duration

    total_consumed = 0

    # to give producer time to put first data
    # time.sleep(1/args.datarate)

    logger.log("INTERVAL START {0}".format(cur_interval))
    while current_time - start_time < total_runtime:
        current_time = time.time()
        if current_time - cur_interval_start_time > args.interval_duration:
            # mark of interval start
            logger.log("INTERVAL END {0}".format(cur_interval))
            cur_interval += 1
            cur_interval_start_time = current_time
            # mark of interval start
            logger.log("INTERVAL START {0}".format(cur_interval))

        # start of current interval processing
        # checking for signal existance from IPR process
        # this progression needs to be done irrespective of interval
        # as IPR will keep running for data from interval 0 also (for which model is already trained)
        # it will indicate ground truth is gnereted for some data and IPR has moved from that portion
        # which means completion of SHM_MARKER_FMT_IPRINTERVAL_END.format(cur_ipriteration+1)
        ipr_iteration_time_start = time.time()

        if ptychonn.ipc.exist_shm(ptychonn.parameters.SHM_MARKER_FMT_IPRINTERVAL_END.format(cur_ipriteration + 1)):
            cur_ipriteration += 1
            logger.log("IPR ITERATION START", cur_ipriteration)
            # update the current inference idx and training data idx
            # the files are named in such a way that
            # t1 = time.time()
            # infer_datareader.reposition()
            # print("first infer data selection takes {0}s".format(time.time() - t1))
            # ipriter_time_limit = deadline_sec# args.ipr_throughput * deadline_sec / (args.datarate - args.ipr_throughput)
            
            trainsize = int(math.floor(ipriter_time_limit * args.ipr_throughput))
            infersize = int(math.floor(ipriter_time_limit * (args.datarate - args.ipr_throughput))) # same as args.ipr_throughput * deadline_sec
            # for inference location on datastream repositioning
            train_readidx_curpos = (cur_ipriteration-1)*(trainsize + infersize)
            logger.log("INFER DATAREADER STATUS", infer_datareader.cur_readidx, train_readidx_curpos, train_readidx_curpos - infersize + 1, infersize)
            infer_datareader.cur_readidx = train_readidx_curpos
            # set the reader length for the unipipe call
            # to handle initial boundary condition
            infer_datareader.set_len(infersize if infer_datareader.cur_readidx >= 0 else 0)
        else:
            continue

        # put unipipe traininfer for one ipriteration data here
        metrics, consumed = multicontext_inferonly_process(
            model, infer_datareader, 
            logger=logger, datarate=args.datarate, time_limit=deadline_sec,
            inferdatalist_fileobj=inferdatalist_file, ipriteration_no=cur_ipriteration,
            chkpt_dir=multicontext_parameters.MULTICONTEXT_IPRITER_MODEL_DIRNAME_FMT.format(cur_ipriteration))
        
        total_consumed += consumed
        # log how much ipr iteration matches with unipipe iteration


    # postmortem of data, calculate error
    amp_error, ph_error, nn_amp_error, nn_ph_error = ptychonn.error_calculation.postsimulation_error_calc(
        skip_line=args.skip_line_pretrained, large_dataset=args.large_dataset
    )

    with open(args.csvlog_file, "w") as fout:
        # amp error, ph error, nn amp error, nn ph error
        fout.write("{0},{1},{2},{3}\n".format(amp_error, ph_error, nn_amp_error, nn_ph_error))

    inferdatalist_file.close()

    logger.log(
        "MODEL RESTORE OVERHEADS", model_load_spenttime_list
    )
    logger.log(
        "MODEL RESTORE OVERHEAD (MIN/AVG/MAX)", min(model_load_spenttime_list),
        sum(model_load_spenttime_list)/len(model_load_spenttime_list),
        max(model_load_spenttime_list)
    )
    logger.persist(args.csvlog_file[:-4] + "_infer.log")

    # print(
    #     "Model Restore Mean Overhead: {0}s".format(
    #         sum(model_load_spenttime_list)/len(model_load_spenttime_list)
    #     )
    # )
    # print("Model Restore Overheads: ", model_load_spenttime_list)
