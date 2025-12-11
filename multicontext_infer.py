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

import multicontext_parameters

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

# for checkpoint overhead experiment
model_load_spenttime_list = []


def multicontext_inferonly_process(
        model, teststream:ptychonn.shm_datareader.SHMInferDataReader,
        datarate:float, start_timestamp:float, time_limit:float, cur_ipriteration:int,
        inferdatalist_fileobj, logger:logfast.fastlogger.FastLogger):

    logger.log("MULTICONTEXT BEGIN")

    # to store training related metrics
    total_consumed = 0
    total_missed = 0
    total_iter_count = 0
    # this is not needed I kept it from the beginning that's why not want to remove
    metrics = {}
    
    iteration_start_time = start_timestamp
    last_consumption_time = start_timestamp
    inference_iter_count = 0
    ipriteration_no = 0

    # which directory to load model from at the beginning
    chkpt_dir = multicontext_parameters.MULTICONTEXT_IPRITER_MODEL_DIRNAME_FMT.format(cur_ipriteration)
    # what will be the next model to load
    next_model = 0

    while time.time() - start_timestamp < time_limit and total_consumed < len(teststream):
        # measure how much in the queue based on time
        inferbs = min(
            ptychonn.parameters.INFERENCE_BATCH_SIZE, 
            int(math.floor(datarate * (time.time() - last_consumption_time)))
        )
        logger.log("STREAM ACCUMULATED COUNT", inferbs, last_consumption_time)
        # print(inferbs, total_consumed, len(teststream))
        # inferbs = ptychonn.parameters.INFERENCE_BATCH_SIZE
        # while int(math.floor(datarate * (time.time() - last_consumption_time))) < inferbs and time.time() - start_timestamp < time_limit:
        #     pass
        # print(len(teststream)-total_consumed, inferbs)
        if inferbs > 0:
            try:
                infer_batch, consumed, missed, inferidxlist = teststream.read(
                    bs=min(inferbs, len(teststream) - total_consumed), blocking_call=True
                )

                if infer_batch is not None:
                    infer_count = infer_batch.shape[0]
                    total_missed += missed
                    total_consumed += infer_count
            except Exception as e:
                print(e)
                continue

        if infer_count == 0:
            continue

        last_consumption_time = time.time()
        forward_pass_arrival_time = last_consumption_time
        logger.log("MULTICONTEXT INFER READ LATENCY", last_consumption_time - iteration_start_time)
        # move the infer data to GPU
        ft_images = torch.tensor(infer_batch).to("cuda")

        logger.log("MULTICONTEXT INFER BS", infer_count)
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
        model_load_time = time.time()
        
        if ptychonn.ipc.exist_shm(ptychonn.parameters.SHM_MARKER_FMT_IPRINTERVAL_END.format(cur_ipriteration + 1)):
            cur_ipriteration += 1
            next_model = 0
            logger.log("MULTICONTEXT IPR ITERATION INCREMENT TO", cur_ipriteration)
            chkpt_dir = multicontext_parameters.MULTICONTEXT_IPRITER_MODEL_DIRNAME_FMT.format(cur_ipriteration)
        
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

        tmp = time.time()
        logger.log("ITERATION TIME", tmp - iteration_start_time)
        iteration_start_time = tmp

    if total_consumed >= len(infer_datareader):
        logger.log("ALL INFER DATA CONSUMED")

    logger.log("TOTAL CONSUMED", total_consumed)

    return metrics, total_consumed



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
    # GPU environment is assumend
    model.to("cuda")

    # other variants are just for performance test
    if args.model_type == "1.25M":
        if os.path.exists(os.path.join("pretrained_model", "pretrained_bestmodel.pth")):
            model = torch.load(os.path.join("pretrained_model", "pretrained_bestmodel.pth"), weights_only=False)
        else:
            print("Pretrained Model Not Found...Exiting")
            exit()

    _, _, _, nn_uf, nn_ub = ptychonn.model.benchmark_model(model)

    # initiate the logger
    logger = logfast.fastlogger.FastLogger()
    
    # make a result directory where generated images will be stored
    ptychonn.ipc.create_shm_folder(ptychonn.parameters.SHM_MARKER_NNRES_FOLDER)

    # initiate the file name to log down which data got consumed for what
    inferdatalist_file = open(
        "/dev/shm/inferdatalist_multicontext_{0}_{1}_{2}_{3}.csv".format(
            args.interval_count, args.interval_duration, args.datarate, int(args.ipr_throughput)), "w") 

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
    deadline_sec = args.deadline / 1000

    if args.large_dataset:
        total_runtime = args.interval_count * args.interval_duration
    else:
        # first interval data is used to pretrain the model
        total_runtime = args.interval_count * args.interval_duration

    # wait to synchronize time calculation with produce process
    start_timestamp, total_runtime = ptychonn.ipc.producer_transmit_wait()
    current_time = start_timestamp
    infersize = int(args.datarate * total_runtime)
    # init the data reader
    infer_datareader = ptychonn.shm_datareader.SHMInferDataReader(
        start_timestamp=start_timestamp, datarate=args.datarate,
        deadline_sec=args.deadline/1000, stream_alive_time=total_runtime
    )
    total_consumed = 0
    # to check which ipr iteration is finished, to identify appropriate model directory
    # model directory is named based on finished ipr iteration number
    # will start with 0 as no iteration is finished at the beginning
    cur_ipriteration = 0

    logger.log("MULTICONTEXT CONSUMPTION START", start_timestamp)
    
    while current_time - start_timestamp < total_runtime:
        infer_datareader.cur_readidx = total_consumed
        # set the reader length for the unipipe call
        # to handle initial boundary condition
        infer_datareader.set_len(infersize - total_consumed)
        # for inference location on datastream repositioning
        logger.log("INFER DATAREADER STATUS", infer_datareader.cur_readidx, infersize)
        
        # put unipipe traininfer for one ipriteration data here
        metrics, consumed = multicontext_inferonly_process(
            model, infer_datareader, cur_ipriteration=cur_ipriteration,
            inferdatalist_fileobj=inferdatalist_file, start_timestamp=start_timestamp,
            logger=logger, datarate=args.datarate, time_limit=total_runtime)
        
        # log how much ipr iteration matches with unipipe iteration

        current_time = time.time()
        total_consumed += consumed
        logger.log("MULTICONTEXT INTERIM TOTAL CONSUMED", total_consumed)


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
    if len(model_load_spenttime_list) > 0:
        logger.log(
            "MODEL RESTORE OVERHEAD (MIN/AVG/MAX)", min(model_load_spenttime_list),
            sum(model_load_spenttime_list)/len(model_load_spenttime_list),
            max(model_load_spenttime_list)
        )

    logger.persist(args.csvlog_file[:-4] + "_infer.log")
