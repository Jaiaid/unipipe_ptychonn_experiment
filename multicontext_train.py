"""
 We assume training will be done with full overhead but through unipipe
 We will keep serving inference in same context but partially trained model
 
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

import multicontext_parameters
# for logging
import logfast.fastlogger


global MAX_INFER_BATCH_SIZE
# for checkpoint overhead experiment
model_save_spenttime_list = []


def multicontext_train(model, trainloader:ptychonn.shm_datareader.SHMTrainDataReader,
                       epoch_count, datarate, deadline_sec,
                       traindatalist_fileobj, ipriteration_no,  
                       logger:logfast.fastlogger.FastLogger, trainbs=64, chkpt_dir=None, time_limit=None):

    logger.log("MULTICONTEXT TRAIN BEGIN")
    logger.log("MULTICONTEXT TRAIN DATASET SIZE", len(trainloader))
    logger.log("MULTICONTEXT TRAIN BATCH SIZE", trainbs)

    start_time = time.time()
    interval_init_time = time.time()
    interval_remaining_time = time_limit
    chkpt_process = None

    # taken from paper's code
    # if optimizer_objects is None:

    # print("training mechanism creation takes ", time.time() - init_time)
    # to store training related metrics
    metrics = {"lrs": [], "losses": [], "val_losses": [], "best_val_loss": math.inf}

    # for loss measure and some stats
    total_iter_count = 0

    # list down the train data idx
    for i in trainloader.get_dataidxlist():
        traindatalist_fileobj.write("{0},{1}\n".format(i, ipriteration_no))

    # iter_creation_start_time = time.time()
    # print("iterator creation time:", time.time() - iter_creation_start_time)
    # print("intialization time:", time.time() - interval_init_time)

    train_start_time = time.time()
    # to control when the training of current interval will stop
    previous_loss = math.inf
    # if epochs are finished or convergence happen we stop train but inference continues
    stop_train = False
    next_model = 0
    total_traindata_consumed = 0
    # Temporary
    # train_batch_large = torch.zeros(64, 1,64,64)
    # target_phase_large = torch.zeros(64, 1,64,64)
    # target_amp_large = torch.zeros(64, 1,64,64)
    #
    # arbitrary large epoch, for coding ease in tracking an epoch
    # actual breaking condition is on time limit and loss
    for cur_epoch in range(epoch_count):
        logger.log("MULTICONTEXT TRAIN EPOCH BEGIN", cur_epoch + 1)

        tot_loss = 0.0
        loss_amp = 0.0
        loss_ph = 0.0
        total_iter_count = 0
        epoch_consumed = 0

        # to get iteration time
        iteration_time = 0
        if time_limit is not None and time.time() - start_time > time_limit or stop_train:
            logger.log("MULTICONTEXT TRAIN EPOCH END DUE TO TIME LIMIT", cur_epoch + 1)
            break


        while time_limit is not None and time.time() - start_time < time_limit and epoch_consumed < len(trainloader):
            # logger.log("MULTICONTEXT TRAIN ITERATION START", total_iter_count, trainbs, len(trainloader) - epoch_consumed, len(trainloader))
            iteration_start_time = time.time()
            
            train_batch = trainloader.read(bs=min(trainbs, len(trainloader) - epoch_consumed))
            if train_batch[0] is None:
                continue
            # Temporary
            # train_batch_large[0:] = torch.tensor(train_batch[0])
            # train_batch = train_batch_large
            #
            # some infer data is there, merge and pass to context
            # or training is done now to pass only infer data to context
            # or no infer data is in pipeline for now, so only training
            ft_images = torch.tensor(train_batch[0]).to("cuda")

            logger.log("MULTICONTEXT TRAIN BS", train_batch[0].shape[0])
            # Forward pass
            pred_amps, pred_phs = model(ft_images) 
            forward_pass_done_time = time.time()

            # update total missed count
            logger.log("FORWARD PASS TOOK(sec.)", forward_pass_done_time - iteration_start_time)

            # Temoporary
            # target_amp_large[0:] = torch.tensor(train_batch[1])
            # target_phase_large[0:] = torch.tensor(train_batch[2])
            # gt_amps = target_amp_large.to("cuda")
            # gt_phs = target_phase_large.to("cuda")
            # 
            gt_amps = torch.tensor(train_batch[1]).to("cuda")
            gt_phs = torch.tensor(train_batch[2]).to("cuda")

            backward_pass_arrival_time = time.time()
            #Compute losses
            loss_a = criterion(pred_amps, gt_amps) #Monitor amplitude loss
            loss_p = criterion(pred_phs, gt_phs) #Monitor phase loss but only within support (which may not be same as true amp)
            loss = loss_a + loss_p #Use equiweighted amps and phase

            #Zero current grads and do backprop
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()

            l_a = loss_a.detach().item()
            l_p = loss_p.detach().item()
            loss_amp += l_a
            loss_ph += l_p
            tot_loss += l_a + l_p
            scheduler.step()

            iter_end_timestamp = time.time()
            logger.log("BACKWARD TAKES(sec.)", iter_end_timestamp - backward_pass_arrival_time)

            metrics['lrs'].append(scheduler.get_last_lr())
            logger.log("ITER_COUNT, TRAIN LOSS", total_iter_count, l_a + l_p, l_a, l_p)

            total_iter_count += 1
            logger.log("ITERATION TAKES(sec.)", iter_end_timestamp - iteration_start_time)
            iteration_time += iter_end_timestamp - iteration_start_time
            epoch_consumed += train_batch[0].shape[0]

        logger.log("TRAINDATASET CONSUMED AT EPOCH", cur_epoch + 1, epoch_consumed, len(trainloader))
        total_traindata_consumed += epoch_consumed

        #Divide cumulative loss by number of batches-- sli inaccurate because last batch is different size
        metrics['losses'].append([tot_loss/(total_iter_count + 1),loss_amp/(total_iter_count + 1),loss_ph/(total_iter_count + 1)])
        logger.log("TRAINING LOSS AT EPOCH", cur_epoch, tot_loss/(total_iter_count + 1),loss_amp/(total_iter_count + 1),loss_ph/(total_iter_count + 1))
        
        # update flag to stop training if training brings out minimal improvement
        # this condition needed because it is possible in current epoch not a single training is run
        if abs(previous_loss - tot_loss/(total_iter_count + 1)) < ptychonn.parameters.LOSS_CHANGE_MIN_THRESHOLD:
            stop_train = True
            logger.log("MULTICONTEXT TRAIN STOP CONVERGENCE")
            interval_remaining_time = time.time() - interval_init_time
            logger.log("MULTICONTEXT TRAIN INTERVAL REM. TIME", interval_remaining_time)

        # save model if loss is lower than before
        # if tot_loss / (total_iter_count + 1) < previous_loss:
        if True:
            model_save_start_time = time.time()
            # don't create new process before previous one is done
            data_queue.put((
                chkpt_dir,
                copy.deepcopy(model).to("cpu"),
                next_model
            ))
            # ptychonn.process_funcs.update_saved_model(
            #     model=model,
            #     path=os.path.join(
            #         "/dev/shm/", chkpt_dir
            #     ),
            #     name=ptychonn.multicontext_parameters.MULTICONTEXT_IPRITER_MODELNAME_FMT.format(next_model)
            # )
            # ptychonn.ipc.create_shm_marker(
            #     os.path.join(chkpt_dir, ptychonn.multicontext_parameters.MULTICONTEXT_SHM_MARKER_IPRITER_END.format(next_model)))
            # ptychonn.ipc.create_shm_marker(
            #     os.path.join(chkpt_dir, ptychonn.multicontext_parameters.MULTICONTEXT_SHM_MARKER_IPRITER_END.format(next_model)+str(time.time())))

            taken_time = time.time() - model_save_start_time
            model_save_spenttime_list.append(taken_time)

            logger.log(
                "CREATING CHECKPOINT",
                os.path.join(chkpt_dir, ptychonn.multicontext_parameters.MULTICONTEXT_SHM_MARKER_IPRITER_END.format(next_model)),
                os.path.join(chkpt_dir, ptychonn.multicontext_parameters.MULTICONTEXT_IPRITER_MODELNAME_FMT.format(next_model))
            )
            next_model += 1
            previous_loss = tot_loss/(total_iter_count + 1)

        logger.log("MULTICONTEXT TRAIN ITER. TIME", iteration_time/(total_iter_count + 1))
        logger.log("MULTICONTEXT TRAIN EPOCH END", cur_epoch + 1)
        if cur_epoch + 1 == epoch_count:
            stop_train = True
            logger.log("MAX EPOCH DONE, NO TRAINING IN CURRENT INTERVAL")

    logger.log("MULTICONTEXT TRAIN TAKES", time.time() - train_start_time)
    logger.log("MULTICONTEXT TRAIN CONSUME", total_traindata_consumed)
    logger.log("MULTICONTEXT TRAIN TOTAL TIME:", time.time() - start_time)

    return metrics, cur_epoch

# signal producer to indicate finish of initiation
# then it will wait for transmission start
# this is part of mechanism to synchronize start of transmission and processing
def signal_producer():
    ptychonn.ipc.create_shm_marker(ptychonn.parameters.SHM_MARKER_ML_INIT_FINISH)

# blocking function to wait for producer to start transmission
# this is part of mechanism to synchronize start of transmission and processing
def producer_transmit_wait():
    while not ptychonn.ipc.exist_shm(ptychonn.parameters.SHM_MARKER_TRANSMIT_START):
        pass
    # for timestamp sync
    with open("/dev/shm/{0}".format(ptychonn.parameters.SHM_MARKER_TIMESTAMP_SYNC), "r") as fd:
        marker_content = fd.read()
        start_timestamp = float(marker_content)

    return start_timestamp


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
    arg_parser.add_argument("--gtdefault", "-gtd", action="store_true", help="what to take as default response for missed request")
    arg_parser.add_argument("--allckpttest", "-ckpttest", action="store_true", help="if all checkpoints will be saved and tested with inference data")
    arg_parser.add_argument("--constant-bs", "-constbs", action="store_true", help="if constant batch size will be used")
    arg_parser.add_argument("--inferbs", "-inferbs", type=int, default=ptychonn.parameters.INFERENCE_BATCH_SIZE,  help="if constant inference batch size will be used what will be the value")
    arg_parser.add_argument("--trainbs", "-trainbs", type=int, default=ptychonn.parameters.TRAIN_BATCH_SIZE, help="if constant train batch size will be used what will be the value")
    arg_parser.add_argument("--csvlog-file", "-csvlog", type=str, required=True, help="name of csv log file")
    arg_parser.add_argument("--iprfrac", "-iprfrac", type=float, default=None, help="what portion of training data will come from IPR")
    arg_parser.add_argument("--ipr-throughput", "-iprt", type=float, default=None, help="IPR process throughput")
    arg_parser.add_argument("--inffrac", "-inffrac", type=float, default=1.0, help="how much factor to multiply with infer bs")
    arg_parser.add_argument("--skip-line-pretrained", "-skipline", type=int, help="how many data to skip as model is pretrained on it")
    arg_parser.add_argument("--large-dataset", "-largedataset", action="store_true", help="if larger dataset will be ysed")
    arg_parser.add_argument("--model-type", "-type", type=str, choices=["1.25M", "5M", "10M", "20M", "100M", "200M"], help="which model to choose", default="1.25M")
    arg_parser.add_argument("--gtcount", "-gtcount", type=int, required=False, help="how many ground truth will be consumed by phase retrieval process", default=1)
    arg_parser.add_argument("--maxinfer-bs", "-maxinferbs", type=int, default=None,  help="what is the max infer batch size to use, if not set use perf model to decide")
    
    # get the arguments
    args = arg_parser.parse_args()

    # init the model
    model = ptychonn.model.get_model(type_name=args.model_type)
    # from profiling data
    # forward pass tuned for latency
    # backward pass tuned for throughput
    nn_uf = 0.0023
    nn_ub = 0.00027
    
    if args.model_type in ["1.25M", "5M", "10M", "20M"]:
        model_path = os.path.join(
            "pretrained_model", "pretrained_bestmodel_{0}.pth".format(
                ptychonn.model.get_model_name_from_type(type_name=args.model_type)
            )
        )
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, weights_only=True))
        else:
            print("Pretrained Model Not Found...Exiting")
            exit()
    # GPU environment is assumed
    model.to("cuda")
    model.train()

    # taken from paper's code
    # if optimizer_objects is None:
    iter_per_epoch = 1
    step_size = 6
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr = ptychonn.parameters.LR)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=ptychonn.parameters.LR/10, max_lr=ptychonn.parameters.LR,
        step_size_up=step_size, cycle_momentum=False, mode='triangular2')

    _, _, _, nn_uf, nn_ub = ptychonn.model.benchmark_model(model)

    # initiate the logger
    logger = logfast.fastlogger.FastLogger()

    # initiate the file name to log down which data got consumed for what
    traindatalist_file = open(
        "/dev/shm/traindatalist_multicontext_{0}_{1}_{2}_{3}.csv".format(
            args.interval_count, args.interval_duration, args.datarate, int(args.ipr_throughput)), "w")

    # init the data reader
    train_datareader = ptychonn.shm_datareader.SHMTrainDataReader()

    # warmup run
    # warmup_start_time = time.time()
    # metrics, epoch_count = multicontext_train(
    #     model, train_datareader, epoch_count=3,
    #     trainbs=1, datarate=args.datarate, deadline_sec=args.deadline/1000,
    #     traindatalist_fileobj=traindatalist_file, ipriteration_no=0,
    #     chkpt_dir="MODEL_MULTICONTEXT_dummy",
    #     logger=logger, time_limit=args.deadline/1000)
    # logger.log("Warmup Run took {0}s".format(time.time() - warmup_start_time))

    forward_time_per_sample = []
    backward_time_per_sample = []
    # read profile data to get
    with open(f"ptychonn/benchmark_{ptychonn.model.get_model_name_from_type(type_name=args.model_type)}_nn_step.csv") as f:
        for line in f.readlines()[1:]:
            tokens = line.split()
            bs = int(tokens[0])
            fwd_time_per_sample = float(tokens[4])
            bwd_time_per_sample = float(tokens[6])
            forward_time_per_sample.append(fwd_time_per_sample)
            backward_time_per_sample.append(bwd_time_per_sample)

    # training state controller variable initiation
    cur_ipriteration = 0
    interipr_model_idx = 0
    deadline_sec = args.deadline / 1000

    # estimate ipriteration time limit from perf. model
    # for coordination with ground truth data generation
    # although we are not training here, to make things fair with unipipe
    # we have to generate some ground truth data
    # ipriter_time_limit = ptychonn.perf_model.estimate_T_IPR(
    #     phase_retrieval_genrate=args.ipr_throughput, deadline_sec=deadline_sec,
    #     acquisition_rate=args.datarate, nn_uf=nn_uf, nn_ub=nn_ub
    # )
    global MAX_INFER_BATCH_SIZE
    if args.maxinfer_bs is not None:
        MAX_INFER_BATCH_SIZE = args.maxinfer_bs
        logger.log("SET MAXBS", MAX_INFER_BATCH_SIZE)
    else:
        MAX_INFER_BATCH_SIZE = ptychonn.perf_model.SystemTuner.calc_deadline_aware_maxthpt_bs(
            forward_time_per_sample, deadline_sec, args.datarate
        )
        logger.log("TUNED MAXBS", MAX_INFER_BATCH_SIZE)

    ipriter_time_limit, _ = ptychonn.perf_model.estimate_unipipe_schedule(
        phase_retrieval_genrate=args.ipr_throughput, deadline_sec=deadline_sec,
        acquisition_rate=args.datarate, ground_truth_count=args.gtcount,
        maxbs=MAX_INFER_BATCH_SIZE, forward_time_per_sample=forward_time_per_sample,
        backward_time_per_sample=backward_time_per_sample
    )

    if args.large_dataset:
        total_runtime = args.interval_count * args.interval_duration
    else:
        # first interval data is used to pretrain the model
        total_runtime = args.interval_count * args.interval_duration
    
    # # It's best practice to use 'spawn' as the start method for PyTorch multiprocessing
    # torch.set_num_threads(1)
    # try:
    #     torch.multiprocessing.set_start_method('spawn', force=True)
    # except RuntimeError:
    #     pass

    # start the checkpointing process to save model periodically
    # without incurring compute stall
    torch.multiprocessing.set_start_method('spawn')
    data_queue = torch.multiprocessing.Queue(maxsize=1000)

    checkpointing_process = torch.multiprocessing.Process(
        target=ptychonn.process_funcs.checkpointing_process_function,
        args=(data_queue,)
    )
    # so that it will close if main process is closed
    checkpointing_process.start()

    # signal producer that done, needed if initiation become expensive
    # doing it only for training process assuming inference initiation is faster
    # needs a better approach
    ptychonn.ipc.signal_producer_from_ML_surrogate()
    # wait to synchronize time calculation with produce process
    start_time, total_runtime = ptychonn.ipc.producer_transmit_wait()
    logger.log("MULTICONTEXT TRAIN CONSUMPTION START", start_time)
    current_time = start_time
    cur_interval_start_time = current_time

    trainbs = ptychonn.parameters.TRAIN_BATCH_SIZE
    # from profile data, tuned for throughput

    # to give producer time to put first data
    # time.sleep(1/args.datarate)

    while current_time - start_time < total_runtime:
        current_time = time.time()

        # checking for signal existance from IPR process
        # this progression needs to be done irrespective of interval
        # as IPR will keep running for data from interval 0 also (for which model is already trained)
        # it will indicate ground truth is gnereted for some data and IPR has moved from that portion
        # which means completion of SHM_MARKER_FMT_IPRINTERVAL_END.format(cur_ipriteration+1)
        ipr_training_time_start = start_time + cur_ipriteration*ipriter_time_limit
        
        if ptychonn.ipc.exist_shm(ptychonn.parameters.SHM_MARKER_FMT_IPRINTERVAL_END.format(cur_ipriteration)):
            logger.log("IPR ITERATION START", cur_ipriteration)
            # create the directory for saving model
            ptychonn.ipc.create_shm_folder(multicontext_parameters.MULTICONTEXT_IPRITER_MODEL_DIRNAME_FMT.format(cur_ipriteration))
            # print("first infer data selection takes {0}s".format(time.time() - t1))

            # t1 = time.time()
            # train_datareader.reposition()
            # start of unipipe initiation and call
            
            # ipriter_time_limit = deadline_sec #args.ipr_throughput * deadline_sec / (args.datarate - args.ipr_throughput)
            
            trainsize = int(math.floor(ipriter_time_limit * args.ipr_throughput))
            infersize = int(math.floor(ipriter_time_limit * (args.datarate - args.ipr_throughput)))
            train_readidx_curpos = cur_ipriteration*(trainsize + infersize)
            train_datareader.set_curipriteration(cur_ipriteration=cur_ipriteration)

            logger.log("TRAIN DATAREADER STATUS", train_readidx_curpos, trainsize, train_datareader.cur_ipriteration)

            train_datareader.set_len(begin=train_readidx_curpos, end=train_readidx_curpos+trainsize-1)
            epoch_count = ptychonn.parameters.EPOCHS
        else:
            continue
            # print("first train data selection takes {0}s".format(time.time() - t1))

        

        # log the performance model related states
        logger.log(
            "CURIPRITERATION,TIME_LIMIT,TRAIN_SIZE",
            cur_ipriteration, ipriter_time_limit, trainsize)

        # put unipipe traininfer for one ipriteration data here
        metrics, epoch_count = multicontext_train(
            model, train_datareader, epoch_count=epoch_count,
            trainbs=trainbs, datarate=args.datarate, deadline_sec=deadline_sec,
            traindatalist_fileobj=traindatalist_file, ipriteration_no=cur_ipriteration,
            chkpt_dir=multicontext_parameters.MULTICONTEXT_IPRITER_MODEL_DIRNAME_FMT.format(cur_ipriteration),
            logger=logger, time_limit=ipriter_time_limit-(time.time() - ipr_training_time_start)
        )
        # log how much ipr iteration matches with unipipe iteration
        logger.log(
            "CURIPRITERATION,EPOCH,TIME_LIMIT,ACTUAL_TIME", 
            cur_ipriteration, epoch_count, ipriter_time_limit, time.time() - ipr_training_time_start)

        # busy wait until time is passed
        while time.time() - ipr_training_time_start < ipriter_time_limit:
            pass
        cur_ipriteration = int((time.time() - start_time) // ipriter_time_limit)

    traindatalist_file.close()

    logger.log(
        "MODEL SAVE OVERHEADS", model_save_spenttime_list
    )
    if len(model_save_spenttime_list)>0:
        logger.log(
            "MODEL SAVE OVERHEAD (MIN/AVG/MAX)", min(model_save_spenttime_list),
            sum(model_save_spenttime_list)/len(model_save_spenttime_list),
            max(model_save_spenttime_list)
        )
    
    logger.persist(args.csvlog_file[:-4] + "_train.log")
    checkpointing_process.terminate()

    # doing at last, in case it hangs we will still have the log
    # terminate the output process by putting sentinel
    data_queue.put((None, None, None, None))
    checkpointing_process.join()

