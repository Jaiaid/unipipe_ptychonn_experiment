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


def multicontext_train(model, trainloader:ptychonn.shm_datareader.SHMTrainDataReader,
                       trainbs, epoch_count, datarate, deadline_sec, chkpt_dir,
                       logger:logfast.fastlogger.FastLogger, time_limit=None):

    logger.log("MULTICONTEXT TRAIN BEGIN")
    logger.log("MULTICONTEXT TRAIN DATASET SIZE", len(trainloader))
    logger.log("MULTICONTEXT TRAIN BATCH SIZE", trainbs)

    start_time = time.time()
    interval_init_time = time.time()
    interval_remaining_time = time_limit

    # taken from paper's code
    # if optimizer_objects is None:
    iter_per_epoch = np.floor(len(trainloader)/trainbs) + 1
    step_size = 6 * iter_per_epoch
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr = ptychonn.parameters.LR)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=ptychonn.parameters.LR/10, max_lr=ptychonn.parameters.LR,
        step_size_up=step_size, cycle_momentum=False, mode='triangular2')

    # print("training mechanism creation takes ", time.time() - init_time)
    # to store training related metrics
    metrics = {"lrs": [], "losses": [], "val_losses": [], "best_val_loss": math.inf}

    # for loss measure and some stats
    total_iter_count = 0

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
            logger.log("MULTICONTEXT TRAIN ITERATION START", total_iter_count, trainbs, len(trainloader) - epoch_consumed, len(trainloader))
            iteration_start_time = time.time()
            
            train_batch = trainloader.read(bs=min(trainbs, len(trainloader) - epoch_consumed))
            if train_batch[0] is None:
                continue

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
            ptychonn.process_funcs.update_saved_model(
                model=model,
                path=os.path.join(
                    "/dev/shm/", chkpt_dir
                ),
                name=multicontext_parameters.MULTICONTEXT_IPRITER_MODELNAME_FMT.format(next_model)
            )
            ptychonn.ipc.create_shm_marker(
                os.path.join(chkpt_dir, multicontext_parameters.MULTICONTEXT_SHM_MARKER_IPRITER_END.format(next_model)))
            
            logger.log(
                "CREATING CHECKPOINT",
                os.path.join(chkpt_dir, multicontext_parameters.MULTICONTEXT_SHM_MARKER_IPRITER_END.format(next_model)),
                os.path.join(chkpt_dir, multicontext_parameters.MULTICONTEXT_IPRITER_MODELNAME_FMT.format(next_model))
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

    # get the arguments
    args = arg_parser.parse_args()

    # initiate the logger
    logger = logfast.fastlogger.FastLogger()

    # init the model
    model = ptychonn.model.recon_model()
    if os.path.exists(os.path.join("pretrained_model", "pretrained_bestmodel.pth")):
        model = torch.load(os.path.join("pretrained_model", "pretrained_bestmodel.pth"), weights_only=False)
    else:
        print("Pretrained Model Not Found...Exiting")
        exit()
    # init the data reader
    infer_datareader = ptychonn.shm_datareader.SHMInferDataReader()
    train_datareader = ptychonn.shm_datareader.SHMTrainDataReader()

    # wait to synchronize time calculation with produce process
    producer_transmit_wait()
    print("multicontext train consumption start ", time.time())
    
    # training state controller variable initiation
    start_time = time.time()
    cur_ipriteration = 0
    interipr_model_idx = 0
    cur_interval = 1
    current_time = start_time
    cur_interval_start_time = current_time
    deadline_sec = args.deadline / 1000
    # first interval data is used to pretrain the model
    total_runtime = (args.interval_count - 1) * deadline_sec
    trainbs = ptychonn.parameters.TRAIN_BATCH_SIZE

    # to give producer time to put first data
    time.sleep(1/args.acquisition_rate)

    logger.log("INTERVAL START {0}".format(cur_interval + 1))
    while current_time - start_time < total_runtime:
        current_time = time.time()
        if current_time - cur_interval_start_time > args.interval_duration:
            # mark of interval start
            logger.log("INTERVAL END {0}".format(cur_interval + 1))
            cur_interval += 1
            cur_interval_start_time = current_time
            # mark of interval start
            logger.log("INTERVAL START {0}".format(cur_interval + 1))

        # checking for signal existance from IPR process
        # this progression needs to be done irrespective of interval
        # as IPR will keep running for data from interval 0 also (for which model is already trained)
        # it will indicate ground truth is gnereted for some data and IPR has moved from that portion
        # which means completion of SHM_MARKER_FMT_IPRINTERVAL_END.format(cur_ipriteration+1)
        if ptychonn.ipc.exist_shm(ptychonn.parameters.SHM_MARKER_FMT_IPRINTERVAL_END.format(cur_ipriteration + 1)):
            cur_ipriteration += 1
            logger.log("IPR ITERATION START", cur_ipriteration)
            # create the directory for saving model
            ptychonn.ipc.create_shm_folder(multicontext_parameters.MULTICONTEXT_IPRITER_MODEL_DIRNAME_FMT.format(cur_ipriteration))
            # print("first infer data selection takes {0}s".format(time.time() - t1))

            # t1 = time.time()
            train_datareader.set_curipriteration(cur_ipriteration=cur_ipriteration)
            train_datareader.reposition()
            # print("first train data selection takes {0}s".format(time.time() - t1))

        # start of unipipe initiation and call
        ipr_training_time_start = time.time()
        
        ipriter_time_limit = args.ipr_throughput * deadline_sec / (args.datarate - args.ipr_throughput)
        trainsize = int(round(ipriter_time_limit * args.ipr_throughput))
        epoch_count = ptychonn.parameters.EPOCHS

        # log the performance model related states
        logger.log(
            "CURIPRITERATION,TIME_LIMIT,TRAIN_SIZE",
            cur_ipriteration, ipriter_time_limit, trainsize)

        # put unipipe traininfer for one ipriteration data here
        metrics, epoch_count = multicontext_train(
            model, train_datareader, epoch_count=epoch_count,
            trainbs=trainbs, datarate=args.datarate, deadline_sec=deadline_sec,
            chkpt_dir=multicontext_parameters.MULTICONTEXT_IPRITER_MODEL_DIRNAME_FMT.format(cur_ipriteration),
            logger=logger, time_limit=ipriter_time_limit)
        # log how much ipr iteration matches with unipipe iteration
        logger.log(
            "CURIPRITERATION,EPOCH,TIME_LIMIT,ACTUAL_TIME", 
            cur_ipriteration, epoch_count, ipriter_time_limit, time.time() - ipr_training_time_start)

        # busy wait until time is passed
        # while time.time() - unipipe_time_start < unipipe_time_limit:
        #     pass

    logger.persist(args.csvlog_file[:-4] + "_train.log")
