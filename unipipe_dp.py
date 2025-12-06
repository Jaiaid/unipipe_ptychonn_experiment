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
import dp_scheduler.unipipe_scheduler

# for logging
import logfast.fastlogger



def unipipe_dp_traininfer(model, trainloader:ptychonn.shm_datareader.SHMTrainDataReader,
                       teststream:ptychonn.shm_datareader.SHMInferDataReader,
                       epoch_count, datarate, deadline_sec, 
                       traindatalist_fileobj, inferdatalist_fileobj, ipriteration_no, chkpt_path,
                       logger:logfast.fastlogger.FastLogger, schedule=[],
                       time_limit=None, periter_validation=False, inffrac=1.0):

    logger.log("UNIPIPE BEGIN")
    logger.log("UNIPIPE TRAINING DATASET SIZE", len(trainloader))
    logger.log("UNIPIPE INFER DATASET SIZE", len(teststream))

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

    # here we will determine the batch size for training and inference from following parameters
    #
    # 1. Deadline
    # 2. Data rate
    # 3. forward pass time per data sample
    # 4. backward pass time per data sample

    # for loss measure and some stats
    inference_iter_count = 0
    total_iter_count = 0
    total_consumed = 0
    total_missed = 0
    time_uf = None
    time_ub = None

    # list down the train data idx
    for i in trainloader.get_dataidxlist():
        traindatalist_fileobj.write("{0},{1}\n".format(i, ipriteration_no))

    # iter_creation_start_time = time.time()
    # print("iterator creation time:", time.time() - iter_creation_start_time)
    # print("intialization time:", time.time() - interval_init_time)

    train_start_time = time.time()
    # to control when the training of current interval will stop
    previous_loss = 0
    prev_val_loss = math.inf
    val_ratio = 0.0
    # if epochs are finished or convergence happen we stop train but inference continues
    stop_train = False

    cur_iteration_idx = 0
    # run until time limit or epoch count is reached

    # actual breaking condition is on time limit and loss
    cur_epoch = 0
    while time_limit is not None and time.time() - start_time < time_limit and (total_consumed < len(teststream) or cur_epoch < epoch_count):
    # while total_consumed < len(teststream) or cur_epoch < epoch_count:
        cur_epoch += 1
        logger.log("UNIPIPE DP EPOCH BEGIN", cur_epoch, epoch_count)

        tot_loss = 0.0
        loss_amp = 0.0
        loss_ph = 0.0
        total_train_iter_count = 0
        total_iter_count = 0
        train_consumed = 0
        stream_read_error_count = 0

        # to get iteration time
        iteration_time = 0

        # while time_limit is not None and time.time() - start_time < time_limit:
        while time_limit is not None and time.time() - start_time < time_limit and (total_consumed < len(teststream) or cur_epoch < epoch_count):
        # while total_consumed < len(teststream) or cur_epoch < epoch_count:
            logger.log("UNIPIPE DP ITERATION START", total_iter_count)
            iteration_start_time = time.time()
            # first take from test
            infer_count = 0

            inferbs = schedule[cur_iteration_idx][1]
            if total_consumed < len(teststream):
                try:
                    logger.log("READING FROM TEST STREAM", stream_read_error_count)
                    infer_batch, consumed, missed, inferidxlist = teststream.read(
                        bs=min(inferbs, len(teststream) - total_consumed))
                    # if consumed != inferbs:
                    # print(consumed) 
                    if infer_batch is not None:
                        infer_count = infer_batch.shape[0]
                        inference_iter_count += 1
                        total_missed += missed
                        total_consumed += infer_count
                    else:
                        stream_read_error_count += 1
                        if stream_read_error_count > 10:
                            total_consumed = len(teststream)
                except Exception as e:
                    print(e)
                    stream_read_error_count += 1
                    if stream_read_error_count > 10:
                        total_consumed = len(teststream)
                    continue

            train_count = 0
            trainbs = schedule[cur_iteration_idx][0]

            if not stop_train:
                train_batch = trainloader.read(bs=min(trainbs, len(trainloader) - train_consumed))
                if train_batch[0] is None:
                    logger.log("TRAIN READ FAILED, STOPPED TRAIN")
                    stop_train = True
                    train_count = 0
                else:
                    train_consumed += train_batch[0].shape[0]
                    train_count = train_batch[0].shape[0]

            if infer_count > 0 or train_count > 0:
                cur_iteration_idx += 1

            # some infer data is there, merge and pass to context
            # or training is done now to pass only infer data to context
            # or no infer data is in pipeline for now, so only training
            
            forward_pass_arrival_time = time.time()
            if infer_count > 0 and not stop_train:
                ft_images = torch.concat((torch.tensor(infer_batch), torch.tensor(train_batch[0])), axis=0).to("cuda")
            elif infer_count > 0 and stop_train:
                ft_images = torch.tensor(infer_batch).to("cuda")
            elif not stop_train:
                ft_images = torch.tensor(train_batch[0]).to("cuda")
            else:
                break

            logger.log("UNIPIPE DP TRAIN, INFER BS", train_count, infer_count, trainbs, inferbs)
            

            pred_amps, pred_phs = model(ft_images) #Forward pass
            forward_pass_done_time = time.time()

            # before proceeding to backward pass release the inference results
            # by release means put them in result folder
            # to avoid deadline miss as much as possible
            if infer_count > 0:
                pred_amps_cpu_np = pred_amps.cpu().detach().numpy()
                pred_ph_cpu_np = pred_phs.cpu().detach().numpy()
                # print(pred_amps.shape, pred_phs.shape)
                if infer_count != len(inferidxlist):
                    print(infer_count != len(inferidxlist), infer_count, len(inferidxlist))
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
            logger.log("FORWARD PASS TOOK(sec.)", forward_pass_done_time - iteration_start_time)

            # for ideal train batch size calculation
            # time_uf = (forward_pass_done_time -  forward_pass_arrival_time)/ft_images.shape[0]
            # print("uf ", time_uf)

            if not stop_train:
                gt_amps = torch.tensor(train_batch[1]).to("cuda")
                gt_phs = torch.tensor(train_batch[2]).to("cuda")
                val_count = int(gt_amps.shape[0] * val_ratio)

                backward_pass_arrival_time = time.time()
                
                
                # if the flag is set update this for per iteration validaiton
                if periter_validation:
                    logger.log("VAL SIZE", val_count)
                    #Compute validation losses
                    loss_a_val = criterion(pred_amps[infer_count:infer_count+val_count,:], gt_amps[:val_count,])
                    loss_p_val = criterion(pred_phs[infer_count:infer_count+val_count,:], gt_phs[:val_count,])
                    loss_val = (loss_a_val + loss_p_val).detach().item()
                else:
                    loss_val = 0
                    val_count = 0

                loss_a = criterion(pred_amps[infer_count+val_count:,:], gt_amps[val_count:,]) #Monitor amplitude loss
                loss_p = criterion(pred_phs[infer_count+val_count:,:], gt_phs[val_count:,]) #Monitor phase loss but only within support (which may not be same as true amp)
                loss = loss_a + loss_p #Use equiweighted amps and phase

                # first time it will be true always as prev_val_loss == math.inf
                # if we do not update prev_val_loss
                if loss_val < prev_val_loss:
                    if periter_validation:
                        logger.log("VAL LOSS IMPROVED, UPDATING PARAMETERS", prev_val_loss, loss_val)
                        prev_val_loss = loss_val
                    #Zero current grads and do backprop
                    optimizer.zero_grad() 
                    loss.backward()
                    optimizer.step()

                    l_a = loss_a.detach().item()
                    l_p = loss_p.detach().item()
                    loss_amp += l_a
                    loss_ph += l_p
                    tot_loss += l_a + l_p
                    logger.log("ITER_COUNT, TRAIN LOSS", total_iter_count, tot_loss, loss_amp, loss_ph)
                    # logger.log("ITER_COUNT, TRAIN LOSS", l_a + l_p, l_a, l_p)

                    scheduler.step() 
                    metrics['lrs'].append(scheduler.get_last_lr())
                    backward_pass_done_time = time.time()

                    time_ub = (backward_pass_done_time - backward_pass_arrival_time)/(ft_images.shape[0] - infer_count)
                    # print("ub ", time_ub)
                    total_train_iter_count += 1

                    iter_end_timestamp = time.time()
                    logger.log("BACKWARD TAKES(sec.)", iter_end_timestamp - backward_pass_arrival_time)
            else:
                iter_end_timestamp = time.time()

            total_iter_count += 1
            iter_end_timestamp = time.time()
            logger.log("ITERATION TAKES(sec.)", iter_end_timestamp - iteration_start_time)
            iteration_time += iter_end_timestamp - iteration_start_time

            if train_consumed >= len(trainloader):
                logger.log("TRAINDATASET CONSUMED AT EPOCH", cur_epoch, len(trainloader))
                break

        if not stop_train:
            # ptychonn.process_funcs.update_saved_model(
            #     model=model, path="/dev/shm/", name="{0}_e{1}.pth".format(chkpt_path[:-4], cur_epoch))
            #Divide cumulative loss by number of batches-- sli inaccurate because last batch is different size
            metrics['losses'].append([tot_loss/(total_train_iter_count + 1),loss_amp/(total_train_iter_count + 1),loss_ph/(total_train_iter_count + 1)])
            logger.log("TRAINING LOSS AT EPOCH", cur_epoch, tot_loss/(total_train_iter_count + 1),loss_amp/(total_train_iter_count + 1),loss_ph/(total_train_iter_count + 1))
        
        # update flag to stop training if training brings out minimal improvement
        # this condition needed because it is possible in current epoch not a single training is run
        if not stop_train:
            if abs(tot_loss/(total_train_iter_count + 1) - previous_loss) <= ptychonn.parameters.LOSS_CHANGE_MIN_THRESHOLD:
                stop_train = True
                logger.log("UNIPIPE TRAINING STOP CONVERGENCE")
                interval_remaining_time = (time.time() - interval_init_time) - time_limit
                logger.log("UNIPIPE INTERVAL REM. TIME", interval_remaining_time)
            previous_loss = tot_loss/(total_train_iter_count + 1)

        logger.log("UNIPIPE ITER. TIME", iteration_time/(total_iter_count + 1))
        logger.log("UNIPIPE EPOCH END", cur_epoch)
        if cur_epoch >= epoch_count:
            stop_train = True
            logger.log("MAX EPOCH DONE")

    if total_consumed >= len(infer_datareader):
        logger.log("ALL INFER DATA CONSUMED")

    if time_limit is not None and time.time() - start_time > time_limit:
        logger.log("UNIPIPE TIMELIMIT OVER", cur_epoch)


    total_time = time.time() - start_time
    training_time = time.time() - train_start_time
    logger.log("UNIPIPE TRAINING TAKES", training_time)
    # expected that time_limit - total_time > 0 because of rounding down of epoch
    logger.log("UNIPIPE TOTAL TIME:", total_time, time_limit, time_limit - total_time)
    logger.log("TOTAL CONSUMED,MISSED", total_consumed, total_missed)

    return metrics, total_consumed, total_missed, time_uf, time_ub


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
    arg_parser.add_argument("--validation-training", "-validation", action="store_true", help="if per iteration validation will be used")
    arg_parser.add_argument("--inffrac", "-inffrac", type=float, default=1.0, help="how much factor to multiply with infer bs")
    arg_parser.add_argument("--skip-line-pretrained", "-skipline", type=int, help="how many data to skip as model is pretrained on it")
    arg_parser.add_argument("--large-dataset", "-largedataset", action="store_true", help="if larger dataset will be used")
    arg_parser.add_argument("--model-type", "-type", type=str, choices=["1.25M", "5M", "10M", "20M", "100M", "200M"], help="which model to choose", default="1.25M")
    arg_parser.add_argument("--gtcount", "-gtcount", type=int, required=False, help="how many ground truth will be consumed by phase retrieval process", default=1)
    
    # get the arguments
    args = arg_parser.parse_args()

    # to keep total batch size constant
    args.inferbs = ptychonn.parameters.INFERENCE_BATCH_SIZE + ptychonn.parameters.TRAIN_BATCH_SIZE - args.trainbs

    # init the model
    model = ptychonn.model.get_model(type_name=args.model_type)

    _, _, _, nn_uf, nn_ub = ptychonn.model.benchmark_model(model, bs=args.trainbs+args.inferbs, warmup=30, iters=220)
    nn_uf = 0.00027
    nn_ub = 0.00036
    # other variants are just for performance test
    if args.model_type == "1.25M":
        if os.path.exists(os.path.join("pretrained_model", "pretrained_bestmodel.pth")):
            model = torch.load(os.path.join("pretrained_model", "pretrained_bestmodel.pth"), weights_only=False)
        else:
            print("Pretrained Model Not Found...Exiting")
            exit()
    # GPU environment is assumed
    model.to("cuda")

    # taken from paper's code
    # if optimizer_objects is None:
    iter_per_epoch = 1
    step_size = 6
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr = ptychonn.parameters.LR)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=ptychonn.parameters.LR/10, max_lr=ptychonn.parameters.LR,
        step_size_up=step_size, cycle_momentum=False, mode='triangular2')

    # initiate the logger
    logger = logfast.fastlogger.FastLogger()
    
    # make a result directory where generated images will be stored
    ptychonn.ipc.create_shm_folder(ptychonn.parameters.SHM_MARKER_NNRES_FOLDER)

    # initiate the file name to log down which data got consumed for what
    traindatalist_file = open(
        "/dev/shm/traindatalist_unipipe_{0}_{1}_{2}_{3}.csv".format(
            args.interval_count, args.interval_duration, args.datarate, int(args.ipr_throughput)), "w") 
    inferdatalist_file = open(
        "/dev/shm/inferdatalist_unipipe_{0}_{1}_{2}_{3}.csv".format(
            args.interval_count, args.interval_duration, args.datarate, int(args.ipr_throughput)), "w") 

    # # init the model
    # model = ptychonn.model.recon_model()
    # if os.path.exists(os.path.join("pretrained_model", "pretrained_bestmodel.pth")):
    #     model = torch.load(os.path.join("pretrained_model", "pretrained_bestmodel.pth"), weights_only=False)
    # else:
    #     print("Pretrained Model Not Found...Exiting")
    #     exit()
    # init the data reader
    infer_datareader = ptychonn.shm_datareader.SHMInferDataReader()
    train_datareader = ptychonn.shm_datareader.SHMTrainDataReader()
    # wait to synchronize time calculation with produce process
    print("waiting for others")
    
    # warmup run
    warmup_start_time = time.time()
    metrics, _, _, _, _ = unipipe_traininfer(
            model, train_datareader, infer_datareader, epoch_count=1,
            datarate=args.datarate, deadline_sec=args.deadline/1000,
            traindatalist_fileobj=traindatalist_file, inferdatalist_fileobj=inferdatalist_file, ipriteration_no=0,
            chkpt_path="dummy.pth", logger=logger, periter_validation=args.validation_training,
            time_limit=args.deadline/1000)
    logger.log("Warmup Run took {0}s".format(time.time() - warmup_start_time))

    deadline_sec = args.deadline / 1000
    unipipe_time_limit = deadline_sec# args.ipr_throughput * deadline_sec / (args.datarate - args.ipr_throughput)
    unipipe_time_limit, schedule = ptychonn.perf_model.estimate_unipipe_schedule(
        phase_retrieval_genrate=args.ipr_throughput, deadline_sec=deadline_sec,
        acquisition_rate=args.datarate, ground_truth_count=args.gtcount
    )


    # signal producer that done, needed if initiation become expensive
    signal_producer()
    # wait for producer to start transmission
    producer_transmit_wait()
    
    # training state controller variable initiation
    start_time = time.time()
    print("starting ", start_time)
    cur_ipriteration = -1
    cur_interval = 1
    current_time = start_time
    cur_interval_start_time = current_time
    
    if args.large_dataset:
        total_runtime = args.interval_count * args.interval_duration
    else:
        # first interval data is used to pretrain the model
        total_runtime = args.interval_count * args.interval_duration

    # to give producer time to put first data
    # time.sleep(1/args.datarate)
    time_list = []
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

        # checking for signal existance from IPR process
        # this progression needs to be done irrespective of interval
        # as IPR will keep running for data from interval 0 also (for which model is already trained)
        # it will indicate ground truth is gnereted for some data and IPR has moved from that portion
        # which means completion of SHM_MARKER_FMT_IPRINTERVAL_END.format(cur_ipriteration+1)
        unipipe_time_start = time.time()

        if ptychonn.ipc.exist_shm(ptychonn.parameters.SHM_MARKER_FMT_IPRINTERVAL_END.format(cur_ipriteration + 1)):
            cur_ipriteration += 1
            logger.log("IPR ITERATION START", cur_ipriteration)

            # t1 = time.time()
            train_datareader.set_curipriteration(cur_ipriteration=cur_ipriteration)
            # train_datareader.reposition()
            
            logger.log("TIPR LEN", unipipe_time_limit, nn_uf, nn_ub)
            
            trainsize = int(math.floor(unipipe_time_limit * args.ipr_throughput))
            infersize = int(math.floor(unipipe_time_limit * (args.datarate - args.ipr_throughput))) # same as args.ipr_throughput * deadline_sec
            # for inference location on datastream repositioning
            train_readidx_curpos = (cur_ipriteration-1)*(trainsize + infersize)
            logger.log("INFER DATAREADER STATUS", infer_datareader.cur_readidx, train_readidx_curpos, train_readidx_curpos - infersize + 1, infersize)
            logger.log("TRAIN DATAREADER STATUS", infer_datareader.cur_readidx, train_readidx_curpos, train_readidx_curpos - trainsize + 1, trainsize)
            
            # infer_datareader.cur_readidx = train_readidx_curpos - infersize + 1
            
            # infer_datareader.cur_readidx  = train_datareader.cur_readidx_begin - infersize + 1
            infer_datareader.cur_readidx = train_readidx_curpos + trainsize
            # set the reader length for the unipipe call
            # to handle initial boundary condition
            train_datareader.set_len(begin=train_readidx_curpos, end=train_readidx_curpos+trainsize-1)
            infer_datareader.set_len(infersize if infer_datareader.cur_readidx >= 0 else 0)
        else:
            continue
            # print("first train data selection takes {0}s".format(time.time() - t1))

        # start of unipipe initiation and call
        
        # log the performance model related states
        logger.log(
            "CURIPRITERATION,TIME_LIMIT,TRAIN_SIZE,INFER_SIZE",
            cur_ipriteration, unipipe_time_limit, trainsize, infersize)

        # calculate number of epoch to run the unipipe train
        # it depends on some value which are unknown initially for that we just set an arbitrary value
        if nn_uf is None or nn_ub is None:
            # need one epoch to count the times
            epoch_count = 1
        elif trainsize == 0:
            # if no training data just run an epoch but that will only consume inference data
            epoch_count = 1
        else:
            # from performance model
            epoch_count = 1 #int(round((unipipe_time_limit - nn_uf*infersize)/((nn_uf + nn_ub) * trainsize)))
            logger.log("PRECISE EPOCH COUNT", (unipipe_time_limit - nn_uf*infersize)/((nn_uf + nn_ub) * trainsize))
        # epoch_count=1
        # put unipipe traininfer for one ipriteration data here
        metrics, _, _, _, _ = unipipe_dp_traininfer(
            model, train_datareader, infer_datareader, epoch_count=epoch_count,
            datarate=args.datarate, deadline_sec=deadline_sec,
            traindatalist_fileobj=traindatalist_file, inferdatalist_fileobj=inferdatalist_file, ipriteration_no=cur_ipriteration,
            chkpt_path="inctrained_interval{0}_model.pth".format(cur_ipriteration), logger=logger, periter_validation=args.validation_training,
            time_limit=unipipe_time_limit, schedule=schedule)
        # log how much ipr iteration matches with unipipe iteration
        time_taken = time.time() - unipipe_time_start
        logger.log(
            "CURIPRITERATION,EPOCH,UNIPIPE_TIME_LIMIT,ACTUAL_TIME", 
            cur_ipriteration, epoch_count, unipipe_time_limit, time_taken)
        # print("CURIPRITERATION,EPOCH,UNIPIPE_TIME_LIMIT,ACTUAL_TIME", 
        #     cur_ipriteration, epoch_count, unipipe_time_limit, time_taken)
        time_list.append(time_taken)


        # busy wait until time is passed
        # condition need for case when we can not estimate epoch count
        # if nn_uf is not None and nn_ub is not None:
        while time.time() - unipipe_time_start < unipipe_time_limit:
            pass
        # print("Hi")


    # postmortem of data, calculate error
    amp_error, ph_error, nn_amp_error, nn_ph_error = ptychonn.error_calculation.postsimulation_error_calc(
        skip_line=args.skip_line_pretrained, large_dataset=args.large_dataset
    )

    with open(args.csvlog_file, "w") as fout:
        # amp error, ph error, nn amp error, nn ph error
        fout.write("{0},{1},{2},{3}\n".format(amp_error, ph_error, nn_amp_error, nn_ph_error))

    traindatalist_file.close()
    inferdatalist_file.close()
    logger.log("MEAN TIME EACH INTERVAL", sum(time_list[2:])/(len(time_list)-2), unipipe_time_limit)
    logger.persist(args.csvlog_file[:-4] + ".log")
    print("Mean time: ", sum(time_list[2:])/(len(time_list)-2), unipipe_time_limit)


    # # do this at the end to avoid any performance in continual training
    # if args.allckpttest:
    #     model = torch.load(os.path.join("pretrained_model", "pretrained_bestmodel.pth"))
    #     model.to("cuda")
    #     # pretrained model evaluation at different interval
    #     for interval_count in range(args.interval_count):
    #         testloader = torch.utils.data.DataLoader(
    #             torch.utils.data.Subset(
    #                         test_data, list(range(interval_count * len(test_data)//args.interval_count,
    #                                               (interval_count + 1)* len(test_data)//args.interval_count))),
    #             batch_size=ptychonn.parameters.INFERENCE_BATCH_SIZE,
    #             shuffle=False, num_workers=1)

    #         loss_total, loss_amp, loss_ph = ptychonn.process_funcs.testloss(model=model, testloader=testloader)
    #         logger.log("ALL CKPT TESTDATA INTERVAL, EPOCH, LOSS", 0, 0, loss_total, loss_amp, loss_ph)

    #     for interval_count in range(1, args.interval_count):
    #         for epoch_count in range(ptychonn.parameters.EPOCHS):
    #             try:
    #                 model = torch.load(os.path.join("/dev/shm", "inctrained_interval{0}_model_e{1}.pth".format(interval_count, epoch_count)))
    #                 model.to("cuda")
    #                 testloader = torch.utils.data.DataLoader(
    #                     torch.utils.data.Subset(
    #                         test_data, list(range(interval_count * len(test_data)//args.interval_count,
    #                                               (interval_count + 1)* len(test_data)//args.interval_count))),
    #                     batch_size=ptychonn.parameters.INFERENCE_BATCH_SIZE,
    #                     shuffle=False, num_workers=1)

    #                 loss_total, loss_amp, loss_ph = ptychonn.process_funcs.testloss(model=model, testloader=testloader)
    #                 logger.log("ALL CKPT TESTDATA INTERVAL, EPOCH, LOSS", interval_count, epoch_count, loss_total, loss_amp, loss_ph)
    #             except Exception as e:
    #                 print(e)
    #                 break
