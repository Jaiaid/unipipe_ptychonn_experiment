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

# for logging
import logfast.fastlogger


def unipipe_traininfer(model, trainloader:ptychonn.shm_datareader.SHMTrainDataReader,
                       teststream:ptychonn.shm_datareader.SHMInferDataReader,
                       epoch_count, datarate, deadline_sec, chkpt_path,
                       logger:logfast.fastlogger.FastLogger,
                       time_limit=None, constant_bs=False,
                       trainbs=ptychonn.parameters.TRAIN_BATCH_SIZE, inferbs=ptychonn.parameters.INFERENCE_BATCH_SIZE, inffrac=1.0):

    logger.log("UNIPIPE BEGIN")
    logger.log("UNIPIPE TRAINING DATASET SIZE", len(trainloader))
    logger.log("UNIPIPE TRAINING BATCH SIZE", trainbs)
    logger.log("UNIPIPE CONSTANT TRAINING BS", constant_bs)

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
    # for training batch size control
    train_dataset_size = len(trainloader)

    # here we will determine the batch size for training and inference from following parameters
    #
    # 1. Deadline
    # 2. Data rate
    # 3. forward pass time per data sample
    # 4. backward pass time per data sample
    if not constant_bs:
        ideal_infer_bs = min(
            ptychonn.parameters.INFERENCE_BATCH_SIZE,
            len(infer_datareader),
            int(inffrac * ptychonn.perf_model.estimate_infer_bs(datarate, time_limit)))
    else:
        ideal_infer_bs = inferbs

    if ideal_infer_bs == 0:
        ideal_infer_bs = 1
    logger.log("INFER BATCH SIZE IDEAL", ideal_infer_bs)

    ideal_train_batchsize = trainbs
    negative_train_batchsize_count = 0
    

    # for loss measure and some stats
    inference_iter_count = 0
    total_iter_count = 0
    total_consumed = 0
    total_missed = 0
    time_uf = None
    time_ub = None

    iter_creation_start_time = time.time()
    print("iterator creation time:", time.time() - iter_creation_start_time)
    print("intialization time:", time.time() - interval_init_time)

    train_start_time = time.time()
    # to control when the training of current interval will stop
    previous_loss = 0
    # if epochs are finished or convergence happen we stop train but inference continues
    stop_train = False

    # arbitrary large epoch, for coding ease in tracking an epoch
    # actual breaking condition is on time limit and loss
    for cur_epoch in range(epoch_count):
        logger.log("UNIPIPE EPOCH BEGIN", cur_epoch + 1)

        tot_loss = 0.0
        loss_amp = 0.0
        loss_ph = 0.0
        total_train_iter_count = 0
        total_iter_count = 0

        # to get iteration time
        iteration_time = 0
        if time_limit is not None and time.time() - start_time > time_limit:
            logger.log("UNIPIPE EPOCH END TIMELIMIT OVER", cur_epoch + 1)
            break

        print("Epoch count:", cur_epoch)

        traindata_start_idx = 0
        time_uf = None
        time_ub = None

        while time_limit is not None and time.time() - start_time < time_limit and total_consumed < len(infer_datareader):
            logger.log("UNIPIPE ITERATION START", total_iter_count)
            iteration_start_time = time.time()
            # first take from test
            infer_count = 0

            try:
                infer_batch, consumed, missed, inferidxlist = teststream.read(bs=ideal_infer_bs)
                if infer_batch is None:
                    return None, None, None, None, None
                infer_count = infer_batch.shape[0]
                total_consumed += infer_count
                inference_iter_count += 1
            except Exception as e:
                print(e)

            if not stop_train:
                train_batch = trainloader.read(bs=ideal_train_batchsize)
                if train_batch[0] is None:
                    return None, None, None, None, None

            # some infer data is there, merge and pass to context
            # or training is done now to pass only infer data to context
            # or no infer data is in pipeline for now, so only training
            if infer_count > 0 and not stop_train:
                ft_images = torch.concat((torch.tensor(infer_batch), torch.tensor(train_batch[0])), axis=0).to("cuda")
            elif infer_count > 0 and stop_train:
                ft_images = torch.tensor(infer_batch[0]).to("cuda")
            else:
                ft_images = torch.tensor(train_batch[0]).to("cuda")

            logger.log("UNIPIPE TRAIN, INFER BS", ideal_train_batchsize, infer_count)
            # to keep track how many infer request missed due to forward pass latency
            forward_pass_arrival_time = time.time()
            pred_amps, pred_phs = model(ft_images) #Forward pass
            forward_pass_done_time = time.time()

            # before proceeding to backward pass release the inference results
            # by release means put them in result array
            # to avoid deadline miss as much as possible
            if infer_count > 0:
                true_amp = infer_batch[1]
                true_ph = infer_batch[2]
                for i in range(infer_count):
                    ptychonn.ipc.create_shm_data(
                        os.path.join(
                            ptychonn.parameters.SHM_MARKER_NNRES_FOLDER,
                            ptychonn.parameters.SHM_MARKER_NNRES_PHASE_NAMEFMT.format(inferidxlist[i])
                        ),
                        true_ph
                    )
                    ptychonn.ipc.create_shm_data(
                        os.path.join(
                            ptychonn.parameters.SHM_MARKER_NNRES_FOLDER,
                            ptychonn.parameters.SHM_MARKER_NNRES_AMP_NAMEFMT.format(inferidxlist[i])
                        ),
                        true_amp
                    )
            # update total missed count
            logger.log("FORWARD PASS TOOK(sec.)", forward_pass_done_time - iteration_start_time)
            total_missed += missed
            total_consumed += consumed

            # for ideal train batch size calculation
            time_uf = (forward_pass_done_time -  forward_pass_arrival_time)/ft_images.shape[0]
            # print("uf ", time_uf)

            if not stop_train:
                gt_amps = torch.tensor(train_batch[1]).to("cuda")
                gt_phs = torch.tensor(train_batch[2]).to("cuda")

                backward_pass_arrival_time = time.time()
                #Compute losses
                loss_a = criterion(pred_amps[infer_count:,:], gt_amps) #Monitor amplitude loss
                loss_p = criterion(pred_phs[infer_count:,:], gt_phs) #Monitor phase loss but only within support (which may not be same as true amp)
                loss = loss_a + loss_p #Use equiweighted amps and phase

                #Zero current grads and do backprop
                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()

                tot_loss += loss.detach().item()
                loss_amp += loss_a.detach().item()
                loss_ph += loss_p.detach().item()

                scheduler.step() 
                metrics['lrs'].append(scheduler.get_last_lr())
                backward_pass_done_time = time.time()

                time_ub = (backward_pass_done_time - backward_pass_arrival_time)/(ft_images.shape[0] - infer_count)
                # print("ub ", time_ub)

                # calculate batch size according to performance model if not constant batch size experiment
                prev_ideal_train_bs = ideal_train_batchsize
                if not constant_bs:
                    ideal_train_batchsize = min(
                        ptychonn.perf_model.estimate_train_bs(ideal_infer_bs, time_uf, time_ub, datarate),
                        ptychonn.parameters.TRAIN_BATCH_SIZE)

                if ideal_train_batchsize != prev_ideal_train_bs:
                    logger.log("TRAINBATCH SIZE FROM", prev_ideal_train_bs, ideal_train_batchsize,
                               time_ub, time_uf, ideal_infer_bs/datarate,
                               ideal_infer_bs * time_uf + ideal_train_batchsize * (time_uf + time_ub))
                if ideal_train_batchsize <= 0:
                    logger.log("TRAINBATCH SIZE TO <=0", deadline_sec, 1-datarate*time_uf, time_uf, time_ub, datarate)
                    # if the configuration does not allow training and inference, 
                    # we will prioritize training 
                    ideal_train_batchsize = 1
                    negative_train_batchsize_count += 1
                    logger.log("TRAINBATCH SIZE TO <=0 FOR TIMES", negative_train_batchsize_count)
                    if negative_train_batchsize_count > 15:
                        logger.log("TRAINBATCH SIZE TO <=0 ABOVE THRESHOLD, PRIORITIZE TRAINING", negative_train_batchsize_count)
                        ideal_train_batchsize = ptychonn.parameters.TRAIN_BATCH_SIZE
                        negative_train_batchsize = True
                        # ideal_infer_bs = None
                        constant_bs = True
                else:
                    negative_train_batchsize_count = 0
                total_train_iter_count += 1

                iter_end_timestamp = time.time()
                logger.log("BACKWARD TAKES(sec.)", iter_end_timestamp - backward_pass_arrival_time)
            else:
                iter_end_timestamp = time.time()

            total_iter_count += 1
            logger.log("ITERATION TAKES(sec.)", iter_end_timestamp - iteration_start_time)
            iteration_time += iter_end_timestamp - iteration_start_time

            if traindata_start_idx >= len(trainloader):
                logger.log("TRAINDATASET CONSUMED AT EPOCH", cur_epoch + 1, traindata_start_idx, len(trainloader))
                break

        if not stop_train:
            ptychonn.process_funcs.update_saved_model(
                model=model, path="/dev/shm/", name="{0}_e{1}.pth".format(chkpt_path[:-4], cur_epoch))
            #Divide cumulative loss by number of batches-- sli inaccurate because last batch is different size
            metrics['losses'].append([tot_loss/(total_train_iter_count + 1),loss_amp/(total_train_iter_count + 1),loss_ph/(total_train_iter_count + 1)])
            logger.log("TRAINING LOSS AT EPOCH", cur_epoch, tot_loss/(total_train_iter_count + 1),loss_amp/(total_train_iter_count + 1),loss_ph/(total_train_iter_count + 1))
        
        # update flag to stop training if training brings out minimal improvement
        # this condition needed because it is possible in current epoch not a single training is run
        if not stop_train:
            if abs(tot_loss/(total_train_iter_count + 1) - previous_loss) <= ptychonn.parameters.LOSS_CHANGE_MIN_THRESHOLD:
                stop_train = True
                logger.log("UNIPIPE TRAINING STOP CONVERGENCE")
                interval_remaining_time = time.time() - interval_init_time
                logger.log("UNIPIPE INTERVAL REM. TIME", interval_remaining_time)
            previous_loss = tot_loss/(total_train_iter_count + 1)

        logger.log("UNIPIPE ITER. TIME", iteration_time/total_iter_count)
        logger.log("UNIPIPE EPOCH END", cur_epoch + 1)
        if cur_epoch + 1 == epoch_count:
            stop_train = True
            logger.log("MAX EPOCH DONE, NO TRAINING IN CURRENT INTERVAL")

        # if train stopped switch to ideal infer  bs
        if stop_train:
            ideal_infer_bs = math.ceil((1 + deadline_sec * datarate) / 2)
            logger.log("TRAIN DONE, SWITCHING TO PERF. MODEL INFER BS.", ideal_infer_bs)

        if total_consumed >= len(infer_datareader):
            logger.log("ALL INFER DATA CONSUMED")
            break

    logger.log("UNIPIPE TRAINING TAKES", time.time() - train_start_time)
    logger.log("UNIPIPE TOTAL TIME:", time.time() - start_time)
    logger.log("TOTAL CONSUMED,MISSED", total_consumed, total_missed)

    return metrics, total_consumed, total_missed, time_uf, time_ub


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
    # get the arguments
    args = arg_parser.parse_args()

    # initiate the logger
    logger = logfast.fastlogger.FastLogger()
    
    # make a result directory where generated images will be stored
    ptychonn.ipc.create_shm_folder(ptychonn.parameters.SHM_MARKER_NNRES_FOLDER)

    # init the model
    model = ptychonn.model.recon_model()
    # init the data reader
    infer_datareader = ptychonn.shm_datareader.SHMInferDataReader()
    train_datareader = ptychonn.shm_datareader.SHMTrainDataReader()

    # wait to synchronize time calculation with produce process
    producer_transmit_wait()
    
    # training state controller variable initiation
    start_time = time.time()
    cur_ipriteration = 0
    cur_ipriteration_infercountlimit = 0
    cur_interval = 0
    current_time = start_time
    cur_interval_start_time = current_time
    deadline_sec = args.deadline / 1000
    total_runtime = args.interval_count * deadline_sec
    nn_uf = None
    nn_ub = None

    logger.log("INTERVAL START {0}".format(cur_interval + 1))
    while current_time - start_time < total_runtime:
        current_time = time.time()
        if current_time - cur_interval_start_time > deadline_sec:
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
            
            cur_ipriteration_infercountlimit = int(args.ipr_throughput * deadline_sec)
            # update the current inference idx and training data idx
            # the files are named in such a way that
            # t1 = time.time()
            infer_datareader.reposition()
            # print("first infer data selection takes {0}s".format(time.time() - t1))

            # t1 = time.time()
            train_datareader.set_curipriteration(cur_ipriteration=cur_ipriteration)
            train_datareader.reposition()
            # print("first train data selection takes {0}s".format(time.time() - t1))

        # pretrain stage
        if cur_interval == 0:
            # if we have done pretraining already with some model no need to redo it
            if os.path.exists(os.path.join("pretrained_model", "pretrained_bestmodel.pth")):
                model = torch.load(os.path.join("pretrained_model", "pretrained_bestmodel.pth"))
                continue
            continue

        # start of unipipe initiation and call
        unipipe_time_start = time.time()
        # TODO:
        # put performance model call here to determine the batch size, length of unipipe inference
        # same as T_IPR
        unipipe_time_limit = args.ipr_throughput * deadline_sec / args.datarate
        trainsize = unipipe_time_limit * args.ipr_throughput
        infersize = unipipe_time_limit * args.datarate # same as args.ipr_throughput * deadline_sec
        # set the reader length for the unipipe call
        infer_datareader.set_len(len)

        # calculate number of epoch to run the unipipe train
        # it depends on some value which are unknown initially for that we just set an arbitrary value
        if nn_uf is None:
            epoch_count = 1000000
        else:
            # from performance model
            epoch_count = int(round((unipipe_time_limit - nn_uf*infersize)/((nn_uf + nn_ub) * trainsize)))

        # put unipipe traininfer for one ipriteration data here
        metrics, _, _, nn_uf, nn_ub = unipipe_traininfer(
            model, train_datareader, infer_datareader, epoch_count=epoch_count,
            datarate=args.datarate, deadline_sec=deadline_sec,
            chkpt_path="inctrained_interval{0}_model.pth".format(cur_ipriteration), logger=logger,
            time_limit=unipipe_time_limit)
        # log how much ipr iteration matches with unipipe iteration
        logger.log(
            "CURIPRITERATION,EPOCH,UNIPIPE_TIME_LIMIT,ACTUAL_TIME", 
            cur_ipriteration, epoch_count, unipipe_time_limit, time.time() - unipipe_time_start)

        # busy wait until time is passed
        while time.time() - unipipe_time_start < unipipe_time_limit:
            pass


    # postmortem of data, calculate error
    amp_error, ph_error = ptychonn.error_calculation.postsimulation_error_calc()

    with open(args.csvlog_file, "w") as fout:
        # amp error, ph error
        fout.write("{0},{1}\n".format(amp_error, ph_error))

    logger.persist(args.csvlog_file[:-4] + ".log")


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
