import time
import math
import os
import numpy as np
import torch
import copy

from tqdm import tqdm

from . import parameters
from . import error_calculation
from . import ipc
from . import datastream

from logfast import fastlogger

def train(model, trainloader, epoch, bs, chkpt_path, device="cuda", lr=1e-3,
          do_validate=True, validloader=None, time_limit=None, shm_signal_name=None, logger:fastlogger.FastLogger=None, constant_bs=False):
    logger.log("TRAINING BEGIN")
    logger.log("TRAINING DATASET SIZE", len(trainloader.dataset))
    logger.log("TRAINING BATCH SIZE", bs)

    start_time = time.time()
    tot_loss = 0.0
    loss_amp = 0.0
    loss_ph = 0.0

    iter_per_epoch = np.floor(len(trainloader.dataset)/bs) + 1
    logger.log("ITER PER EPOCH", iter_per_epoch)
    # taken from paper's code
    step_size = 6 * iter_per_epoch
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr/10, max_lr=lr, step_size_up=step_size,
                                                cycle_momentum=False, mode='triangular2')

    # to store training related metrics
    metrics = {"lrs": [], "losses": [], "val_losses": [], "best_val_loss": math.inf}
    # move model to GPU
    model.to(device)

    # to control when the training of current interval will stop
    previous_loss = 0
    total_iter_count = 0

    for cur_epoch in range(epoch):
        logger.log("TRAINING EPOCH BEGIN", cur_epoch + 1)
        for i, batch in enumerate(trainloader):
            ft_images = batch[0].to(device) #Move everything to device
            amps = batch[1].to(device)
            phs = batch[2].to(device)

            pred_amps, pred_phs = model(ft_images) #Forward pass

            #Compute losses
            loss_a = criterion(pred_amps,amps) #Monitor amplitude loss
            loss_p = criterion(pred_phs,phs) #Monitor phase loss but only within support (which may not be same as true amp)
            loss = loss_a + loss_p #Use equiweighted amps and phase

            #Zero current grads and do backprop
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()

            tot_loss += loss.detach().item()
            loss_amp += loss_a.detach().item()
            loss_ph += loss_p.detach().item()

            #Update the LR according to the schedule -- CyclicLR updates each batch
            scheduler.step() 
            metrics['lrs'].append(scheduler.get_last_lr())
            total_iter_count += 1

        logger.log("TRAINING LOSS AT EPOCH", cur_epoch, tot_loss/(i+1), loss_amp/(i+1), loss_ph/(i+1))

        # do a validation test if flag set
        if validloader is not None and do_validate:
            val_loss, val_loss_amp, val_loss_ph = validate(
                model=model, criterion=criterion, validloader=validloader,
                device=device, metrics=metrics)
            # update the validation loss
            metrics['val_losses'].append([val_loss, val_loss_amp,val_loss_ph])
            if metrics["best_val_loss"] > val_loss:
                metrics["best_val_loss"] = val_loss
                # update save model if to update
                if chkpt_path is not None:
                    update_saved_model(
                        model=model, path=os.path.dirname(chkpt_path),
                        name=os.path.basename(chkpt_path))
                    # create marker for inference process to load new model
                    if shm_signal_name is not None:
                        ipc.create_shm_marker(shm_signal_name)
            logger.log("VAL. LOSS AT EPOCH", cur_epoch, val_loss, val_loss_amp, val_loss_ph)

        elif chkpt_path is not None:
            # update save model if to update
            logger.log("TRAINING SAVING MODEL EPOCH", cur_epoch + 1)
            update_saved_model(
                model=model, path=os.path.dirname(chkpt_path),
                name=os.path.basename(chkpt_path))

            if shm_signal_name is not None:
                logger.log("TRAINING CREATING IPC SIGNAL EPOCH", cur_epoch + 1)
                # create marker for inference process to load new model
                ipc.create_shm_marker(shm_signal_name)

        logger.log("TRAINING EPOCH END", cur_epoch + 1)

        if time_limit is not None and time.time() - start_time > time_limit:
            logger.log("TRAINING STOP TIMELIMIT OVER", time_limit)
            break
        #Divide cumulative loss by number of batches-- sli inaccurate because last batch is different size
        metrics['losses'].append([tot_loss/(i+1),loss_amp/(i+1),loss_ph/(i+1)])

        if abs(metrics['losses'][-1][0]- previous_loss) <= parameters.LOSS_CHANGE_MIN_THRESHOLD:
            logger.log("TRAINING STOP CONVERGENCE")
            break

        previous_loss = tot_loss/(i+1)
        tot_loss = 0.0
        loss_amp = 0.0
        loss_ph = 0.0

    logger.log("TRAINING END")
    # print(metrics['best_val_loss'])
    # print(metrics['val_losses'])
    # print(len(metrics['lrs']))
    # exit(0)

    return metrics

def unipipe_traininfer(model, trainloader, teststream:datastream.DataStream, result_fiilup_list, epoch, bs,
                       chkpt_path, device="cuda", lr=1e-3, time_limit=None, logger:fastlogger.FastLogger=None, constant_bs=False):
    logger.log("UNIPIPE BEGIN")
    logger.log("UNIPIPE TRAINING DATASET SIZE", len(trainloader.dataset))
    logger.log("UNIPIPE TRAINING BATCH SIZE", bs)
    logger.log("UNIPIPE CONSTANT TRAINING BS", constant_bs)

    start_time = time.time()
    interval_init_time = time.time()
    interval_remaining_time = time_limit

    # taken from paper's code
    # if optimizer_objects is None:
    iter_per_epoch = np.floor(len(trainloader.dataset)/bs) + 1
    step_size = 6 * iter_per_epoch
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr/10, max_lr=lr, step_size_up=step_size,
                                                cycle_momentum=False, mode='triangular2')

    # print("training mechanism creation takes ", time.time() - init_time)
    # to store training related metrics
    metrics = {"lrs": [], "losses": [], "val_losses": [], "best_val_loss": math.inf}
    # for training batch size control
    train_dataset_size = len(trainloader.dataset)
    
    # here we will determine the batch size for training and inference from following parameters
    #
    # 1. Deadline
    # 2. Data rate
    # 3. forward pass time per data sample
    # 4. backward pass time per data sample
    ideal_infer_bs = round(teststream.get_deadlinesec() * teststream.get_datarate() / 2)
    logger.log("INFER BATCH SIZE IDEAL", ideal_infer_bs)
    if ideal_infer_bs == 0:
        ideal_infer_bs = 1

    ideal_train_batchsize = bs
    missed_infer = 0

    # for loss measure and some stats
    inference_iter_count = 0
    total_iter_count = 0
    # for time limit testing
    total_served = 0
    total_missed = len(result_fiilup_list[0])
    total_streamed = 0

    # test data will be consumed only once in the interval
    iter_creation_start_time = time.time()
    # testloader_iter = iter(teststream)
    print("iterator creation time:", time.time() - iter_creation_start_time)
    print("intialization time:", time.time() - interval_init_time)

    train_start_time = time.time()
    # to control when the training of current interval will stop
    previous_loss = 0
    stop_train = False

    # arbitrary large epoch, for coding ease in tracking an epoch
    # actual breaking condition is on time limit and loss
    for cur_epoch in range(int(1e6)):
        logger.log("UNIPIPE EPOCH BEGIN", cur_epoch + 1)

        tot_loss = 0.0
        loss_amp = 0.0
        loss_ph = 0.0
        total_train_iter_count = 0

        # to get iteration time
        iteration_time = 0
        if time_limit is not None and time.time() - start_time > time_limit:
            logger.log("UNIPIPE EPOCH END TIMELIMIT OVER", cur_epoch + 1)
            break
        print("Epoch count:", cur_epoch)

        traindata_start_idx = 0
        time_uf = None
        time_ub = None

        while time_limit is not None and time.time() - start_time < time_limit:
            logger.log("UNIPIPE ITERATION START", total_iter_count + 1)
            iteration_start_time = time.time()
            # first take from test
            infer_count = 0

            try:
                infer_batch, inferbatch_deadline_list, missed_infer_at_datastream = teststream.extract(bs=ideal_infer_bs)
                infer_count = infer_batch[0].shape[0]
                inference_iter_count += 1

                # if missed_infer_at_datastream > missed_infer:
                    # missed_infer = missed_infer_at_datastream
                    # print(
                    #     "Ideal train batchsize from {0} to {1}".format(
                    #         ideal_train_batchsize, max(int(ideal_train_batchsize/2), 4)
                    #     )
                    # )
                    # ideal_train_batchsize = max(int(ideal_train_batchsize/2), 1)
                # else:
                    # missed_infer = missed_infer_at_datastream
                    # print(
                    #     "Ideal train batchsize from {0} to {1}".format(
                    #         ideal_train_batchsize, min(int(ideal_train_batchsize * 1.5), trainloader.batch_size)
                    #     )
                    # )
                    # ideal_train_batchsize = min(int(ideal_train_batchsize * 1.5), trainloader.batch_size)
            except StopIteration:
                pass

            total_served += infer_count
            #print(total_served, infer_count, time.time() - start_time)
            total_streamed += infer_count + missed_infer_at_datastream

            if not stop_train:
                train_batch = trainloader.dataset[
                    traindata_start_idx:min(traindata_start_idx+ideal_train_batchsize, len(trainloader.dataset))]
                traindata_start_idx += ideal_train_batchsize

            # some infer data is there, merge and pass to context
            # or training is done now to pass only infer data to context
            # or no infer data is in pipeline for now, so only training
            if infer_count > 0 and not stop_train:
                ft_images = torch.concat((infer_batch[0], train_batch[0]), axis=0).to(device)
            elif infer_count > 0 and stop_train:
                ft_images = infer_batch[0].to(device)
            else:
                ft_images = train_batch[0].to(device)

            logger.log("UNIPIPE TRAIN, INFER BS", ideal_train_batchsize, infer_count)
            logger.log("UNIPIPE MISSED INFER DATASTREAM", missed_infer)
            # to keep track how many infer request missed due to forward pass latency
            forward_pass_arrival_time = time.time()
            pred_amps, pred_phs = model(ft_images) #Forward pass
            forward_pass_done_time = time.time()

            # for ideal train batch size calculation
            time_uf = (forward_pass_done_time -  forward_pass_arrival_time)/ft_images.shape[0]
            # print(result_fiilup_list[3][total_served + j].shape)
            
            # before proceeding to backward pass release the inference results
            # by release means put them in result array
            # to avoid deadline miss as much as possible
            forward_pass_related_infer_miss = 0
            if infer_count > 0:
                true_amp = infer_batch[1]
                true_ph = infer_batch[2]
                for j in range(infer_count):
                    if inferbatch_deadline_list[j] <= forward_pass_done_time:
                        forward_pass_related_infer_miss += 1
                        continue
                    result_fiilup_list[0][total_served + j] = pred_amps[j].detach().to("cpu").numpy()
                    result_fiilup_list[1][total_served + j] = pred_phs[j].detach().to("cpu").numpy()
                    result_fiilup_list[2][total_served + j] = true_amp[j].detach().to("cpu").numpy()
                    result_fiilup_list[3][total_served + j] = true_ph[j].detach().to("cpu").numpy()
            # update total missed count
            logger.log("UNIPIPE MISSED INFER FORWARD LATENCY", forward_pass_related_infer_miss)
            total_missed -= infer_count - forward_pass_related_infer_miss

            if not stop_train:
                gt_amps = train_batch[1].to(device)
                gt_phs = train_batch[2].to(device)

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

                # calculate batch size according to performance model if not constant batch size experiment
                prev_ideal_train_bs = ideal_train_batchsize
                if not constant_bs:
                    ideal_train_batchsize = round(
                        teststream.get_deadlinesec() * (1-teststream.get_datarate()*time_uf) /\
                        (2 * (time_uf + time_ub))
                    )

                if ideal_train_batchsize != prev_ideal_train_bs:
                    logger.log("TRAINBATCH SIZE FROM", prev_ideal_train_bs, ideal_train_batchsize)
                if ideal_train_batchsize <= 0:
                    logger.log("TRAINBATCH SIZE TO <=0", teststream.get_deadlinesec(), 1-teststream.get_datarate()*time_uf, time_uf, time_ub)
                    ideal_train_batchsize = 1
                total_train_iter_count += 1

            total_iter_count += 1
            iteration_time += time.time() - iteration_start_time

            if traindata_start_idx >= len(trainloader.dataset):
                logger.log("TRAINDATASET CONSUMED AT EPOCH", cur_epoch + 1)
                stop_train = True
                break

        update_saved_model(model=model, path="/dev/shm/", name="{0}_e{1}.pth".format(chkpt_path[:-4], cur_epoch))
        #Divide cumulative loss by number of batches-- sli inaccurate because last batch is different size
        metrics['losses'].append([tot_loss/(total_train_iter_count + 1),loss_amp/(total_train_iter_count + 1),loss_ph/(total_train_iter_count + 1)])
        logger.log("TRAINING LOSS AT EPOCH", cur_epoch, tot_loss/(total_train_iter_count + 1),loss_amp/(total_train_iter_count + 1),loss_ph/(total_train_iter_count + 1))
        
        # update flag to stop training if training brings out minimal improvement
        # this condition needed because it is possible in current epoch not a single training is run
        if not stop_train:
            if abs(tot_loss/(total_train_iter_count) - previous_loss) <= parameters.LOSS_CHANGE_MIN_THRESHOLD:
                stop_train = True
                logger.log("UNIPIPE TRAINING STOP CONVERGENCE")
                interval_remaining_time = time.time() - interval_init_time
                logger.log("UNIPIPE INTERVAL REM. TIME", interval_remaining_time)
            previous_loss = tot_loss/(total_train_iter_count)

        logger.log("UNIPIPE ITER. TIME", iteration_time/total_iter_count)
        logger.log("UNIPIPE EPOCH END", cur_epoch + 1)
        if cur_epoch + 1 == parameters.EPOCHS:
            stop_train = True
            logger.log("MAX EPOCH DONE, NO TRAINING IN CURRENT INTERVAL")

    logger.log("UNIPIPE TRAINING TAKES", time.time() - train_start_time)
    logger.log("UNIPIPE TOTAL TIME:", time.time() - start_time)
    logger.log("TOTAL STREAMED", total_streamed)
    logger.log("TOTAL SERVED,MISSED", total_served, total_missed)
    logger.log(teststream.get_perf())
    return metrics, total_served, total_missed

# process validation dataset and return val loss
def validate(model, criterion, validloader, device, metrics):
    tot_val_loss = 0.0
    val_loss_amp = 0.0
    val_loss_ph = 0.0
    for j, batch in enumerate(validloader):
        ft_images = batch[0].to(device)
        amps = batch[1].to(device)
        phs = batch[2].to(device)
        pred_amps, pred_phs = model(ft_images) #Forward pass
    
        val_loss_a = criterion(pred_amps,amps) 
        val_loss_p = criterion(pred_phs,phs)
        val_loss = val_loss_a + val_loss_p
    
        tot_val_loss += val_loss.detach().item()
        val_loss_amp += val_loss_a.detach().item()
        val_loss_ph += val_loss_p.detach().item()

    return tot_val_loss / (j+1), val_loss_amp / (j+1), val_loss_ph / (j+1)

# process test dataset and return results
def test(model, testloader, device="cuda"):
    # to hold the result
    amps = []
    phs = []
    true_amps = []
    true_phs = []
    for _, batch in enumerate(testloader):
        ft_images =  batch[0].to(device)
        true_amp = copy.deepcopy(batch[1])
        true_ph = copy.deepcopy(batch[2])
        # print(batch[0].shape, batch[1].shape, batch[2].shape)
        amp, ph = model(ft_images)
        for j in range(ft_images.shape[0]):
            amps.append(amp[j].detach().to("cpu").numpy())
            phs.append(ph[j].detach().to("cpu").numpy())
            true_amps.append(true_amp[j].detach().to("cpu").numpy())
            true_phs.append(true_ph[j].detach().to("cpu").numpy())
        del batch
    
    point_size = 3
    overlap = 4*point_size
    # print(len(amps), amps[0].shape, len(true_amps), true_amps[0].shape)
    amp_error, ph_error = error_calculation.calc_error(amps=amps, phs=phs, true_amp=true_amps, true_ph=true_phs,
                                 point_size=point_size, overlap=overlap)
    
    # bring back model to training
    return amp_error, ph_error

# process test dataset and return results
def testloss(model, testloader, device="cuda"):
    # to hold the result
    amps = []
    phs = []
    true_amps = []
    true_phs = []
    
    loss_amp = 0
    loss_ph = 0
    loss_total = 0
    loss_func = torch.nn.L1Loss()
    for iter_count, batch in enumerate(testloader):
        ft_images =  batch[0].to(device)
        true_amp = copy.deepcopy(batch[1]).to(device)
        true_ph = copy.deepcopy(batch[2]).to(device)
        # print(batch[0].shape, batch[1].shape, batch[2].shape)
        amp, ph = model(ft_images)

        loss_a = loss_func(amp, true_amp).detach().item()
        loss_p = loss_func(ph, true_ph).detach().item()
        loss_t = loss_a + loss_p

        del batch

        loss_amp += loss_a
        loss_ph += loss_p
        loss_total += loss_t

    iter_count += 1

    # bring back model to training
    return loss_total/iter_count,  loss_amp/iter_count, loss_ph/iter_count


# process test dataset and return results
def test_calculatedinference(pred_gt_list, device="cuda"):
    point_size = 3
    overlap = 4*point_size
    # print(len(amps), amps[0].shape, len(true_amps), true_amps[0].shape)
    amp_error, ph_error = error_calculation.calc_error(amps=pred_gt_list[0], phs=pred_gt_list[1],
                                                       true_amp=pred_gt_list[2], true_ph=pred_gt_list[3],
                                 point_size=point_size, overlap=overlap)
    
    # bring back model to training
    return amp_error, ph_error


# process test dataset and return results
def time_limit_test(model, testloader, time_limit, device="cuda"):
    # to hold the result
    amps = []
    phs = []
    true_amps = []
    true_phs = []
    # for time limit testing
    total_served = 0
    total_missed = 0
    start_time = time.time()
    for _, batch in enumerate(testloader):
        ft_images =  batch[0].to(device)
        true_amp = batch[1]
        true_ph = batch[2]
        # print(batch[0].shape, batch[1].shape, batch[2].shape)
        if time.time() - start_time > time_limit:
            # we are assuming missing inference results are just random normal noise (white noise)
            # lower = 0
            # upper = 1
            # mu = 0.5
            # loc = 0.5 
            # amp = scipy.stats.truncnorm.rvs((lower-mu)/mu,(upper-mu)/mu,loc=loc,scale=mu,size=ft_images.shape)
            # ph = scipy.stats.truncnorm.rvs((lower-mu)/mu,(upper-mu)/mu,loc=loc,scale=mu,size=ft_images.shape)
            
            # we are assuming missing inference means black image
            void_image = np.zeros(shape=ft_images.shape, dtype=np.float32)
            for j in range(ft_images.shape[0]):
                amps.append(void_image[j])
                phs.append(void_image[j])
                true_amps.append(true_amp[j].detach().to("cpu").numpy())
                true_phs.append(true_ph[j].detach().to("cpu").numpy())
            total_missed += ft_images.shape[0]
        else:
            amp, ph = model(ft_images)
            for j in range(ft_images.shape[0]):
                amps.append(amp[j].detach().to("cpu").numpy())
                phs.append(ph[j].detach().to("cpu").numpy())
                true_amps.append(true_amp[j].detach().to("cpu").numpy())
                true_phs.append(true_ph[j].detach().to("cpu").numpy())
            total_served += ft_images.shape[0]
        
    point_size = 3
    overlap = 4*point_size
    # print(len(amps), amps[0].shape, len(true_amps), true_amps[0].shape)
    amp_error, ph_error = error_calculation.calc_error(amps=amps, phs=phs, true_amp=true_amps, true_ph=true_phs,
                                 point_size=point_size, overlap=overlap)
    
    # bring back model to training
    return amp_error, ph_error, total_served, total_missed

# process test dataset and return results
def test_time_constrained(model, teststream, result_fiilup_list, time_limit, device="cuda",
                          next_model="0", chkpt_dir=None, logger:fastlogger.FastLogger=None):
    # for time limit testing
    total_served = 0
    total_missed = len(result_fiilup_list[0])
    start_time = time.time()
    
    # to keep track if next model is loaded
    next_model_loaded = False
    # move model to GPU
    model.to(device)
    
    while time.time() - start_time < time_limit:
        batch, inferbatch_deadline_list, missed = teststream.extract()
        ft_images =  batch[0].to(device)
        true_amp = batch[1]
        true_ph = batch[2]
        # print(batch[0].shape, batch[1].shape, batch[2].shape)
        # print("infernece model is serving")
        logger.log("INFERENCE BATCH SIZE", ft_images.shape[0])
        logger.log("MISSED INFER DATASTREAM", missed)
        amp, ph = model(ft_images)
        # to keep track how much missed due to forward pass latency
        forward_pass_done_time = time.time()

        forward_pass_related_infer_miss = 0
        for j in range(ft_images.shape[0]):
            if total_served + j >= len(result_fiilup_list[0]):
                break
            if inferbatch_deadline_list[j] <= forward_pass_done_time:
                forward_pass_related_infer_miss += 1
                continue
            result_fiilup_list[0][total_served + j] = amp[j].detach().to("cpu").numpy()
            result_fiilup_list[1][total_served + j] = ph[j].detach().to("cpu").numpy()
            result_fiilup_list[2][total_served + j] = true_amp[j].detach().to("cpu").numpy()
            result_fiilup_list[3][total_served + j] = true_ph[j].detach().to("cpu").numpy()
        total_served += ft_images.shape[0]
        logger.log("MISSED INFER FORWARD LATENCY", forward_pass_related_infer_miss)
        total_missed -= ft_images.shape[0] - forward_pass_related_infer_miss
    
        # check after each serve that if model is updated
        model_load_time = time.time()
        if not next_model_loaded and os.path.exists(os.path.join("/dev/shm", "unipipe_exp_" + str(next_model) + "th_interval_modeltrained")):
            logger.log("MODEL UPDATE TO", next_model)
            print("inference process is swapping model, ", os.path.join(chkpt_dir, "inctrained_interaval{0}_model.pth".format(next_model)))
            model = torch.load(os.path.join(chkpt_dir, "inctrained_interaval{0}_model.pth".format(next_model)))
            model.to(device)
            next_model_loaded = True
            print("total served with prev: ", total_served, " total remaninig:", total_missed)
        # print(time.time() - model_load_time)

    # bring back model to training
    return total_served, total_missed

#Function to update saved model if validation loss is minimum
def update_saved_model(model, path, name, logger=None):
    if not os.path.isdir(path):
        os.mkdir(path)
    if logger is None:
        print("save model in ", os.path.join(path, name))
    else:
        logger.log("SAVE CHKPT ", os.path.join(path, name))
    torch.save(model, os.path.join(path, name))
