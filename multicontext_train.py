"""
 We assume training will be done with full overhead but through multicontext

 Therefore, old model will be alive and when training done will be redployed
 We will keep serving inference in same context once training done
 
 Objectives:
 1. Collect lost inference rate at each interval
 2. Collect mean inference accuracy, assume missed inference as 0 accuracy
"""

import argparse
import os
import shutil
import random
import torch
import numpy as np
import torch.utils
import torch.utils.data

import ptychonn.model
import ptychonn.dataset
import ptychonn.parameters
import ptychonn.process_funcs
import ptychonn.ipc

import logfast

PRETRAIN_FRACTION = 0.20
# pre train + incremental training in 4 interval
INC_TRAIN_INTERVAL = 5
PER_INTERVAL_TIME = 10#11#12#12.94336063


if __name__ == "__main__":
    # for reproducability
    # https://discuss.pytorch.org/t/training-reproducibility-problem/37143/3
    # https://vandurajan91.medium.com/random-seeds-and-reproducible-results-in-pytorch-211620301eba
    random.seed(ptychonn.parameters.SEED)
    torch.manual_seed(ptychonn.parameters.SEED)
    torch.cuda.manual_seed(ptychonn.parameters.SEED)
    torch.cuda.manual_seed_all(ptychonn.parameters.SEED)
    np.random.seed(1)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    # define arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--interval-duration", "-idur", type=int, required=True, help="length of interval in seconds")
    arg_parser.add_argument("--interval-count", "-icount", type=int, required=True, help="number of interval")
    arg_parser.add_argument("--csvlog-file", "-csvlog", type=str, required=True, help="name of csv log file")
    # get the arguments
    args = arg_parser.parse_args()

    # initiate the logger
    logger = logfast.fastlogger.FastLogger()

    dataset_dict = ptychonn.dataset.get_dataset(
        nvalid_percentage=ptychonn.parameters.VALID_PERCENTAGE,
        ntest_percentage=ptychonn.parameters.TEST_PERCENTAGE)

    train_data = dataset_dict["train"]
    valid_data = dataset_dict["valid"]

    #Training data
    X_train_tensor = torch.Tensor(train_data[0]) 
    Y_I_train_tensor = torch.Tensor(train_data[1]) 
    Y_phi_train_tensor = torch.Tensor(train_data[2])

    #Validation data
    X_valid_tensor = torch.Tensor(valid_data[0]) 
    Y_I_valid_tensor = torch.Tensor(valid_data[1]) 
    Y_phi_valid_tensor = torch.Tensor(valid_data[2])

    # print(X_train_tensor.shape, Y_I_train_tensor.shape, Y_phi_train_tensor.shape)

    train_data = torch.utils.data.TensorDataset(X_train_tensor,Y_I_train_tensor,Y_phi_train_tensor)
    valid_data = torch.utils.data.TensorDataset(X_valid_tensor,Y_I_valid_tensor,Y_phi_valid_tensor)

    print(X_train_tensor.shape, Y_I_train_tensor.shape, Y_phi_train_tensor.shape)
    print(X_valid_tensor.shape, Y_I_valid_tensor.shape, Y_phi_valid_tensor.shape)
    logger.log(X_train_tensor.shape, Y_I_train_tensor.shape, Y_phi_train_tensor.shape)
    logger.log(X_valid_tensor.shape, Y_I_valid_tensor.shape, Y_phi_valid_tensor.shape)

    performance_metrics = {"train time": [], "inference time": [], "miss rate": []}
    total_served = 0
    total_missed = 0

    import time

    # init the model
    model_train = ptychonn.model.recon_model()

    for interval_count in range(args.interval_count):
        # to realize when interval ends 
        interval_start_time = time.time()
        # to mark the interval starts, so inference process knows
        ptychonn.ipc.create_shm_marker("unipipe_exp_" + str(interval_count) + "th_interval_start")

        #download and load training data
        trainloader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(
            train_data, list(range(interval_count * len(train_data)//args.interval_count,
                             (interval_count + 1)* len(train_data)//args.interval_count))),
            batch_size=ptychonn.parameters.TRAIN_BATCH_SIZE, shuffle=False
        )

        # pretrain
        if interval_count == 0:
            # if we have done pretraining already with some model no need to redo it
            if os.path.exists(os.path.join("pretrained_model", "pretrained_bestmodel.pth")):
                model = torch.load(os.path.join("pretrained_model", "pretrained_bestmodel.pth"))
                continue

            validloader = torch.utils.data.DataLoader(
                valid_data,
                batch_size=ptychonn.parameters.TRAIN_BATCH_SIZE, shuffle=False)
            # train and save
            start_time = time.time()
            train_metrics = ptychonn.process_funcs.train(
                model=model_train, trainloader=trainloader, chkpt_path="model_multicontext/inctrained_interaval0_model.pth",
                epoch=ptychonn.parameters.PRETRAIN_EPOCHS, bs=ptychonn.parameters.TRAIN_BATCH_SIZE, do_validate=True, validloader=validloader,
                shm_signal_name="unipipe_exp_" + str(interval_count) + "th_interval_modeltrained", logger=logger
            )
            # also copy it to a standard named file
            shutil.copy("model_multicontext/inctrained_interaval0_model.pth", "pretrained_model/pretrained_bestmodel_multicontext.pth")
            performance_metrics["train time"].append(time.time() - start_time)

            # create a marker in /dev/shm for inference process to swap model
            ptychonn.ipc.create_shm_marker("unipipe_exp_" + str(interval_count) + "th_interval_modeltrained")
            continue

        # mark of interval start
        logger.log("INTERVAL START {0}".format(interval_count + 1))

        # incremental training
        # so all inference can be served, we can think that the test will run completely
        start_time = time.time()
        # we assume in an interval the training will start after half of interval
        # this half will be used to prepare the training data (generate ground truth)
        time.sleep(args.interval_duration/2)

        train_metrics = ptychonn.process_funcs.train(
            model=model_train, trainloader=trainloader, chkpt_path="model_multicontext/inctrained_interaval{0}_model.pth".format(interval_count),
            epoch=ptychonn.parameters.EPOCHS, bs=ptychonn.parameters.TRAIN_BATCH_SIZE, time_limit=args.interval_duration/2,
            shm_signal_name="unipipe_exp_" + str(interval_count) + "th_interval_modeltrained", logger=logger
        )

        spent_time = time.time() - start_time

        performance_metrics["train time"].append(spent_time)

        # wait for next interval
        while time.time() - interval_start_time < args.interval_duration:
            time.sleep((time.time() - interval_start_time)/2)

        # mark of interval start
        logger.log("INTERVAL END {0}".format(interval_count + 1))

    # average
    # performance_metrics["train time"] = performance_metrics["train time"]/(INC_TRAIN_INTERVAL * ptychonn.parameters.EPOCHS)
    # performance_metrics["inference time"] = performance_metrics["inference time"]/(INC_TRAIN_INTERVAL)
    # -1 as the first interval is for pretrained model generation
    # performance_metrics["miss rate"] = performance_metrics["miss rate"]/(INC_TRAIN_INTERVAL-1)

    # print(train_metrics)
    print("Interval\tTrain Time")
    for i, datarow in enumerate(performance_metrics["train time"]):
        print(i, "\t", datarow)

    print("\t".join([str(t) for t in performance_metrics["train time"]]))

    logger.log("TRAINING TIME", performance_metrics["train time"])
    logger.persist(args.csvlog_file[:-4] + "_train.log")
