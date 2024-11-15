"""
 We assume training will be done with full overhead
 
 Therefore, each interval window will have least amount of time to serve inferences

 Objectives:
 1. Collect mean inference accuracy, assume missed inference as 0 accuracy
"""

import argparse
import os
import shutil
import random
import copy
import time
import math
import torch
import numpy as np
import torch.utils
import torch.utils.data

import ptychonn.model
import ptychonn.dataset
import ptychonn.parameters
import ptychonn.process_funcs
import ptychonn.error_calculation
import ptychonn.datastream

import logfast


if __name__ == "__main__":
    # for reproducability
    random.seed(2661)
    torch.manual_seed(2661)
    torch.cuda.manual_seed(2661)
    np.random.seed(2661)
    torch.backends.cudnn.deterministic = True

    # define arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--interval-duration", "-idur", type=int, required=True, help="length of interval in seconds")
    arg_parser.add_argument("--interval-count", "-icount", type=int, required=True, help="number of interval")
    arg_parser.add_argument("--datarate", "-drate", type=int, required=True, help="request/datasample per second")
    arg_parser.add_argument("--deadline", "-dead", type=int, required=True, help="each request deadline after arrival in millisecond")
    arg_parser.add_argument("--gtdefault", "-gtd", action="store_true", help="what to take as default response for missed request")
    arg_parser.add_argument("--csvlog-file", "-csvlog", type=str, required=True, help="name of csv log file")
    # get the arguments
    args = arg_parser.parse_args()

    # initiate the logger
    logger = logfast.fastlogger.FastLogger()

    dataset_dict = ptychonn.dataset.get_dataset(nlines=161, nvalid_percentage=20, ntest_percentage=10)

    train_data = dataset_dict["train"]
    valid_data = dataset_dict["valid"]
    test_data = dataset_dict["test"]

    #Training data
    X_train_tensor = torch.Tensor(train_data[0]) 
    Y_I_train_tensor = torch.Tensor(train_data[1]) 
    Y_phi_train_tensor = torch.Tensor(train_data[2])

    #Validation data
    X_valid_tensor = torch.Tensor(valid_data[0]) 
    Y_I_valid_tensor = torch.Tensor(valid_data[1]) 
    Y_phi_valid_tensor = torch.Tensor(valid_data[2])

    #Test data
    X_test_tensor = torch.Tensor(test_data[0]) 
    Y_I_test_tensor = torch.Tensor(test_data[1]) 
    Y_phi_test_tensor = torch.Tensor(test_data[2])

    # print(X_train_tensor.shape, Y_I_train_tensor.shape, Y_phi_train_tensor.shape)

    train_data = torch.utils.data.TensorDataset(X_train_tensor,Y_I_train_tensor,Y_phi_train_tensor)
    valid_data = torch.utils.data.TensorDataset(X_valid_tensor,Y_I_valid_tensor,Y_phi_valid_tensor)
    test_data = torch.utils.data.TensorDataset(X_test_tensor, Y_I_test_tensor, Y_phi_test_tensor)

    logger.log(X_train_tensor.shape, Y_I_train_tensor.shape, Y_phi_train_tensor.shape)
    logger.log(X_valid_tensor.shape, Y_I_valid_tensor.shape, Y_phi_valid_tensor.shape)
    logger.log(X_test_tensor.shape, Y_I_test_tensor.shape, Y_phi_test_tensor.shape)

    test_metrics = []
    performance_metrics = {"train time": [], "inference time": [], "miss rate": [], "miss rate stat datastreamer": []}
    
    # to create some datastructures beforehand
    # this is done to avoid some overhead when interval and make the scenario more realistic
    # in secnario we will just calculate the prediction and fill up a initiated array
    # error calculation will be done after all intervals are finished and we have result
    result_list = []
    void_image = np.zeros(shape=test_data[0][0].shape, dtype=np.float32)#np.random.normal(size=test_data[0][0].shape)#  np.random.normal(size=test_data[0][0].shape)
    # test loader to prefill data for later error calculation
    testloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1, shuffle=True)
    testloader_iter = iter(testloader)
    for interval_count in range(args.interval_count):
        result_list.append([[], [], [], []])
        #same for test
        #download and load training data
        for j in range(args.datarate * (args.interval_duration + 1)):
            try:
                batch = next(testloader_iter)
            except StopIteration:
                testloader = torch.utils.data.DataLoader(
                    test_data,
                    batch_size=1, shuffle=True)

            # if args.gtdefault ground truth is default response
            # needed to evaluate just the training quality
            # else fill with black frame
            if args.gtdefault:
                result_list[-1][0].append(copy.deepcopy(batch[1].numpy()[0]))
                result_list[-1][1].append(copy.deepcopy(batch[2].numpy()[0]))
            else:
                result_list[-1][0].append(copy.deepcopy(void_image))
                result_list[-1][1].append(copy.deepcopy(void_image))

            result_list[-1][2].append(copy.deepcopy(batch[1].numpy()[0]))
            result_list[-1][3].append(copy.deepcopy(batch[2].numpy()[0]))
            # print(result_list[-1][2][-1].shape)

    # init the model
    model = ptychonn.model.recon_model()
    
    for interval_count in range(args.interval_count):
        #download and load training data
        trainloader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(
                train_data,
                list(range(interval_count * len(train_data)//args.interval_count,
                           (interval_count + 1)* len(train_data)//args.interval_count)
                )
            ),
            batch_size=ptychonn.parameters.TRAIN_BATCH_SIZE, shuffle=False
        )

        # pretrain
        if interval_count == 0:
            # if we have done pretraining already with some model no need to redo it
            if os.path.exists(os.path.join("pretrained_model", "pretrained_bestmodel.pth")):
                model = torch.load(os.path.join("pretrained_model", "pretrained_bestmodel.pth"))
                continue
            #same for test
            #download and load training data
            testloader = torch.utils.data.DataLoader(
                test_data,
                batch_size=ptychonn.parameters.INFERENCE_BATCH_SIZE, shuffle=False, num_workers=1)

            validloader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(
                    valid_data,
                    list(range(interval_count * len(valid_data)//args.interval_count,
                            (interval_count + 1)* len(valid_data)//args.interval_count)
                    )
                ),
                batch_size=ptychonn.parameters.TRAIN_BATCH_SIZE, shuffle=False
            )
            # train and save
            start_time = time.time()
            train_metrics = ptychonn.process_funcs.train(
                model=model, trainloader=trainloader, chkpt_path="pretrained_model/pretrained_bestmodel.pth",
                epoch=ptychonn.parameters.PRETRAIN_EPOCHS, bs=ptychonn.parameters.TRAIN_BATCH_SIZE, do_validate=True, validloader=validloader, logger=logger)
            performance_metrics["train time"].append(time.time() - start_time)
            # also copy it to a standard named file
            # shutil.copy("model_pretrained/pretrained_bestmodel.pth", "pretrained_model/pretrained_bestmodel.pth")

            performance_metrics["train time"].append(time.time() - start_time)
            # test
            # print(len(train_data), len(valid_data), len(test_data))
            start_time = time.time()
            test_metrics.append(ptychonn.process_funcs.test(model=model, testloader=testloader))
            performance_metrics["inference time"].append(time.time() - start_time)
            continue

        # custom continuous data producer stream
        teststream = ptychonn.datastream.DataStream(
            datarate=args.datarate,
            deadline_sec=args.deadline/1000, dataset=test_data)

        # mark of interval start
        logger.log("INTERVAL START {0}".format(interval_count + 1))

        # incremental training
        # all inference may not be served as training will take time
        start_time = time.time()
        # to signal that continuous data stream should start
        logger.log("WORSTCASE DATASTREAM START")
        spent_time = 0
        performance_metrics["train time"].append(spent_time)

        # test with timelimit
        remaining_time = args.interval_duration - spent_time
        start_time = time.time()
        served, missed = ptychonn.process_funcs.test_time_constrained(
            model=model, teststream=teststream, time_limit=remaining_time, result_fiilup_list=result_list[interval_count], logger=logger)
        performance_metrics["inference time"].append(time.time() - start_time)
        performance_metrics["miss rate"].append(missed/(missed + served))
        try:
            performance_metrics["miss rate stat datastreamer"].append(teststream.get_perf()[2] / teststream.get_perf()[0])
        except ZeroDivisionError:
            performance_metrics["miss rate stat datastreamer"].append(1)

        # mark of interval start
        logger.log("INTERVAL END {0}".format(interval_count + 1))

    # average
    # performance_metrics["train time"] = performance_metrics["train time"]/(INC_TRAIN_INTERVAL * ptychonn.parameters.EPOCHS)
    # performance_metrics["inference time"] = performance_metrics["inference time"]/(INC_TRAIN_INTERVAL)
    # -1 as the first interval is for pretrained model generation
    # performance_metrics["miss rate"] = performance_metrics["miss rate"]/(INC_TRAIN_INTERVAL-1)
    # calculate error
    for interval_count in range(1, args.interval_count):
        point_size = 3
        overlap = 4*point_size
        amp_error, ph_error = ptychonn.error_calculation.calc_error(
            amps=result_list[interval_count][0], phs=result_list[interval_count][1],
            true_amp=result_list[interval_count][2], true_ph=result_list[interval_count][3],
            point_size=point_size, overlap=overlap
        )

        # to measure only the training quality
        if args.gtdefault:
            if performance_metrics["miss rate"][interval_count - 1] < 1-1e-16:    
                amp_error /= (1 - performance_metrics["miss rate"][interval_count - 1])
                ph_error /= (1 - performance_metrics["miss rate"][interval_count - 1])
            else:
                # how should we measure the training quality for the response
                # if no response is generated?
                # for now we are just setting a default high value
                amp_error = 0.01
                ph_error = 3

        test_metrics.append((amp_error, ph_error))

    logger.log("AMP. ERROR", ",".join([str(entry[0]) for entry in test_metrics]))
    logger.log("PH. ERROR", ",".join([str(entry[1]) for entry in test_metrics]))

    logger.log("TRAINING TIME", performance_metrics["train time"])
    logger.log("INFERENCE TIME", performance_metrics["inference time"])
    logger.log("MISSRATE", performance_metrics["miss rate"])
    logger.log("STREAM MISSRATE", performance_metrics["miss rate stat datastreamer"])

    logger.persist(args.csvlog_file[:-4] + ".log")

    with open(args.csvlog_file, "a+") as fout:
        fout.write(str(args.interval_count))
        fout.write(",")
        fout.write(str(args.interval_duration))
        fout.write(",")
        fout.write(",".join([str(entry[0]) for entry in test_metrics]))
        fout.write(",")
        fout.write(",".join([str(entry[1]) for entry in test_metrics]))
        fout.write(",")
        fout.write(",".join([str(entry) for entry in performance_metrics["inference time"]]))
        fout.write(",")
        fout.write(",".join([str(entry) for entry in performance_metrics["miss rate"]]))
        fout.write(",")
        fout.write(",".join([str(entry) for entry in performance_metrics["miss rate stat datastreamer"]]))
        fout.write("\n")
