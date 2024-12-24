"""
 We assume training will be done with 0 overhead
 
 Therefore, each interval window will be able to serve
 maximum inference requests possible

 Objectives:
 1. Collect mean inference accuracy, assume missed inference as 0 accuracy
"""

import os
import copy
import argparse
import random
import time
import torch
import numpy as np
import torch.utils
import torch.utils.data

import ptychonn.model
import ptychonn.dataset
import ptychonn.parameters
import ptychonn.process_funcs
import ptychonn.datastream

import logfast

PRETRAIN_FRACTION = 0.20
# pre train + incremental training in 4 interval
INC_TRAIN_INTERVAL = 5


if __name__ == "__main__":
    # for reproducability
    random.seed(2661)
    torch.manual_seed(2661)
    torch.cuda.manual_seed(2661)
    np.random.seed(2661)
    # print the metadata of the experiments from parameters module
    for attr in ptychonn.parameters.__dict__:
        if type(attr) in [str, int, float] and not attr.startswith("__"):
            print(attr, "=", ptychonn.parameters.__dict__[attr])

    # define arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--interval-duration", "-idur", type=int, required=True, help="length of interval in seconds")
    arg_parser.add_argument("--interval-count", "-icount", type=int, required=True, help="number of interval")
    arg_parser.add_argument("--datarate", "-drate", type=int, required=True, help="request/datasample per second")
    arg_parser.add_argument("--deadline", "-dead", type=int, required=True, help="each request deadline after arrival in millisecond")
    arg_parser.add_argument("--batch-size", "-bs", type=int, default=None, help="batch size of test stream")
    arg_parser.add_argument("--gtdefault", "-gtd", action="store_true", help="what to take as default response for missed request")
    arg_parser.add_argument("--csvlog-file", "-csvlog", type=str, required=True, help="name of csv log file")
    
    # get the arguments
    args = arg_parser.parse_args()

    # initiate the logger
    logger = logfast.fastlogger.FastLogger()

    dataset_dict = ptychonn.dataset.get_dataset(nlines=161, nvalid_percentage=20, ntest_percentage=10)

    test_data = dataset_dict["test"]

    #Test data
    X_test_tensor = torch.Tensor(test_data[0]) 
    Y_I_test_tensor = torch.Tensor(test_data[1]) 
    Y_phi_test_tensor = torch.Tensor(test_data[2])

    print(X_test_tensor.shape, Y_I_test_tensor.shape, Y_phi_test_tensor.shape)
    logger.log(X_test_tensor.shape, Y_I_test_tensor.shape, Y_phi_test_tensor.shape)

    # print(X_train_tensor.shape, Y_I_train_tensor.shape, Y_phi_train_tensor.shape)

    test_data = torch.utils.data.TensorDataset(X_test_tensor, Y_I_test_tensor, Y_phi_test_tensor)

    test_metrics = []
    performance_metrics = {"inference time": [], "missed": [], "served": [], "miss rate": [], "miss rate stat datastreamer": []}

    # init the model
    model = ptychonn.model.recon_model()

    # to create some datastructures beforehand
    # this is done to avoid some overhead when interval and make the scenario more realistic
    # in secnario we will just calculate the prediction and fill up a initiated array
    # error calculation will be done after all intervals are finished and we have result
    result_list = []
    void_image = np.zeros(shape=test_data[0][0].shape, dtype=np.float32)#np.random.normal(size=test_data[0][0].shape)#  np.random.normal(size=test_data[0][0].shape)
    # test loader to prefill data for later error calculation
    testloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_data, list(range(0, len(test_data)//args.interval_count))),
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
                    torch.utils.data.Subset(
                        test_data, list(range(interval_count * len(test_data)//args.interval_count,
                             (interval_count + 1)* len(test_data)//args.interval_count))),
                    batch_size=1, shuffle=True)
                testloader_iter = iter(testloader)
                batch = next(testloader_iter)

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

    interval_start_time = time.time()
    for interval_count in range(args.interval_count):
        # to understand if the next interval has started
        # marked by training process
        # we could do it by interval start time condition, which one is better?
        while not os.path.exists(os.path.join("/dev/shm", "unipipe_exp_" + str(interval_count) + "th_interval_start")):
            time.sleep(args.interval_duration / args.interval_count)

        teststream = ptychonn.datastream.DataStream(
            datarate=args.datarate,
            deadline_sec=args.deadline/1000, dataset=torch.utils.data.Subset(
                            test_data, list(range(interval_count * len(test_data)//args.interval_count,
                                                  (interval_count + 1)* len(test_data)//args.interval_count))))
        interval_start_time = time.time()
        # to signal that continuous data stream should start
        logger.log("MULTICONTEXT INFER DATASTREAM START")
        # mark of interval start
        logger.log("INTERVAL START {0}".format(interval_count + 1))
        teststream.start_stream()

        # following construct is to wait for 1th interval to start without causing CPU consumption
        # first wait until 1th interval start then continue to next iteration
        # only 1st iteration will cause the loop to loop
        while not os.path.exists(os.path.join("/dev/shm", "unipipe_exp_1th_interval_start")):
            time.sleep((time.time() - interval_start_time) / 2)
        logger.log("MULTICONTEXT INFER LOADABLE MODEL READY")
        if interval_count == 0:
            continue

        # load the existing model trained on previous epoch
        # for 1st interval pretrained model is loaded
        logger.log("MULTICONTEXT INFER LOADING UPDATED MODEL")
        if interval_count == 1:
            print("inference process is loading model, ", os.path.join("pretrained_model", "pretrained_bestmodel.pth"))
            model = torch.load(os.path.join("pretrained_model", "pretrained_bestmodel.pth"))
        else:
            print("inference process is loading model, ", os.path.join("model_multicontext", "inctrained_interaval{0}_model.pth".format(interval_count - 1)))
            model = torch.load(os.path.join("model_multicontext", "inctrained_interaval{0}_model.pth".format(interval_count - 1)))

        # test
        start_time = time.time()
        served, missed = ptychonn.process_funcs.test_time_constrained(
            model=model, teststream=teststream, next_model=str(interval_count),
            chkpt_dir="model_multicontext", time_limit=args.interval_duration, result_fiilup_list=result_list[interval_count], logger=logger
        )
        performance_metrics["missed"].append(missed)
        performance_metrics["served"].append(served)
        performance_metrics["inference time"].append(time.time() - start_time)
        performance_metrics["miss rate"].append(missed / (missed + served))
        try:
            performance_metrics["miss rate stat datastreamer"].append(teststream.get_perf()[2] / teststream.get_perf()[0])
        except ZeroDivisionError:
            performance_metrics["miss rate stat datastreamer"].append(1)
        
        # mark of interval start
        logger.log("INTERVAL END {0}".format(interval_count + 1))
        print("Interval {0} took {1}s".format(interval_count, time.time() - interval_start_time))


    # average
    # performance_metrics["train time"] = performance_metrics["train time"]/(INC_TRAIN_INTERVAL * ptychonn.parameters.EPOCHS)
    # performance_metrics["inference time"] = performance_metrics["inference time"]/(INC_TRAIN_INTERVAL)

    for interval_count in range(1, args.interval_count):
        point_size = 3
        overlap = 4*point_size
        # print(result_list[interval_count][0].shape, result_list[interval_count][1].shape, result_list[interval_count][2].shape, result_list[interval_count][3].shape)
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


    print("Interval\tAmp error\tPhase error\tMiss Rate\tInfer Time")
    for i, entry in enumerate(performance_metrics["inference time"]):
        amp_error = test_metrics[i][0]
        ph_error = test_metrics[i][1]
        missrate = performance_metrics["miss rate"][i]
        infer_time = entry

        print(i+1, "\t", amp_error, "\t", ph_error, "\t", missrate, "\t", infer_time)

    print("Amp error", "\t".join([str(elm[0]) for elm in test_metrics]))
    print("ph error", "\t".join([str(elm[1]) for elm in test_metrics]))
    print("miss rate", performance_metrics["miss rate"])
    print("inf time: ", performance_metrics["inference time"])
    print("miss rate with data streamer overhead: ", performance_metrics["miss rate stat datastreamer"])

    logger.log("AMP. ERROR", ",".join([str(entry[0]) for entry in test_metrics]))
    logger.log("PH. ERROR", ",".join([str(entry[1]) for entry in test_metrics]))

    logger.log("INFERENCE TIME", performance_metrics["inference time"])
    logger.log("MISSRATE", performance_metrics["miss rate"])
    logger.log("STREAM MISSRATE", performance_metrics["miss rate stat datastreamer"])

    logger.persist(args.csvlog_file[:-4] + "_infer.log")

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
