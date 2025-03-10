"""
 We assume training will be done with full overhead but through unipipe
 We will keep serving inference in same context but partially trained model
 
 Objectives:
 1. Collect lost inference rate at each interval
 2. Collect mean inference accuracy, assume missed inference as 0 accuracy
"""

import argparse
import os
import shutil
import random
import copy
import time
import traceback
import torch
import numpy as np
import torch.utils
import torch.utils.data

import ptychonn.model
import ptychonn.dataset
import ptychonn.parameters
import ptychonn.ipc
import ptychonn.process_funcs
import ptychonn.error_calculation
import ptychonn.datastream

# for logging
import logfast.fastlogger


# def unipipe_traininfer(model, train_dataset, test_dataset, epoch_count):
#     # calculate batch size according to performance model

#     for i in range

#     return


def read_inferdata_batch(idx, bs):
    consumed = 0
    missed = 0
    ara = None
    for i in range(bs):
        try:
            # read it and add to batch
            if consumed == 0:
                ara = ptychonn.ipc.read_shm_data(
                    ptychonn.parameters.SHM_DATA_DIFFR_NAMEFMT.format(idx)
                ).reshape(1, 1, ptychonn.parameters.H, ptychonn.parameters.W)
            else:
                ara = np.vstack(
                    (
                        ara, ptychonn.ipc.read_shm_data(
                            ptychonn.parameters.SHM_DATA_DIFFR_NAMEFMT.format(idx)
                        ).reshape(1, 1, ptychonn.parameters.H, ptychonn.parameters.W)
                    ) 
                )

            # inference will be done only once
            # so delete
            ptychonn.ipc.remove_shm(ptychonn.parameters.SHM_DATA_DIFFR_NAMEFMT.format(idx))
            consumed += 1 
        except Exception as e:
            missed += 1
        idx += 1

    return idx, ara, consumed, missed


def read_traindata_batch(idx, cur_ipriteration, bs):
    cur_datafoldername = ptychonn.parameters.SHM_MARKER_FMT_GTGENERATION_FOLDER.format(cur_ipriteration)
    consumed = 0
    ara1 = ara2 = ara3 = None
    for i in range(bs):
        try:
            if consumed == 0:
                ara1 = ptychonn.ipc.read_shm_data(
                    os.path.join(
                        cur_datafoldername,
                        ptychonn.parameters.SHM_DATA_DIFFR_NAMEFMT.format(idx)
                    )
                ).reshape(1, 1, ptychonn.parameters.H, ptychonn.parameters.W)
                ara2 = ptychonn.ipc.read_shm_data(
                    os.path.join(
                        cur_datafoldername,
                        ptychonn.parameters.SHM_DATA_GEN_AMP_NAMEFMT.format(idx)
                    )
                ).reshape(1, 1, ptychonn.parameters.H, ptychonn.parameters.W)
                ara3 = ptychonn.ipc.read_shm_data(
                    os.path.join(
                        cur_datafoldername,
                        ptychonn.parameters.SHM_DATA_GEN_PHASE_NAMEFMT.format(idx)
                    )
                ).reshape(1, 1, ptychonn.parameters.H, ptychonn.parameters.W)
            else:
                ara1 = np.vstack(
                    (
                        ara1,
                        ptychonn.ipc.read_shm_data(
                            os.path.join(
                                cur_datafoldername,
                                ptychonn.parameters.SHM_DATA_DIFFR_NAMEFMT.format(idx)
                            )
                        ).reshape(1, 1, ptychonn.parameters.H, ptychonn.parameters.W)
                    ) 
                )
                ara2 = np.vstack(
                    (
                        ara2, ptychonn.ipc.read_shm_data(
                            os.path.join(
                                cur_datafoldername,
                                ptychonn.parameters.SHM_DATA_GEN_AMP_NAMEFMT.format(idx)
                            )
                        ).reshape(1, 1, ptychonn.parameters.H, ptychonn.parameters.W)
                    ) 
                )
                ara3 = np.vstack(
                    (
                        ara3, ptychonn.ipc.read_shm_data(
                            os.path.join(
                                cur_datafoldername,
                                ptychonn.parameters.SHM_DATA_GEN_PHASE_NAMEFMT.format(idx)
                            )
                        ).reshape(1, 1, ptychonn.parameters.H, ptychonn.parameters.W)
                    ) 
                )
            consumed += 1
        except Exception as e:
            pass
        
        idx += 1

    return idx, ara1, ara2, ara3

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

    # print the metadata of the experiments from parameters module
    for attr in ptychonn.parameters.__dict__:
        if type(attr) in [str, int, float] and not attr.startswith("__"):
            print(attr, "=", ptychonn.parameters.__dict__[attr])

    test_metrics = []
    performance_metrics = {"train time": [], "inference time": [], "miss rate": [], "miss rate stat datastreamer": []}

    # to create some datastructures beforehand
    # this is done to avoid some overhead when interval and make the scenario more realistic
    # in secnario we will just calculate the prediction and fill up a initiated array
    # error calculation will be done after all intervals are finished and we have result
    result_list = []
    void_image = None
    # void_image = np.zeros(shape=test_data[0][0].shape, dtype=np.float32)#np.random.normal(size=test_data[0][0].shape)#  np.random.normal(size=test_data[0][0].shape)
    # test loader to prefill data for later error calculation
    
    # make a result directory where generated images will be stored
    ptychonn.ipc.create_shm_folder(ptychonn.parameters.SHM_MARKER_NNRES_FOLDER)

    # init the model
    model = ptychonn.model.recon_model()

    # wait to synchronize time calculation with produce process
    producer_transmit_wait()
    
    # training state controller variable initiation
    start_time = time.time()
    cur_ipriteration = 0
    cur_interval = 0
    current_time = start_time
    cur_interval_start_time = current_time
    deadline_sec = args.deadline / 1000
    total_runtime = args.interval_count * deadline_sec
    total_consumed = 0
    consumed_batch = 0
    total_missed = 0
    missed_iprinterval = 0
    unittime_det_phase = True
    cur_infer_dataid = 0
    cur_train_dataid = 0
    print("unipipe start", start_time)

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
            # update the current inference idx and training data idx
            # the files are named in such a way that
            # t1 = time.time()
            for filename in os.listdir("/dev/shm"):
                if ".raw" in filename:
                    cur_infer_dataid = int(filename.split(".")[0])
            # print("first infer data selection takes {0}s".format(time.time() - t1))

            # t1 = time.time()
            for filename in sorted(
                os.listdir(
                    os.path.join(
                        "/dev/shm",
                        ptychonn.parameters.SHM_MARKER_FMT_GTGENERATION_FOLDER.format(cur_ipriteration)
                    )
                )
            ):
                if ".rawgti" in filename:
                    cur_train_dataid = int(filename.split(".")[0])
            # print("first train data selection takes {0}s".format(time.time() - t1))
            print(time.time(), "From consumer process : ", cur_ipriteration, cur_infer_dataid, cur_train_dataid)


        # pretrain stage
        if cur_interval == 0:
            # if we have done pretraining already with some model no need to redo it
            if os.path.exists(os.path.join("pretrained_model", "pretrained_bestmodel.pth")):
                model = torch.load(os.path.join("pretrained_model", "pretrained_bestmodel.pth"))
                continue

            # start_time = time.time()
            # test_metrics.append(ptychonn.process_funcs.test(model=model, testloader=testloader))
            # performance_metrics["inference time"].append(time.time() - start_time)
            continue

        # if unittime still undetermined 
        # take only one data sample 
        if unittime_det_phase:
            cur_train_dataid, train_data_in, train_data_gtamp, train_data_gtph = read_traindata_batch(
                cur_train_dataid, cur_ipriteration, bs=1)
            cur_infer_dataid, inferdata_in, consumed_batch, missed_iprinterval = read_inferdata_batch(cur_infer_dataid, bs=1)
            
            unittime_det_phase = False
            continue
        else:
            trainbs = ptychonn.parameters.TRAIN_BATCH_SIZE
            inferbs = ptychonn.parameters.INFERENCE_BATCH_SIZE
            # TODO:
            # put performance model call here to determine the batch size

            # reading the data
            cur_train_dataid, train_data_in, train_data_gtamp, train_data_gtph = read_traindata_batch(
                cur_train_dataid, cur_ipriteration, bs=ptychonn.parameters.TRAIN_BATCH_SIZE)
            cur_infer_dataid, inferdata_in, consumed_batch, missed_iprinterval = read_inferdata_batch(
                cur_infer_dataid,
                bs=ptychonn.parameters.INFERENCE_BATCH_SIZE)

        # TODO
        # put unipipe traininfer for one ipriteration data here
        # currently just wait code
        # print("sleeping for {0}s".format(args.ipr_throughput * deadline_sec / args.datarate))
        # print(args.ipr_throughput, deadline_sec, args.datarate)
        # time.sleep(
        #     args.ipr_throughput * deadline_sec / args.datarate
        # )

        total_consumed += consumed_batch
        total_missed += missed_iprinterval
        # print(consumed_batch, missed_iprinterval)


    print(total_consumed, total_missed)
    # calculate error
    # for interval_count in range(1, args.interval_count):
    #     point_size = 3
    #     overlap = 4*point_size
    #     amp_error, ph_error = ptychonn.error_calculation.calc_error(
    #         amps=result_list[interval_count][0], phs=result_list[interval_count][1],
    #         true_amp=result_list[interval_count][2], true_ph=result_list[interval_count][3],
    #         point_size=point_size, overlap=overlap
    #     )

    #     # to measure only the training quality
    #     if args.gtdefault:
    #         if performance_metrics["miss rate"][interval_count - 1] < 1-1e-16:    
    #             amp_error /= (1 - performance_metrics["miss rate"][interval_count - 1])
    #             ph_error /= (1 - performance_metrics["miss rate"][interval_count - 1])
    #         else:
    #             # how should we measure the training quality for the response
    #             # if no response is generated?
    #             # for now we are just setting a default high value
    #             amp_error = 0.01
    #             ph_error = 3

    #     test_metrics.append((amp_error, ph_error))

    # # average
    # # performance_metrics["train time"] = performance_metrics["train time"]/(INC_TRAIN_INTERVAL * ptychonn.parameters.EPOCHS)
    # # performance_metrics["inference time"] = performance_metrics["inference time"]/(INC_TRAIN_INTERVAL)
    # # -1 as the first interval is for pretrained model generation
    # # performance_metrics["miss rate"] = performance_metrics["miss rate"]/(INC_TRAIN_INTERVAL-1)

    # # printed in format to make copy to excel sheet easier
    # # first print the amp. errors
    # logger.log("AMP. ERROR", ",".join([str(entry[0]) for entry in test_metrics]))
    # logger.log("PH. ERROR", ",".join([str(entry[1]) for entry in test_metrics]))

    # logger.log("TRAINING TIME", performance_metrics["train time"])
    # logger.log("INFERENCE TIME", performance_metrics["inference time"])
    # logger.log("MISSRATE", performance_metrics["miss rate"])
    # logger.log("STREAM MISSRATE", performance_metrics["miss rate stat datastreamer"])

    # with open(args.csvlog_file, "a+") as fout:
    #     fout.write(str(args.interval_count))
    #     fout.write(",")
    #     fout.write(str(args.interval_duration))
    #     fout.write(",")
    #     fout.write(",".join([str(entry[0]) for entry in test_metrics]))
    #     fout.write(",")
    #     fout.write(",".join([str(entry[1]) for entry in test_metrics]))
    #     fout.write(",")
    #     fout.write(",".join([str(entry) for entry in performance_metrics["inference time"]]))
    #     fout.write(",")
    #     fout.write(",".join([str(entry) for entry in performance_metrics["miss rate"]]))
    #     fout.write(",")
    #     fout.write(",".join([str(entry) for entry in performance_metrics["miss rate stat datastreamer"]]))
    #     fout.write("\n")


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


    logger.persist(args.csvlog_file[:-4] + ".log")
