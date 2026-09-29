import argparse
import copy
import random
import torch
import torchvision
import time
import math
import model

from tqdm import tqdm
from nvitop import Device, ResourceMetricCollector


BATCH_SIZES = list(range(1, 65)) # [256, 128, 64, 32, 16, 8, 4, 2, 1]
LEARNING_RATE = 0.001
IMAGECOUNT_PER_RUN = 204800000 # 2048
ITERATION_COUNT_PER_RUN = 80 # 10000000
WARMUP_ITERATION = 20

DEVTYPE_GPU = 1
DEVTYPE_CPU = 0
DEVICE = DEVTYPE_GPU

gpu_utilization_dict = {}
gpu_utilization_list = []

# resource status collection
collector = ResourceMetricCollector(Device.cuda.all())

def instrument_w_nvtx(func):
    """decorator that causes an NVTX range to be recorded for the duration of the
    function call."""

    def wrapped_fn(*args, **kwargs):
        torch.cuda.nvtx.range_push(func.__qualname__)
        ret_val = func(*args, **kwargs)
        torch.cuda.nvtx.range_pop()
        return ret_val

    return wrapped_fn


# copied from https://raw.githubusercontent.com/pytorch/examples/main/imagenet/main.py
model_names = sorted(name for name in torchvision.models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision.models.__dict__[name]))

@instrument_w_nvtx
def foo(nn_model: torch.nn.Module, input_data):
    return nn_model(input_data)
    

# will do full update through standard pytorch optim module
def training_all_param_update(nn_model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, batch_size: int):
    global gpu_utilization_list
    # emptying the list before starting
    gpu_utilization_list = []
    # empty the cuda cache done by torch
    torch.cuda.empty_cache()
    # for collection sanity test
    data_collection_count = 0

    # start the resource collector with tag
    # We will collect one sample per inference iteration
    # two sample per 
    collector.start(tag="test")
    # NOT DOING DAEMONIZING
    # daemonize it, this returns a threading.thread object
    # will GIL cause issue? not sure yet
    # IT IS AN ISSUE MOST PROBABLY, daemon thread not getting invoked for given interval
    # daemon = collector.daemonize(on_collect, interval=0.1, on_stop=None)

    # move to GPU
    if DEVICE == DEVTYPE_GPU:
        nn_model.cuda()
    # we are doing iteration steps benchmarking so no need to train more than one epoch

    # this is for only inference forward
    nn_model.eval()
    evalfw_time_count = 0
    eval_iter_count = 0 
    infer_data_load_time = 0
    data_load_start_time = time.time()
    for input_data, label in tqdm(dataloader, leave=False):     
        try:
            if DEVICE == DEVTYPE_GPU:
                input_data = input_data.cuda()
                torch.cuda.synchronize()

            infer_data_load_time += time.time() - data_load_start_time
 
            t = time.time()
            fw = foo(nn_model, input_data)
            if DEVICE == DEVTYPE_GPU:
                torch.cuda.synchronize()
            if eval_iter_count >= WARMUP_ITERATION:
               evalfw_time_count += time.time() - t

            t = time.time()
            # collect metrics
            # collecting after measuring the time to not affect time measurement due to metric collection overhead
            # As we are collecting mean, we should still see the effect from last sampling 
            metrics = collector.collect()

            gpu_utilization_list.append(
                [
                    metrics["test/timestamp"],
                    metrics["test/host/cpu_percent (%)/max"],
                    metrics["test/cuda:0 (gpu:0)/gpu_utilization (%)/max"],
                    metrics["test/cuda:0 (gpu:0)/memory_used (MiB)/max"]
                ]
            )
            data_collection_count += 1
        except torch.cuda.OutOfMemoryError as e:
            print(
                "batch size {0} for imagenet 3x224x224 is not suitable for inference with GPU memory {1}GB".format(
                    batch_size, torch.cuda.mem_get_info()[1]>>30
                )
            )
            evalfw_time_count = math.nan
            eval_iter_count = 1
            break
        eval_iter_count += 1
        if eval_iter_count > IMAGECOUNT_PER_RUN/batch_size or eval_iter_count > ITERATION_COUNT_PER_RUN:
            break
        data_load_start_time = time.time()
    # print("inference completed successfully")
    # exit()
    # to reinit every thing before training benchmarking
    # clear collector status
    collector.clear()
    collector.stop(tag="test")
    # move to CPU
    nn_model.cpu()
    # empty the cuda cache done by torch
    torch.cuda.empty_cache()
    # start collector
    collector.start(tag="test")
    # move to GPU
    if DEVICE == DEVTYPE_GPU:
        nn_model.cuda()
    # init loss and optimizer
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(nn_model.parameters(), lr=LEARNING_RATE)

    # this is for loss calculated forward backward
    nn_model.train()
    iter_count = 0
    fw_time_count = 0
    bw_time_count = 0
    train_data_load_time = 0
    data_load_start_time = time.time()
    for input_data, label in tqdm(dataloader, leave=False):
        # zero the optimizer grad
        optimizer.zero_grad()
        # calculate loss of model
        try:
            if DEVICE == DEVTYPE_GPU:
                input_data = input_data.cuda()
                label = label.cuda()
                torch.cuda.synchronize()

            train_data_load_time += time.time() - data_load_start_time
            t = time.time()
            fw = nn_model(input_data)
            if DEVICE == DEVTYPE_GPU:
                torch.cuda.synchronize()
            if iter_count >= WARMUP_ITERATION:
                fw_time_count += time.time() - t
            # collect metrics
            metrics = collector.collect()
            gpu_utilization_list.append(
                [
                    metrics["test/timestamp"],
                    metrics["test/host/cpu_percent (%)/max"],
                    metrics["test/cuda:0 (gpu:0)/gpu_utilization (%)/max"],
                    metrics["test/cuda:0 (gpu:0)/memory_used (MiB)/max"]
                ]
            )
            data_collection_count += 1
        except torch.cuda.OutOfMemoryError as e:
            print(
                "batch size {0} for imagenet 3x224x224 is not suitable for forward pass and loss calc with GPU memory {1}GB".format(
                    batch_size, torch.cuda.mem_get_info()[1]>>30
                )
            )
            fw_time_count = math.nan
            bw_time_count = math.nan
            iter_count = 1
            break

        # backprop
        try:
            t = time.time()
            # print(batch_size, label[:,0].shape, label[:,1].shape)
            # print(fw[0].shape, label[:][.shape, label[:][1].shape, fw[1].shape)
            loss = loss_func(fw[0], label[:,1]) + loss_func(fw[1], label[:,1]) 
            loss.backward()
            # optimization step
            optimizer.step()
            # append the loss
            loss_val = loss.data.cpu().item()
            if DEVICE == DEVTYPE_GPU:
                torch.cuda.synchronize()
            # per iteration loss record
            if iter_count >= WARMUP_ITERATION:
                bw_time_count += time.time() - t
            # collect metrics
            metrics = collector.collect()
            gpu_utilization_list.append(
                [
                    metrics["test/timestamp"],
                    metrics["test/host/cpu_percent (%)/max"],
                    metrics["test/cuda:0 (gpu:0)/gpu_utilization (%)/max"],
                    metrics["test/cuda:0 (gpu:0)/memory_used (MiB)/max"]
                ]
            )
            data_collection_count += 1
        except torch.cuda.OutOfMemoryError as e:
            print(
                "batch size {0} for imagenet 3x224x224 is not suitable for backward pass and update with GPU memory {1}GB".format(
                    batch_size, torch.cuda.mem_get_info()[1]>>30
                )
            )
            bw_time_count = math.nan
            iter_count = 1
            break

        iter_count += 1
        if iter_count > IMAGECOUNT_PER_RUN/batch_size or iter_count > ITERATION_COUNT_PER_RUN:
            break
        data_load_start_time = time.time()
        # per epoch loss record
        # print("average forward time {0}s with batch size {1}".format(fw_time_count/iter_count, batch_size))
        # print("average loss calc time {0}s with batch size {1}".format(loss_time_count/iter_count, batch_size))
        # print("average backprop time {0}s with batch size {1}".format(bw_time_count/iter_count, batch_size))
        # print("average optimizer step time {0}s with batch size {1}".format(optstep_time_count/iter_count, batch_size))

    # clear collector status
    collector.clear()
    collector.stop(tag="test")

    del nn_model
    eval_iter_count -= WARMUP_ITERATION
    iter_count -= WARMUP_ITERATION
    torch.cuda.empty_cache()
    # print("\n\n", data_collection_count, "\n\n")
    return evalfw_time_count/eval_iter_count, evalfw_time_count/(eval_iter_count*batch_size), fw_time_count/iter_count, fw_time_count/(iter_count*batch_size), bw_time_count/iter_count, bw_time_count/(iter_count*batch_size), infer_data_load_time/(iter_count*batch_size), train_data_load_time/(iter_count*batch_size)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model-type", "-type", type=str, choices=["1.25M", "5M", "10M", "20M", "100M", "200M"], help="which model to choose", required=True)
    args = arg_parser.parse_args()

    model_type = args.model_type
    model_name = model.get_model_name_from_type(model_type)

    benchmark_dict = {}
    gpu_utilization_dict = {}

    # create an artificial dataset
    tensor_dataset = torch.utils.data.TensorDataset(
        torch.randn((BATCH_SIZES[-1]*ITERATION_COUNT_PER_RUN, 1, 64, 64), dtype=torch.float),
        torch.randn((BATCH_SIZES[-1]*ITERATION_COUNT_PER_RUN, 2, 1, 64, 64), dtype=torch.float)
    )

    for batch_size in BATCH_SIZES:
        gpu_utilization_list = []
        random.seed(3400)

        data_dict_per_iteration = {}
        data_dict_per_epoch = {}
        
        dataloader = torch.utils.data.DataLoader(dataset=tensor_dataset, batch_size=batch_size)
        # model
        nn_model = model.get_model(type_name=model_type)
        # nn_model.fc = torch.nn.Linear(512, NUMBER_OF_CLASSES)
        
        evalfw, evalfw_per_samp, fw, fw_per_samp, bw, bw_per_samp, inferdataload_per_sample, traindata_load_per_sample  = training_all_param_update(
            nn_model=copy.deepcopy(nn_model), dataloader=dataloader, batch_size=batch_size
        )

        benchmark_dict[batch_size] = [evalfw, evalfw_per_samp, fw, fw_per_samp, bw, bw_per_samp, inferdataload_per_sample, traindata_load_per_sample]
        gpu_utilization_dict[batch_size] = copy.deepcopy(gpu_utilization_list)

    with open("benchmark_{0}_nn_step.csv".format(model_name), "w") as fout:
        fout.write("Batch Size\tInference\tInference Per Sample\tForward\tForward Per Sample\tBackward\tBackward Per Sample\tInfer Data Load\tTrain data load\n")
        for batch_size in BATCH_SIZES:
            data_list = benchmark_dict[batch_size]
            fout.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n".format(
                    batch_size, data_list[0], data_list[1], data_list[2], data_list[3], data_list[4], data_list[5], data_list[6], data_list[7]
                )
            )

    with open("benchmark_{0}_gpu_utilization.csv".format(model_name), "w") as fout:
            for batch_size in BATCH_SIZES:
                fout.write("{0}\t{0}\t{0}\t".format(batch_size))
            fout.write("\n")
            for batch_size in BATCH_SIZES:
                fout.write("cpu usage(%)\tgpu usage(%)\tgpu memory usage (MiB)\t".format(batch_size))
            fout.write("\n")

            maxlen = max([len(gpu_utilization_dict[batch_size]) for batch_size in gpu_utilization_dict])
            for i in range(maxlen):
                for batch_size in BATCH_SIZES:
                    if batch_size in gpu_utilization_dict:
                        if i < len(gpu_utilization_dict[batch_size]):
                            data_list = gpu_utilization_dict[batch_size][i]
                            fout.write("{0}\t{1}\t{2}\t".format(data_list[1], data_list[2], data_list[3]))
                        else:
                            fout.write("{0}\t{1}\t{2}\t".format(math.nan, math.nan, math.nan))
                fout.write("\n")

