import os
import copy
import random
import torch
import torchvision
import math
import argparse
import statistics

from tqdm import tqdm
import time

ITERATION_COUNT_PER_RUN = 200
WARMUP_ITERATION = 30

# copied from https://raw.githubusercontent.com/pytorch/examples/main/imagenet/main.py
model_names = sorted(name for name in torchvision.models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision.models.__dict__[name]))


# will do full update through standard pytorch optim module
def benchmark(nn_model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, batch_size: int, eval:bool):
    # move to GPU
    nn_model.cuda()
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(nn_model.parameters(), lr=LEARNING_RATE)
    # we are doing iteration steps benchmarking so no need to train more than one epoch

    forward_time_list = []
    backward_time_list = []
    dataload_time_count = 0
    dataload_event_count = 0
    invoke_start_time = time.time()
    invoke_gap = 0
    # this is for only inference forward
    if args.eval:
        nn_model.eval()
        evalfw_time_count = 0
        eval_iter_count = 0 
        t = time.time()
        for input_data, label in tqdm(dataloader, leave=False):
            # if iter_count == 0:
            #     input_data1 = copy.deepcopy(input_data)
            # zero the optimizer grad
            try:
                optimizer.zero_grad()
                # 10 class for cifar 10
                one_hot = torch.nn.functional.one_hot(label, num_classes=1000).float().cuda()
                input_data = input_data.cuda()
                torch.cuda.synchronize()
                time_taken = time.time() - t
                dataload_time_count += time_taken
                dataload_event_count += 1

                # to calculate when inference is getting invoked
                t = time.time()
                if eval_iter_count > WARMUP_ITERATION:
                    invoke_gap += t - invoke_start_time
                    invoke_start_time = t
                
                fw = nn_model(input_data)
                torch.cuda.synchronize()
                time_taken = time.time() - t
                evalfw_time_count += time_taken
                forward_time_list.append(time_taken)
                t = time.time()
            except torch.cuda.OutOfMemoryError as e:
                print(e)
                print(
                    "batch size {0} for imagenet 3x224x224 is not suitable for inference with GPU memory {1}GB".format(
                        batch_size, torch.cuda.mem_get_info()[1]>>30
                    )
                )
                evalfw_time_count = math.nan
                eval_iter_count = 1
                break
            except Exception as e:
                print(e)
                print(
                    "batch size {0} for imagenet 3x224x224 is not suitable for inference with GPU memory {1}GB".format(
                        batch_size, torch.cuda.mem_get_info()[1]>>30
                    )
                )
                evalfw_time_count = math.nan
                eval_iter_count = 1
                break
            eval_iter_count += 1
            if eval_iter_count > ITERATION_COUNT_PER_RUN:
                break
            t = time.time()

        return dataload_time_count/dataload_event_count, dataload_time_count/(dataload_event_count * batch_size), evalfw_time_count/eval_iter_count, evalfw_time_count/(eval_iter_count*batch_size), math.nan, math.nan, statistics.stdev(forward_time_list), statistics.median(forward_time_list), math.nan, math.nan, invoke_gap/(eval_iter_count-WARMUP_ITERATION) 
    else:
        # this is for loss calculated forward backward
        nn_model.train()
        iter_count = 0
        fw_time_count = 0
        bw_time_count = 0
        t = time.time()
        for input_data, label in tqdm(dataloader, leave=False):
            # if iter_count == 0:
            #     input_data1 = copy.deepcopy(input_data)
            # zero the optimizer grad
            try:
                optimizer.zero_grad()
                # 10 class for cifar 10
                one_hot = torch.nn.functional.one_hot(label, num_classes=1000).float().cuda()
                input_data = input_data.cuda()
                torch.cuda.synchronize()
                time_taken = time.time() - t
                dataload_time_count += time_taken
                dataload_event_count += 1
            # calculate loss of model
                t = time.time()
                fw = nn_model(input_data)
                torch.cuda.synchronize()
                loss = loss_func(fw, one_hot)
                time_taken = time.time() - t
                fw_time_count += time_taken
                forward_time_list.append(time_taken)
            except torch.cuda.OutOfMemoryError as e:
                print(e)
                print(
                    "batch size {0} for imagenet 3x224x224 is not suitable for forward pass and loss calc with GPU memory {1}GB".format(
                        batch_size, torch.cuda.mem_get_info()[1]>>30
                    )
                )
                fw_time_count = math.nan
                bw_time_count = math.nan
                iter_count = 1
                break
            except Exception as e:
                print(e)
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
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                time_taken = time.time() - t
                bw_time_count += time_taken
                backward_time_list.append(time_taken)
                # per iteration loss record
            except torch.cuda.OutOfMemoryError as e:
                print(e)
                print(
                    "batch size {0} for imagenet 3x224x224 is not suitable for backward pass and update with GPU memory {1}GB".format(
                        batch_size, torch.cuda.mem_get_info()[1]>>30
                    )
                )
                bw_time_count = math.nan
                iter_count = 1
                break
            except Exception as e:
                print(e)
                print(
                    "batch size {0} for imagenet 3x224x224 is not suitable for backward pass and update with GPU memory {1}GB".format(
                        batch_size, torch.cuda.mem_get_info()[1]>>30
                    )
                )
                bw_time_count = math.nan
                iter_count = 1
                break

            iter_count += 1
            if iter_count > ITERATION_COUNT_PER_RUN/2:
                break
            t = time.time()

        # per epoch loss record
        # print("average forward time {0}s with batch size {1}".format(fw_time_count/iter_count, batch_size))
        # print("average loss calc time {0}s with batch size {1}".format(loss_time_count/iter_count, batch_size))
        # print("average backprop time {0}s with batch size {1}".format(bw_time_count/iter_count, batch_size))
        # print("average optimizer step time {0}s with batch size {1}".format(optstep_time_count/iter_count, batch_size))

    del nn_model
    return dataload_time_count/dataload_event_count, dataload_time_count/(dataload_event_count * batch_size), fw_time_count/iter_count, fw_time_count/(iter_count*batch_size), bw_time_count/iter_count, bw_time_count/(iter_count*batch_size), statistics.stdev(forward_time_list[WARMUP_ITERATION:]), statistics.median(forward_time_list[WARMUP_ITERATION:]), statistics.stdev(backward_time_list[WARMUP_ITERATION:]), statistics.median(backward_time_list[WARMUP_ITERATION:]),  math.nan

LEARNING_RATE = 0.001
parser = argparse.ArgumentParser(description='PyTorch Coincidental Training Inference Benchmark')
parser.add_argument("-n", "--network", choices=["ptychonn", "resnet18", "resnet50", "resnet101", "mobilenet_v2"],
                        help="what network will be used")
parser.add_argument('--eval', action='store_true', help="only forward pass run")
parser.add_argument('-bs', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128)')

if __name__ == "__main__":
    args = parser.parse_args()
    network = args.network
    batch_size = args.batch_size

    benchmark_dict = {}

    random.seed(3400)
    # load dataset and data loader for cifar10
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),               # Convert images to PyTorch tensors
        torchvision.transforms.Resize([224,224]),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to mean 0 and standard deviation 1
    ])
    dataset = torchvision.datasets.CIFAR10(root=".", transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)
    # model
    nn_model = torchvision.models.__dict__[network]()
    
    dl, dl_per_sample, fw, fw_per_sample, bw, bw_per_sample, fw_stddev, fw_med, bw_stddev, bw_med, invoke_gap = benchmark(
        nn_model=copy.deepcopy(nn_model), dataloader=dataloader, batch_size=batch_size, eval=args.eval
    )

    benchmark_dict[batch_size] = [fw, fw_per_sample, fw_stddev, fw_med, bw, bw_per_sample, bw_stddev, bw_med, dl, dl_per_sample, invoke_gap]
    
    if not os.path.exists("benchmark_coincidental_{0}_{1}_nn_step.csv".format(network, "infer" if args.eval else "train")):
        with open("benchmark_coincidental_{0}_{1}_nn_step.csv".format(network, "infer" if args.eval else "train"), "w") as fout:
            fout.write("Batch Size\tForward\tForward Per Sample\tForward Stddev\tForward Median\tBackward\tBackward Per Sample\tBackward Stddev\tBackward Median\tData Load\tData Load Per Sample\tInvoke Gap\n")

    with open("benchmark_coincidental_{0}_{1}_nn_step.csv".format(network, "infer" if args.eval else "train"), "a") as fout:
        data_list = benchmark_dict[batch_size]
        fout.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\n".format(
                batch_size, data_list[0], data_list[1], data_list[2], data_list[3], data_list[4], data_list[5], data_list[6], data_list[7], data_list[8], data_list[9], data_list[10]
            )
        )
