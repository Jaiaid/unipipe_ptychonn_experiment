import os
import copy
import random
import torch
import torch.nn as nn
import numpy as np
import torchvision
import math
import argparse
import statistics

from tqdm import tqdm
import time

ITERATION_COUNT_PER_RUN = 200
WARMUP_ITERATION = 30

nconv = 32


class recon_model(nn.Module):

    def __init__(self):
        super(recon_model, self).__init__()

        self.encoder = nn.Sequential( # Appears sequential has similar functionality as TF avoiding need for separate model definition and activ
          nn.Conv2d(in_channels=1, out_channels=nconv, kernel_size=3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv, nconv, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.MaxPool2d((2,2)),

          nn.Conv2d(nconv, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),          
          nn.ReLU(),
          nn.MaxPool2d((2,2)),

          nn.Conv2d(nconv*2, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),          
          nn.ReLU(),
          nn.MaxPool2d((2,2)),
          )

        self.decoder1 = nn.Sequential(

          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),

          nn.Conv2d(nconv*4, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),
            
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),

          nn.Conv2d(nconv*2, 1, 3, stride=1, padding=(1,1)),
          nn.Sigmoid() #Amplitude model
          )


        self.decoder2 = nn.Sequential(

          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),

          nn.Conv2d(nconv*4, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),
            
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),

          nn.Conv2d(nconv*2, 1, 3, stride=1, padding=(1,1)),
          nn.Tanh() #Phase model
          )

    def forward(self,x):
        x1 = self.encoder(x)
        amp = self.decoder1(x1)
        ph = self.decoder2(x1)

        #Restore -pi to pi range
        ph = ph*np.pi #Using tanh activation (-1 to 1) for phase so multiply by pi

        return amp,ph

# will do full update through standard pytorch optim module
def benchmark(nn_model: torch.nn.Module, batch_size: int, eval:bool):
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
        for i in tqdm(range(ITERATION_COUNT_PER_RUN+WARMUP_ITERATION), leave=False):
            # zero the optimizer grad
            try:
                optimizer.zero_grad()
                # 10 class for cifar 10
                input_tensor = torch.randn(batch_size, 1, 64, 64, device="cuda")
                torch.cuda.synchronize()
                time_taken = time.time() - t
                dataload_time_count += time_taken
                dataload_event_count += 1

                # to calculate when inference is getting invoked
                t = time.time()
                if eval_iter_count > WARMUP_ITERATION:
                    invoke_gap += t - invoke_start_time
                    invoke_start_time = t
                
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
        for i in tqdm(range(ITERATION_COUNT_PER_RUN+WARMUP_ITERATION), leave=False):
            # if iter_count == 0:
            #     input_data1 = copy.deepcopy(input_data)
            # zero the optimizer grad
            try:
                optimizer.zero_grad()
                # 10 class for cifar 10
                input_tensor = torch.randn(batch_size, 1, 64, 64, device="cuda", requires_grad=True)
                target = torch.randn(batch_size, 1, 64, 64, device="cuda")
                torch.cuda.synchronize()
                time_taken = time.time() - t
                dataload_time_count += time_taken
                dataload_event_count += 1
            # calculate loss of model
                t = time.time()
                torch.cuda.synchronize()
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
parser = argparse.ArgumentParser(description='PyTorch Coincidental Training Inference Benchmark for PtychoNN')
parser.add_argument("-n", "--network", choices=["ptychonn"],
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
    nn_model = recon_model()
    
    dl, dl_per_sample, fw, fw_per_sample, bw, bw_per_sample, fw_stddev, fw_med, bw_stddev, bw_med, invoke_gap = benchmark(
        nn_model=copy.deepcopy(nn_model), batch_size=batch_size, eval=args.eval
    )

    benchmark_dict[batch_size] = [fw, fw_per_sample, fw_stddev, fw_med, bw, bw_per_sample, bw_stddev, bw_med, dl, dl_per_sample, invoke_gap]
