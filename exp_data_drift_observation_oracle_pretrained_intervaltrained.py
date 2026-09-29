import torch
import math
import random
import numpy as np
import matplotlib.pyplot as plot

from skimage.metrics import structural_similarity as ssim
from ptychonn import dataset
from ptychonn import model
from ptychonn import parameters
from plot_parameters import YTICK_LABEL_KW, FIGSIZE, SYSTEM_NAME_LIST, SYSTEM_NAME_TO_LEGEND_DICT, SYSTEM_NAME_TO_HATCH_DICT, CSV_FILENAME_FMT, AXLABEL_KW, YTICK_LABEL_KW, LEGEND_COLSPACING, LEGEND_PROP
from matplotlib.ticker import FormatStrFormatter

PRETRAINED_MODELPATH = "pretrained_model/pretrained_bestmodel_ptychonn.pth"
ORACLE_MODELPATH = "pretrained_model/oracle_model.pth"
DEVICE="cuda"
BS = 1
TRAIN_BS = 1

INTERVAL_COUNT =322#322#110
# so training dataset is size of 1 
TRAIN_FRACTION = 0.0158 #0.0054# 0.0054
EPOCH = 1


if __name__=="__main__":
    # for reproducability
    # https://discuss.pytorch.org/t/training-reproducibility-problem/37143/3
    # https://vandurajan91.medium.com/random-seeds-and-reproducible-results-in-pytorch-211620301eba
    random.seed(parameters.SEED)
    torch.manual_seed(parameters.SEED)
    torch.cuda.manual_seed(parameters.SEED)
    torch.cuda.manual_seed_all(parameters.SEED)
    np.random.seed(parameters.SEED)

    diffr_data = dataset.get_diffrdata(skip_line=33)
    gt_data_amp, gt_data_phase = dataset.get_gtdata(skip_line=33)
    # reshape for easier interval division
    gt_data_amp = gt_data_amp.reshape(diffr_data.shape[0]*diffr_data.shape[1], 64, 64)
    gt_data_phase = gt_data_phase.reshape(diffr_data.shape[0]*diffr_data.shape[1], 64, 64)
    diffr_data = diffr_data.reshape(diffr_data.shape[0]*diffr_data.shape[1], 64, 64)

    # we will take 10% at each interval begining for interval training
    total_data = diffr_data.shape[0]
    interval_length = math.floor(total_data/INTERVAL_COUNT)
    training_dataset_length = math.floor(interval_length*TRAIN_FRACTION)
    # print(training_dataset_length, interval_length)
    # exit()

    pretrained_model = model.recon_model()
    oracle_model = model.recon_model()
    intervaltrained_model = model.recon_model()

    # load the model
    pretrained_model.load_state_dict(torch.load(PRETRAINED_MODELPATH, weights_only=True))
    oracle_model.load_state_dict(torch.load(ORACLE_MODELPATH, weights_only=True))
    intervaltrained_model.load_state_dict(torch.load(PRETRAINED_MODELPATH, weights_only=True))

    # to gpu
    pretrained_model = pretrained_model.to(DEVICE)
    oracle_model = oracle_model.to(DEVICE)
    intervaltrained_model = intervaltrained_model.to(DEVICE)
    pretrained_model.eval()
    oracle_model.eval()
    intervaltrained_model.eval()
    oracle_model_error_list = []
    pretrained_model_error_list = []
    intervaltrained_model_error_list = []

    oracle_model_phase_ssim_list = []
    pretrained_model_phase_ssim_list = []
    intervaltrained_model_phase_ssim_list = []

    for i in range(0, diffr_data.shape[0], BS):
        if i % interval_length == 0: # if we are at the beginning of an interval, do training on the next training_dataset_length data points
            # at training phase
            # not well-written but I dont have time
            intervaltrained_model.train()
            iter_per_epoch = np.floor(training_dataset_length/TRAIN_BS) + 1
            step_size = 6 * iter_per_epoch
            criterion = torch.nn.L1Loss()
            optimizer = torch.optim.Adam(intervaltrained_model.parameters(), lr = parameters.LR)
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=parameters.LR/10, max_lr=parameters.LR,
                step_size_up=step_size, cycle_momentum=False, mode='triangular2')

            for epoch in range(EPOCH):
                for train_idx in range(i, i+training_dataset_length, TRAIN_BS):
                    train_data = diffr_data[train_idx:min(diffr_data.shape[0], train_idx+TRAIN_BS)].reshape(min(diffr_data.shape[0]-train_idx, TRAIN_BS), 1, 64, 64)
                    target_amp = gt_data_amp[train_idx: min(train_idx+TRAIN_BS, gt_data_amp.shape[0])].reshape(min(diffr_data.shape[0]-train_idx, TRAIN_BS), 1, 64, 64)
                    target_ph = gt_data_phase[train_idx: min(train_idx+TRAIN_BS, gt_data_phase.shape[0])].reshape(min(diffr_data.shape[0]-train_idx, TRAIN_BS), 1, 64, 64)
                    
                    train_data = torch.tensor(train_data).to(DEVICE)
                    target_amp = torch.tensor(target_amp).to(DEVICE)
                    target_ph = torch.tensor(target_ph).to(DEVICE)

                    model_output = intervaltrained_model(train_data)

                    # Compute loss, backpropagation, and optimizer step here
                    loss_amp = criterion(model_output[0], target_amp)  # Define target_amp appropriately
                    loss_ph = criterion(model_output[1], target_ph)  # Define target_ph appropriately
                    optimizer.zero_grad()
                    (loss_amp + loss_ph).backward()
                    optimizer.step()
                    scheduler.step()

            print(f"Finished Training Interval {i//interval_length + 1}")

            intervaltrained_model.eval()

            i += training_dataset_length
    
        data = diffr_data[i:min(diffr_data.shape[0], i+BS)].reshape(min(diffr_data.shape[0]-i, BS), 1, 64, 64)
        data = torch.tensor(data).to(DEVICE)
        
        if data.shape[0] == 0:
            continue

        pretrained_model_output = pretrained_model.forward(data)
        oracle_model_output = oracle_model.forward(data)
        intervaltrained_model_output = intervaltrained_model.forward(data)

        pretrained_model_error = torch.mean((pretrained_model_output[0].cpu() - torch.tensor(gt_data_amp[i:i+data.shape[0]])) ** 2).item()
        oracle_model_error = torch.mean((oracle_model_output[0].cpu() - torch.tensor(gt_data_amp[i:i+data.shape[0]])) ** 2).item()
        intervaltrained_model_error = torch.mean((intervaltrained_model_output[0].cpu() - torch.tensor(gt_data_amp[i:i+data.shape[0]])) ** 2).item()
        
        pretrained_model_error += torch.mean((pretrained_model_output[1].cpu() - torch.tensor(gt_data_phase[i:i+data.shape[0]])) ** 2).item()
        oracle_model_error += torch.mean((oracle_model_output[1].cpu() - torch.tensor(gt_data_phase[i:i+data.shape[0]])) ** 2).item()
        intervaltrained_model_error += torch.mean((intervaltrained_model_output[1].cpu() - torch.tensor(gt_data_phase[i:i+data.shape[0]])) ** 2).item()
        
        # if i == 2:
        #     print(data.shape, np.mean(pretrained_model_output[0].cpu().detach().numpy()), np.mean(pretrained_model_output[1].cpu().detach().numpy()))
        #     print(data.shape, np.mean(oracle_model_output[0].cpu().detach().numpy()), np.mean(oracle_model_output[1].cpu().detach().numpy()))
        #     print(data.shape, np.mean(intervaltrained_model_output[0].cpu().detach().numpy()), np.mean(intervaltrained_model_output[1].cpu().detach().numpy()))
            
        #     print(np.mean(data.cpu().detach().numpy()))

        #     print(data.shape, gt_data_phase[i:i+data.shape[0]].shape, pretrained_model_output[1].cpu().shape)
        #     print(f"Data idx {i}: Pretrained Model Amp. Error: {torch.mean((pretrained_model_output[0].cpu() - torch.tensor(gt_data_amp[i:i+data.shape[0]])) ** 2).item()}, Pretrained Model Ph. Error: {torch.mean((pretrained_model_output[1].cpu() - torch.tensor(gt_data_phase[i:i+data.shape[0]])) ** 2).item()}")
        
        #     print(np.mean(gt_data_phase[i:i+data.shape[0]]), np.mean(pretrained_model_output[1].cpu().detach().numpy()))

        # print(pretrained_model_error, oracle_model_error, intervaltrained_model_error)

        # Compute SSIM
        pretrained_model_phase_ssim = ssim(pretrained_model_output[1].cpu().detach().numpy()[0].reshape(64, 64), gt_data_phase[i:i+data.shape[0]].reshape(64,64), multichannel=False, data_range=2)
        oracle_model_phase_ssim = ssim(oracle_model_output[1].cpu().detach().numpy()[0].reshape(64, 64), gt_data_phase[i:i+data.shape[0]].reshape(64, 64), multichannel=False, data_range=2)
        intervaltrained_model_phase_ssim = ssim(intervaltrained_model_output[1].cpu().detach().numpy()[0].reshape(64, 64), gt_data_phase[i:i+data.shape[0]].reshape(64, 64), multichannel=False, data_range=2)
        
        pretrained_model_phase_ssim_list.append(pretrained_model_phase_ssim)
        oracle_model_phase_ssim_list.append(oracle_model_phase_ssim)
        intervaltrained_model_phase_ssim_list.append(intervaltrained_model_phase_ssim)

        # print(f"Pretrained Model Error: {pretrained_model_error}, Oracle Model Error: {oracle_model_error}")
        pretrained_model_error_list.append(pretrained_model_error)
        oracle_model_error_list.append(oracle_model_error)
        intervaltrained_model_error_list.append(intervaltrained_model_error)

    # plot the error distribution
    # plot.figure()
    # plot.hist(pretrained_model_error_list, bins=50, alpha=0.5, label="Pretrained Model")
    # plot.hist(oracle_model_error_list, bins=50, alpha=0.5, label="Oracle Model")
    # plot.legend()
    # plot.show()

    print("Pretrained Model Errors Len:", len(pretrained_model_error_list))
    print("Oracle Model Errors Len:", len(oracle_model_error_list))
    print("IntervalTrained Model Errors Len:", len(intervaltrained_model_error_list))
    print("Pretrained Model Errors Mean:", sum(pretrained_model_error_list)/len(pretrained_model_error_list))
    print("Oracle Model Errors Mean:", sum(oracle_model_error_list)/len(oracle_model_error_list))
    print("IntervalTrained Model Errors Mean:", sum(intervaltrained_model_error_list)/len(intervaltrained_model_error_list))

    print("Pretrained Model SSim Mean:", sum(pretrained_model_phase_ssim_list)/len(pretrained_model_phase_ssim_list))
    print("Oracle Model SSim Mean:", sum(oracle_model_phase_ssim_list)/len(oracle_model_phase_ssim_list))
    print("IntervalTrained Model SSim Mean:", sum(intervaltrained_model_phase_ssim_list)/len(intervaltrained_model_phase_ssim_list))

    TREND_BS=512
    pretrained_moving_average = []
    for i in range(0, len(pretrained_model_error_list), TREND_BS):
        pretrained_moving_average.append(sum(pretrained_model_error_list[i:min(i+TREND_BS, len(pretrained_model_error_list))]) / min(TREND_BS, len(pretrained_model_error_list)-i))

    print("Pretrained Model Errors Moving Average:", sum(pretrained_moving_average)/len(pretrained_moving_average))

    count = 0
    oracle_moving_average = []
    for i in range(0, len(oracle_model_error_list), TREND_BS):
        oracle_moving_average.append(sum(oracle_model_error_list[i:min(i+TREND_BS, len(oracle_model_error_list))]) / min(TREND_BS, len(oracle_model_error_list)-i))

    print("Oracle Model Errors Moving Average:", sum(oracle_moving_average)/len(oracle_moving_average))

    count = 0
    intervaltrained_model_moving_average = []
    for i in range(0, len(oracle_model_error_list), TREND_BS):
        intervaltrained_model_moving_average.append(sum(intervaltrained_model_error_list[i:min(i+TREND_BS, len(intervaltrained_model_error_list))]) / min(TREND_BS, len(intervaltrained_model_error_list)-i))

    print("IntervalTrained Model Errors Moving Average:", sum(intervaltrained_model_error_list)/len(intervaltrained_model_error_list))


    fig, ax = plot.subplots(figsize=FIGSIZE)
    ax.plot(pretrained_moving_average, label="Pretrained Model")
    ax.plot(oracle_moving_average, label="Oracle Model")
    ax.plot(intervaltrained_model_moving_average, label="Continual Trained Model")
    ax.set_xlabel("Data Stream Index (x{0})".format(TREND_BS), **AXLABEL_KW)
    ax.set_ylabel("Mean Square Error", **AXLABEL_KW)
    ax.set_yticks(np.arange(0.2, 1.4, 0.2))
    ax.set_ylim([0.3, 1.2])
    ax.set_yticklabels(np.arange(0.2, 1.4, 0.2), **YTICK_LABEL_KW)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    ax.set_xticks(np.arange(0, 45, 10))
    ax.set_xticklabels(np.arange(0, 45, 10), **YTICK_LABEL_KW)
    ax.legend(frameon=False, prop=LEGEND_PROP)
    fig.savefig("fig_oracle_20percentpretrained_intervaltrained_error_distribution.png", dpi=600, bbox_inches="tight")
    fig.savefig("fig_oracle_20percentpretrained_intervaltrained_error_distribution.pdf", format="pdf", dpi=600, bbox_inches="tight")


    # # calculate moving average of SSIM for smoother curve

    # pretrained_moving_average = []
    # for i in range(0, len(pretrained_model_phase_ssim_list), TREND_BS):
    #     pretrained_moving_average.append(sum(pretrained_model_phase_ssim_list[i:min(i+TREND_BS, len(pretrained_model_phase_ssim_list))]) / min(TREND_BS, len(pretrained_model_phase_ssim_list)-i))

    # print("Pretrained Model Errors Moving Average:", sum(pretrained_moving_average)/len(pretrained_moving_average))

    # count = 0
    # oracle_moving_average = []
    # for i in range(0, len(oracle_model_phase_ssim_list), TREND_BS):
    #     oracle_moving_average.append(sum(oracle_model_phase_ssim_list[i:min(i+TREND_BS, len(oracle_model_phase_ssim_list))]) / min(TREND_BS, len(oracle_model_phase_ssim_list)-i))

    # print("Oracle Model Errors Moving Average:", sum(oracle_moving_average)/len(oracle_moving_average))

    # count = 0
    # intervaltrained_model_moving_average = []
    # for i in range(0, len(oracle_model_phase_ssim_list), TREND_BS):
    #     intervaltrained_model_moving_average.append(sum(intervaltrained_model_phase_ssim_list[i:min(i+TREND_BS, len(intervaltrained_model_phase_ssim_list))]) / min(TREND_BS, len(intervaltrained_model_phase_ssim_list)-i))

    # print("IntervalTrained Model Errors Moving Average:", sum(intervaltrained_model_moving_average)/len(intervaltrained_model_moving_average))


    fig, ax = plot.subplots(figsize=FIGSIZE)
    ax.plot(pretrained_moving_average, label="Pretrained Model")
    ax.plot(oracle_moving_average, label="Oracle Model")
    ax.plot(intervaltrained_model_moving_average, label="Continual Trained Model")
    ax.set_xlabel("Data Stream Index (x{0})".format(TREND_BS), **AXLABEL_KW)
    ax.set_ylabel("Structural Similarity Index", **AXLABEL_KW)
    ax.legend(frameon=False, prop=LEGEND_PROP)
    fig.savefig("fig_oracle_20percentpretrained_intervaltrained_phase_ssim_distribution.png", dpi=600)
    fig.savefig("fig_oracle_20percentpretrained_intervaltrained_phase_ssim_distribution.pdf", format="pdf", dpi=600, bbox_inches="tight")

    # print(pretrained_model_error_list, oracle_model_error_list)
    