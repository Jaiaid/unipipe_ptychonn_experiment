import torch
import copy
import numpy as np
import matplotlib.pyplot as plot

from skimage.metrics import structural_similarity as ssim
from ptychonn import dataset
from ptychonn import model
from ptychonn import parameters

PRETRAINED_MODELPATH = "pretrained_model/pretrained_bestmodel.pth"

E = 1


if __name__=="__main__":
    # to skip the data which are used to train the pretrained model
    diffr_data = dataset.get_diffrdata(skip_line=33)
    gt_data_amp, gt_data_phase = dataset.get_gtdata(skip_line=33)

    # for final plot
    fig, ax = plot.subplots()

    # reshape
    diffr_data = diffr_data.reshape(diffr_data.shape[0]*diffr_data.shape[1], 1, 64, 64)
    gt_data_amp = gt_data_amp.reshape(diffr_data.shape[0]*diffr_data.shape[1], 1, 64, 64)
    gt_data_phase = gt_data_phase.reshape(diffr_data.shape[0]*diffr_data.shape[1], 1, 64, 64)

    # TRAIN_DATASET_SIZE:INFER_DATASET_SIZE = 160:1840 = 2:23
    TRAIN_DATASET_SIZE = 160
    INFER_DATASET_SIZE = 1840

    # if 32 or greater BACKWARD_BS, what to put in forward pass from inference?
    for BACKWARD_BS in [16, 8, 4, 2, 1]:
        FORWARD_BS = 32
        
        model_error_list = []
        model_phase_ssim_list = []

        # for reproducibility
        torch.manual_seed(0)
        np.random.seed(0)
        torch.cuda.manual_seed(0)
        # load the model
        model = torch.load(PRETRAINED_MODELPATH, weights_only=False)
        # create the learner
        iter_per_epoch = np.floor(TRAIN_DATASET_SIZE/BACKWARD_BS) + 1
        step_size = 6 * iter_per_epoch
        criterion = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr = parameters.LR)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=parameters.LR/10, max_lr=parameters.LR,
            step_size_up=step_size, cycle_momentum=False, mode='triangular2')
        # to gpu
        model = model.to("cuda")

        for i in range(0, diffr_data.shape[0], TRAIN_DATASET_SIZE+INFER_DATASET_SIZE):
            train_data = diffr_data[i:min(diffr_data.shape[0], i+TRAIN_DATASET_SIZE)]
            train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_data))
            gt_data_amp_subset = gt_data_amp[i:min(gt_data_amp.shape[0], i+TRAIN_DATASET_SIZE)]
            gt_data_phase_subset = gt_data_phase[i:min(gt_data_phase.shape[0], i+TRAIN_DATASET_SIZE)]
            # print(train_dataset.tensors[0].shape, gt_data_amp_subset.shape, gt_data_phase_subset.shape)

            # training loop for one epoch
            infer_data_offset = 0
            for epoch in range(E):
                for train_data_idx in range(0, train_dataset.tensors[0].shape[0], BACKWARD_BS):
                    train_data = train_dataset.tensors[0][train_data_idx: min(train_data_idx+BACKWARD_BS, train_dataset.tensors[0].shape[0])]

                    # for mixed forward pass inference error calculation
                    infer_data = torch.tensor(diffr_data[i+TRAIN_DATASET_SIZE+infer_data_offset:min(diffr_data.shape[0], i+TRAIN_DATASET_SIZE+infer_data_offset+FORWARD_BS-BACKWARD_BS)]).to("cuda")
                    if infer_data.shape[0] > 0:
                        model.eval()
                        model_output = model.forward(infer_data)
                        model_error = torch.mean((model_output[0].cpu() - torch.tensor(gt_data_amp[i+TRAIN_DATASET_SIZE+infer_data_offset:i+TRAIN_DATASET_SIZE+infer_data_offset+infer_data.shape[0]])) ** 2).item()
                        model_error += torch.mean((model_output[1].cpu() - torch.tensor(gt_data_phase[i+TRAIN_DATASET_SIZE+infer_data_offset:i+TRAIN_DATASET_SIZE+infer_data_offset+infer_data.shape[0]])) ** 2).item()
                        model_error_list.append(model_error)
                        infer_data_offset += infer_data.shape[0]

                    target_amp = gt_data_amp_subset[train_data_idx: min(train_data_idx+BACKWARD_BS, gt_data_amp_subset.shape[0])]
                    target_ph = gt_data_phase_subset[train_data_idx: min(train_data_idx+BACKWARD_BS, gt_data_phase_subset.shape[0])]
                    
                    train_data = torch.tensor(train_data).to("cuda")
                    target_amp = torch.tensor(target_amp).to("cuda")
                    target_ph = torch.tensor(target_ph).to("cuda")

                    # print(target_amp.shape, target_ph.shape, train_data.shape)

                    model.train()
                    model_output = model(train_data)

                    # Compute loss, backpropagation, and optimizer step here
                    loss_amp = criterion(model_output[0], target_amp)  # Define target_amp appropriately
                    loss_ph = criterion(model_output[1], target_ph)  # Define target_ph appropriately
                    optimizer.zero_grad()
                    (loss_amp + loss_ph).backward()
                    optimizer.step()
                    scheduler.step()
            

            infer_dataset = torch.tensor(diffr_data[i+TRAIN_DATASET_SIZE:min(diffr_data.shape[0], i+TRAIN_DATASET_SIZE+INFER_DATASET_SIZE)])
            if infer_dataset.shape[0] == 0:
                print("TRAIN DATASET SIZE:{0}".format(TRAIN_DATASET_SIZE), i)
                continue

            for infer_data_idx in range(infer_data_offset, infer_dataset.shape[0], FORWARD_BS):
                infer_data = infer_dataset[infer_data_idx: min(infer_data_idx+FORWARD_BS, infer_dataset.shape[0])].to("cuda")
                model_output = model.forward(infer_data)
                model_error = torch.mean((model_output[0].cpu() - torch.tensor(gt_data_amp[i+TRAIN_DATASET_SIZE+infer_data_idx:i+TRAIN_DATASET_SIZE+infer_data_idx+infer_data.shape[0]])) ** 2).item()
                model_error += torch.mean((model_output[1].cpu() - torch.tensor(gt_data_phase[i+TRAIN_DATASET_SIZE+infer_data_idx:i+TRAIN_DATASET_SIZE+infer_data_idx+infer_data.shape[0]])) ** 2).item()
                model_error_list.append(model_error)
            # print(len(model_error_list))
            # Compute SSIM
            # print(model_output[1].shape, gt_data_phase[i+TRAIN_DATASET_SIZE:i+TRAIN_DATASET_SIZE+infer_data.shape[0]].shape)
            # model_phase_ssim = ssim(model_output[1].cpu().detach().numpy(), gt_data_phase[i+TRAIN_DATASET_SIZE:i+TRAIN_DATASET_SIZE+infer_data.shape[0]], multichannel=True, data_range=2)
            # model_phase_ssim_list.append(model_phase_ssim)

            # print(f"Pretrained Model Error: {pretrained_model_error}, Oracle Model Error: {oracle_model_error}")
            # model_error_list.append(model_error)
            #     count += 1
            #     if count == 1:
            #         break
            # if count == 1:
            #     break

        # plot the error distribution
        # plot.figure()
        # plot.hist(pretrained_model_error_list, bins=50, alpha=0.5, label="Pretrained Model")
        # plot.hist(oracle_model_error_list, bins=50, alpha=0.5, label="Oracle Model")
        # plot.legend()
        # plot.show()

        print("====================================================================")
        print("TRAIN DATASET SIZE={0}, INFERENCE DATASET SIZE={1}".format(TRAIN_DATASET_SIZE, INFER_DATASET_SIZE))
        print("Model Errors Mean:", sum(model_error_list)/len(model_error_list), len(model_error_list))
        # print("Model SSim Mean:", sum(model_phase_ssim_list)/len(model_phase_ssim_list))

        # TREND_BS=512
        # model_moving_average = []
        # for i in range(0, len(model_error_list), TREND_BS):
        #     model_moving_average.append(sum(model_error_list[i:min(i+TREND_BS, len(model_error_list))]) / min(TREND_BS, len(model_error_list)-i))

        # print("Model Errors Moving Average:", sum(model_moving_average)/len(model_moving_average))
        # print(model_moving_average)
        ax.plot(model_error_list, label="{0}:{1}".format(FORWARD_BS, BACKWARD_BS))

        # calculate moving average of SSIM for smoother curve

        # model_moving_average = []
        # for i in range(0, len(model_phase_ssim_list), TREND_BS):
        #     model_moving_average.append(sum(model_phase_ssim_list[i:min(i+TREND_BS, len(model_phase_ssim_list))]) / min(TREND_BS, len(model_phase_ssim_list)-i))

        # print("Model SSIM Moving Average:", sum(model_moving_average)/len(model_moving_average))

        # fig, ax = plot.subplots()
        # ax.plot(model_moving_average, label="Model")
        # ax.set_xlabel("Data Stream Index (x{0})".format(TREND_BS))
        # ax.set_ylabel("Structural Similarity Index")
        # ax.legend()
        # fig.savefig("fig_oracle_vs_20percentpretrained_ssim_distribution.png")

    ax.set_xlabel("Data Stream Index (x{0})".format(INFER_DATASET_SIZE+TRAIN_DATASET_SIZE))
    ax.set_ylabel("Mean Square Error")
    ax.legend()
    fig.savefig("fig_error_distribution_mixedbatch_size_combination_fixeddatasetsize.png")
    