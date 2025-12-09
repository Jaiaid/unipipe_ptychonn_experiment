import torch
import matplotlib.pyplot as plot

from skimage.metrics import structural_similarity as ssim
from ptychonn import dataset
from ptychonn import model

PRETRAINED_MODELPATH = "pretrained_model/pretrained_bestmodel.pth"
ORACLE_MODELPATH = "pretrained_model/oracle_model.pth"
BS = 1


if __name__=="__main__":
    diffr_data = dataset.get_diffrdata(skip_line=0)
    gt_data_amp, gt_data_phase = dataset.get_gtdata(skip_line=0)
    gt_data_amp = gt_data_amp.reshape(diffr_data.shape[0], diffr_data.shape[1], 64, 64)
    gt_data_phase = gt_data_phase.reshape(diffr_data.shape[0], diffr_data.shape[1], 64, 64)

    pretrained_model = model.recon_model()
    oracle_model = model.recon_model()


    # load the model
    pretrained_model = torch.load(PRETRAINED_MODELPATH, weights_only=False)
    oracle_model.load_state_dict(torch.load(ORACLE_MODELPATH, weights_only=False))
    # to gpu
    pretrained_model = pretrained_model.to("cuda")
    oracle_model = oracle_model.to("cuda")
    pretrained_model.eval()
    oracle_model.eval()

    oracle_model_error_list = []
    pretrained_model_error_list = []

    oracle_model_phase_ssim_list = []
    pretrained_model_phase_ssim_list = []

    # count = 0
    for i in range(diffr_data.shape[0]):
        for j in range(0, diffr_data.shape[1], BS):
            # print(diffr_data[i, j:min(diffr_data.shape[1], j+BS)].reshape(min(diffr_data.shape[1]-j, BS), 1, 64, 64).shape)
            data = diffr_data[i, j:min(diffr_data.shape[1], j+BS)].reshape(min(diffr_data.shape[1]-j, BS), 1, 64, 64)
            data = torch.tensor(data).to("cuda")

            pretrained_model_output = pretrained_model.forward(torch.tensor(data))
            oracle_model_output = oracle_model.forward(torch.tensor(data))

            pretrained_model_error = torch.mean((pretrained_model_output[0].cpu() - torch.tensor(gt_data_amp[i,j:j+data.shape[1]])) ** 2).item()
            oracle_model_error = torch.mean((oracle_model_output[0].cpu() - torch.tensor(gt_data_amp[i,j:j+data.shape[1]])) ** 2).item()

            pretrained_model_error += torch.mean((pretrained_model_output[1].cpu() - torch.tensor(gt_data_phase[i,j:j+data.shape[1]])) ** 2).item()
            oracle_model_error += torch.mean((oracle_model_output[1].cpu() - torch.tensor(gt_data_phase[i,j:j+data.shape[1]])) ** 2).item()

            # Compute SSIM
            pretrained_model_phase_ssim = ssim(pretrained_model_output[1].cpu().detach().numpy()[0][0], gt_data_phase[i,j:j+data.shape[1]][0], multichannel=True, data_range=2)
            oracle_model_phase_ssim = ssim(oracle_model_output[1].cpu().detach().numpy()[0][0], gt_data_phase[i,j:j+data.shape[1]][0], multichannel=True, data_range=2)

            pretrained_model_phase_ssim_list.append(pretrained_model_phase_ssim)
            oracle_model_phase_ssim_list.append(oracle_model_phase_ssim)

            # print(f"Pretrained Model Error: {pretrained_model_error}, Oracle Model Error: {oracle_model_error}")
            pretrained_model_error_list.append(pretrained_model_error)
            oracle_model_error_list.append(oracle_model_error)

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

    print("Pretrained Model Errors Mean:", sum(pretrained_model_error_list)/len(pretrained_model_error_list))
    print("Oracle Model Errors Mean:", sum(oracle_model_error_list)/len(oracle_model_error_list))

    print("Pretrained Model SSim Mean:", sum(pretrained_model_phase_ssim_list)/len(pretrained_model_phase_ssim_list))
    print("Oracle Model SSim Mean:", sum(oracle_model_phase_ssim_list)/len(oracle_model_phase_ssim_list))

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

    # fig, ax = plot.subplots()
    # ax.plot(pretrained_moving_average, label="Pretrained Model")
    # ax.plot(oracle_moving_average, label="Oracle Model")
    # ax.set_xlabel("Data Stream Index (x{0})".format(TREND_BS))
    # ax.set_ylabel("Mean Square Error")
    # ax.legend()
    # fig.savefig("fig_oracle_vs_20percentpretrained_error_distribution.png")


    # calculate moving average of SSIM for smoother curve

    pretrained_moving_average = []
    for i in range(0, len(pretrained_model_phase_ssim_list), TREND_BS):
        pretrained_moving_average.append(sum(pretrained_model_phase_ssim_list[i:min(i+TREND_BS, len(pretrained_model_phase_ssim_list))]) / min(TREND_BS, len(pretrained_model_phase_ssim_list)-i))

    print("Pretrained Model Errors Moving Average:", sum(pretrained_moving_average)/len(pretrained_moving_average))

    count = 0
    oracle_moving_average = []
    for i in range(0, len(oracle_model_phase_ssim_list), TREND_BS):
        oracle_moving_average.append(sum(oracle_model_phase_ssim_list[i:min(i+TREND_BS, len(oracle_model_phase_ssim_list))]) / min(TREND_BS, len(oracle_model_phase_ssim_list)-i))

    print("Oracle Model Errors Moving Average:", sum(oracle_moving_average)/len(oracle_moving_average))

    # fig, ax = plot.subplots()
    # ax.plot(pretrained_moving_average, label="Pretrained Model")
    # ax.plot(oracle_moving_average, label="Oracle Model")
    # ax.set_xlabel("Data Stream Index (x{0})".format(TREND_BS))
    # ax.set_ylabel("Structural Similarity Index")
    # ax.legend()
    # fig.savefig("fig_oracle_vs_20percentpretrained_ssim_distribution.png")

    # print(pretrained_model_error_list, oracle_model_error_list)
    