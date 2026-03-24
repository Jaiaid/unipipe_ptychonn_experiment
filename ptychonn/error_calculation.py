import os
import numpy as np
import torch

from matplotlib import pyplot as plt
from skimage.transform import resize
from sklearn.metrics import mean_squared_error as mse
from PIL import Image
from typing import Tuple

from . import parameters
from . import dataset
from . import ipc


def calc_error(amps, phs, true_amp, true_ph, point_size, overlap):
    # amps = np.array(amps).squeeze()
    # phs = np.array(phs).squeeze()

    # true_amp = np.array(true_amp).squeeze()
    # true_ph = np.array(true_ph).squeeze()
    
    # find the nearest side length which will make amps a grid of 64x64 images
    # this is to arrange the images in a grid and compare with ground truth
    # print(np.isnan(amps).any(), np.isinf(amps).any(), np.isnan(phs).any(), np.isinf(phs).any())
    # print(np.isnan(true_amp).any(), np.isinf(true_amp).any(), np.isnan(true_ph).any(), np.isinf(true_ph).any())

    test_side_w = amps.shape[1]
    test_side_h = amps.shape[0]
    pattern_h = amps.shape[2]
    pattern_w = amps.shape[3]
    print(amps.shape, true_amp.shape, pattern_h, pattern_w)

    true_amp = true_amp.reshape(test_side_h, test_side_w, pattern_h, pattern_w)
    true_ph = true_ph.reshape(test_side_h, test_side_w, pattern_h, pattern_w)

    composite_amp = np.zeros((test_side_h*point_size+overlap,test_side_w*point_size+overlap),float)
    ctr = np.zeros_like(composite_amp)
    data_reshaped = amps.reshape(test_side_h, test_side_w,pattern_h,pattern_w)[:,:,pattern_h//2-int(overlap/2):pattern_h//2+int(overlap/2),
                                                        pattern_w//2-int(overlap/2):pattern_w//2+int(overlap/2)]

    import time
    total_time = 0
    for i in range(test_side_h):
        for j in range(test_side_w):
            start_time = time.time()
            composite_amp[point_size*i:point_size*i+overlap, point_size*j:point_size*j+overlap] += data_reshaped[i,j]
            ctr[point_size*i:point_size*i+overlap, point_size*j:point_size*j+overlap] +=1
            total_time += time.time() - start_time
    print("Time taken for stitching per diffraction:", total_time/(test_side_h*test_side_w))

    composite_phase = np.zeros((test_side_h*point_size+overlap, test_side_w*point_size+overlap),float)
    ctr = np.zeros_like(composite_phase)
    data_reshaped = phs.reshape(test_side_h,test_side_w,pattern_h,pattern_w)[:,:,pattern_h//2-int(overlap/2):pattern_h//2+int(overlap/2),
                                                        pattern_w//2-int(overlap/2):pattern_w//2+int(overlap/2)]

    total_time = 0
    for i in range(test_side_h):
        for j in range(test_side_w):
            start_time = time.time()
            composite_phase[point_size*i:point_size*i+overlap, point_size*j:point_size*j+overlap] += data_reshaped[i,j]
            ctr[point_size*i:point_size*i+overlap, point_size*j:point_size*j+overlap] +=1
            total_time += time.time() - start_time
    print("Time taken for stitching per diffraction:", total_time/(test_side_h*test_side_w))

    stitched_phase = composite_phase[int(overlap/2):-int(overlap/2), int(overlap/2):-int(overlap/2)]/ctr[int(overlap/2)
                                                                        :-int(overlap/2), int(overlap/2):-int(overlap/2)]

    stitched_amp = composite_amp[int(overlap/2):-int(overlap/2), int(overlap/2):-int(overlap/2)]/ctr[int(overlap/2)
                                                                        :-int(overlap/2), int(overlap/2):-int(overlap/2)]

    # print(stitched_amp.shape, stitched_phase.shape)
    stitched_amp_down = resize(stitched_amp, (test_side_h,test_side_w), preserve_range=True, anti_aliasing=True)
    stitched_phase_down = resize(stitched_phase, (test_side_h,test_side_w), preserve_range=True, anti_aliasing=True)

    # true_amp = Y_I_test.reshape(NLTEST,NLTEST,64,64)
    # true_ph = Y_phi_test.reshape(NLTEST,NLTEST,64,64)
    # print(stitched_amp_down.shape, stitched_phase_down.shape,  true_amp[:,:,pattern_h//2,pattern_w//2].shape,  true_ph[:,:,pattern_h//2,pattern_w//2].shape)

    fig, axs = plt.subplots(2,3, figsize=(20,10))
    im = axs[0,0].imshow(stitched_amp_down)
    plt.colorbar(im, ax=axs[0,0], format="%0.2f")
    axs[0,0].set_title("Stitched Amp")
    im = axs[0,1].imshow(true_amp[:,:,pattern_h//2,pattern_w//2])
    plt.colorbar(im, ax=axs[0,1], format="%0.2f")
    axs[0,1].set_title("True Amp")
    im = axs[1,0].imshow(stitched_phase_down)
    plt.colorbar(im, ax=axs[1,0], format="%0.2f")
    axs[1,0].set_title("Stitched Phase")
    im = axs[1,1].imshow(true_ph[:,:,pattern_h//2,pattern_w//2])
    plt.colorbar(im, ax=axs[1,1], format="%0.2f")
    axs[1,1].set_title("True Phase")
    
    im = axs[0,2].imshow(stitched_amp_down-true_amp[:,:,pattern_h//2,pattern_w//2])
    axs[0,2].set_title("Diff. Amp")
    plt.colorbar(im, ax=axs[0,2], format="%0.2f")
    im = axs[1,2].imshow(stitched_phase_down-true_ph[:,:,pattern_h//2,pattern_w//2])
    plt.colorbar(im, ax=axs[1,2], format="%0.2f")
    axs[1,2].set_title("Diff. Phase")
    fig.savefig("stitched_construction.png", bbox_inches='tight', dpi=300)

    return (
        mse(stitched_amp_down, true_amp[:,:,pattern_h//2,pattern_w//2]), 
        mse(stitched_phase_down, true_ph[:,:,pattern_h//2,pattern_w//2]),
        mse(stitched_amp_down, true_amp[:,:,pattern_h//2,pattern_w//2]),
        mse(stitched_phase_down, true_ph[:,:,pattern_h//2,pattern_w//2])
    )


def postsimulation_error_calc(skip_line=0, large_dataset=False) -> Tuple[float, float]:
    # first search for IPR generated images
    # for them error will be zero
    ipr_genidx_dict = {}
    nn_genidx_dict = {}

    for gtdata_foldername in os.listdir("/dev/shm"):
        if gtdata_foldername.startswith(parameters.SHM_MARKER_FMT_GTGENERATION_FOLDER[:-6]):
            cur_folder = os.path.join("/dev/shm", gtdata_foldername)
            for gendata_fname in os.listdir(cur_folder):
                # there maybe other files with name not in dataidx format, so only consider those with rawgtph suffix
                if not gendata_fname.endswith("rawgtph"):
                    continue

                idx = int(gendata_fname.split(".")[0])
                if idx == 1:
                    print(f"{cur_folder}/{gendata_fname}, {idx} already exist in dict: {idx in ipr_genidx_dict}")
                ipr_genidx_dict[idx] = (
                    ipc.read_shm_data(
                        os.path.join(cur_folder, parameters.SHM_DATA_GEN_AMP_NAMEFMT.format(idx))
                    ), 
                    ipc.read_shm_data(
                        os.path.join(cur_folder, parameters.SHM_DATA_GEN_PHASE_NAMEFMT.format(idx))
                    )
                )

    # first load the ground truth data
    if not large_dataset:
        Y_I, Y_ph = dataset.get_gtdata(skip_line=skip_line)
        Y_I = Y_I.reshape(parameters.DIFFRLINE - skip_line, parameters.IMGCOUNT, parameters.H, parameters.W)
        Y_ph = Y_ph.reshape(parameters.DIFFRLINE - skip_line, parameters.IMGCOUNT, parameters.H, parameters.W)
        width = parameters.H
        height = parameters.W
        point_size = 3
    else:
        Y_I, Y_ph = dataset.get_large_gtdata(skip_line=skip_line)
        Y_I = Y_I.reshape(parameters.DIFFRLINE_L -skip_line, parameters.SCANPOINT_L, parameters.H, parameters.W)
        Y_ph = Y_ph.reshape(parameters.DIFFRLINE_L - skip_line, parameters.SCANPOINT_L, parameters.H, parameters.W)
        width = parameters.H
        height = parameters.W
        point_size = 2
    
    overlap = point_size*4

    gen_amp = np.zeros_like(Y_I)
    gen_ph = np.zeros_like(Y_ph)

    gti_min = np.min(Y_I)
    gtph_min = np.min(Y_ph)
    gti_max = np.max(Y_I)
    gtph_max = np.max(Y_ph)
    print(gti_min, gti_max, gtph_min, gtph_max)

    mse_amp_nn_errorlist = []
    mse_ph_nn_errorlist = []

    for dataidx_i in range(Y_I.shape[0]):
        for dataidx_j in range(Y_I.shape[1]):
            dataidx = dataidx_i*Y_I.shape[1]+dataidx_j
            # first check if generated by IPR
            if dataidx in ipr_genidx_dict:
                gen_amp[dataidx_i, dataidx_j] = ipr_genidx_dict[dataidx][0].reshape(width, height)
                gen_ph[dataidx_i, dataidx_j] = ipr_genidx_dict[dataidx][1].reshape(width, height)
            # else check if ptychonn inferred it
            elif ipc.exist_shm(
                os.path.join(
                    parameters.SHM_MARKER_NNRES_FOLDER,
                    parameters.SHM_MARKER_NNRES_AMP_NAMEFMT.format(dataidx)
                )):

                amp_data = ipc.read_shm_data(os.path.join(
                    parameters.SHM_MARKER_NNRES_FOLDER,
                    parameters.SHM_MARKER_NNRES_AMP_NAMEFMT.format(dataidx)
                ))

                ph_data = ipc.read_shm_data(os.path.join(
                    parameters.SHM_MARKER_NNRES_FOLDER,
                    parameters.SHM_MARKER_NNRES_PHASE_NAMEFMT.format(dataidx)
                ))

                gen_amp[dataidx_i, dataidx_j] = amp_data.reshape(width, height)
                gen_ph[dataidx_i, dataidx_j] = ph_data.reshape(width, height)
                nn_genidx_dict[dataidx] = True

                mse_amp = torch.mean(torch.tensor((Y_I[dataidx_i, dataidx_j][height//2-overlap//2:height//2+overlap//2, width//2-overlap//2:width//2+overlap//2]-amp_data.reshape(parameters.H,parameters.W)[height//2-overlap//2:height//2+overlap//2, width//2-overlap//2:width//2+overlap//2])**2)).item()
                mse_ph = torch.mean(torch.tensor((Y_ph[dataidx_i, dataidx_j][height//2-overlap//2:height//2+overlap//2, width//2-overlap//2:width//2+overlap//2]-ph_data.reshape(parameters.H,parameters.W)[height//2-overlap//2:height//2+overlap//2, width//2-overlap//2:width//2+overlap//2])**2)).item()
                mse_amp_nn_errorlist.append(mse_amp)
                mse_ph_nn_errorlist.append(mse_ph)

    if np.isnan(gen_amp).any() or np.isinf(gen_amp).any() or np.isnan(gen_ph).any() or np.isinf(gen_ph).any():
        print("Gen amp or ph has NaN or Inf values")
        exit()

    for dataidx_i in range(Y_I.shape[0]):
        for dataidx_j in range(Y_I.shape[1]):
            dataidx = dataidx_i*Y_I.shape[1]+dataidx_j

            if dataidx in ipr_genidx_dict or dataidx in nn_genidx_dict:
                pass
            else:
                # print(gen_amp[dataidx_i, dataidx_j].shape, gen_amp[dataidx_i-8:dataidx_i+8, dataidx_j-8:dataidx_j+8,:,:].shape)
                # print(np.mean(gen_amp[dataidx_i-8:dataidx_i+8, dataidx_j-8:dataidx_j+8,:,:], axis=0).shape)
                max_y_coord = min(dataidx_i, gen_amp.shape[0])
                min_y_coord = max(dataidx_i-16, 0)
                max_x_coord = min(dataidx_j, gen_amp.shape[1])
                min_x_coord = max(dataidx_j-16, 0)
                
                ctr = 0
                mean_amp = np.zeros(gen_amp.shape[2:], float) 
                mean_ph = np.zeros(gen_ph.shape[2:], float)
                for y in range(min_y_coord, max_y_coord):
                    for x in range(min_x_coord, max_x_coord):
                        dataidx_neighbor = y*Y_I.shape[1]+x

                        if dataidx_neighbor in ipr_genidx_dict or dataidx_neighbor in nn_genidx_dict:
                            ctr += 1
                            mean_amp += gen_amp[y, x]
                            mean_ph += gen_ph[y, x]

                # mean_ph = np.mean(gen_ph[min_y_coord:max_y_coord, min_x_coord:max_x_coord,:,:], axis=(0,1))
                # mean_amp = np.mean(gen_amp[min_y_coord:max_y_coord, min_x_coord:max_x_coord,:,:], axis=(0,1))
                mean_amp = mean_amp / max(ctr, 1)
                mean_ph = mean_ph / max(ctr, 1)
            
                gen_amp[dataidx_i, dataidx_j] = mean_amp
                gen_ph[dataidx_i, dataidx_j] = mean_ph

    amp_error, ph_error, _, _ = calc_error(gen_amp, gen_ph, Y_I, Y_ph, point_size, overlap)

    print(len(mse_amp_nn_errorlist), len(mse_ph_nn_errorlist))
    print(sum(mse_amp_nn_errorlist)/len(mse_amp_nn_errorlist), sum(mse_ph_nn_errorlist)/len(mse_ph_nn_errorlist))

    fig, ax = plt.subplots(1,1, figsize=(4,2.25))
    yval = np.array(mse_amp_nn_errorlist)[::5]
    ax.scatter(range(yval.shape[0]), yval, s=1, label="MSE Amp Error")
    ax.axhline(y=sum(mse_amp_nn_errorlist)/len(mse_amp_nn_errorlist), color='r', linestyle='--')
    ax.axhline(y=np.percentile(mse_amp_nn_errorlist, 75), color='g', linestyle='--')
    ax.axhline(y=np.median(mse_amp_nn_errorlist), color='orange', linestyle='--')
    ax.set_xlabel("Data Index")
    ax.set_ylabel("MSE Error")
    fig.savefig("nn_generated_data_amp_error.png", bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots(1,1, figsize=(4,2.25))
    yval = np.array(mse_ph_nn_errorlist)[::5]
    ax.scatter(range(yval.shape[0]), yval, s=1, label="MSE Phase Error")
    ax.axhline(y=sum(mse_ph_nn_errorlist)/len(mse_ph_nn_errorlist), color='r', linestyle='--')
    ax.axhline(y=np.percentile(mse_ph_nn_errorlist, 75), color='g', linestyle='--')
    ax.axhline(y=np.median(mse_ph_nn_errorlist), color='orange', linestyle='--')
    ax.set_ylim(0, np.percentile(mse_ph_nn_errorlist, 75)*1.5)
    ax.set_xlabel("Data Index")
    ax.set_ylabel("MSE Error")
    fig.savefig("nn_generated_data_ph_error.png", bbox_inches='tight', dpi=300)

    return amp_error, ph_error, sum(mse_amp_nn_errorlist)/len(mse_amp_nn_errorlist), sum(mse_ph_nn_errorlist)/len(mse_ph_nn_errorlist)

    # # normalize within the range of served requests error
    # nn_error_amp_min = 0 # np.min(mse_nn_amp_errorlist)
    # nn_error_amp_max = np.max(mse_nn_amp_errorlist)
    # nn_error_amp_avg = np.mean(mse_nn_amp_errorlist)
    # nn_error_ph_min = 0 #np.min(mse_nn_ph_errorlist)
    # nn_error_ph_max = np.max(mse_nn_ph_errorlist)
    # nn_error_ph_avg = np.mean(mse_nn_ph_errorlist)

    # print("NN error amp min max", nn_error_amp_min, nn_error_amp_max, (nn_error_amp_avg-nn_error_amp_min)/(nn_error_amp_max-nn_error_amp_min))
    # print("NN error ph min max", nn_error_ph_min, nn_error_ph_max, (nn_error_ph_avg-nn_error_ph_min)/(nn_error_ph_max-nn_error_ph_min))

    # mse_nn_amp_errorlist = []
    # mse_nn_ph_errorlist = []
    # only_nn_idx_count = 0
    # for dataidx in range(Y_I.shape[0]):
    #     # first check if generated by IPR
    #     if ipc.exist_shm(
    #         os.path.join(
    #             parameters.SHM_MARKER_NNRES_FOLDER,
    #             parameters.SHM_MARKER_NNRES_AMP_NAMEFMT.format(dataidx)
    #         )):

    #         mse_amp_errorlist[dataidx] = (mse_amp_errorlist[dataidx] - nn_error_amp_min) / (nn_error_amp_max - nn_error_amp_min)
    #         mse_ph_errorlist[dataidx] = (mse_ph_errorlist[dataidx] - nn_error_ph_min) / (nn_error_ph_max - nn_error_ph_min)
    #         mse_nn_amp_errorlist.append(mse_amp_errorlist[dataidx])
    #         mse_nn_ph_errorlist.append(mse_ph_errorlist[dataidx])
    #         only_nn_idx_count += 1

    # print("Only NN idx count:", only_nn_idx_count)
    print(sum(mse_amp_errorlist)/len(mse_amp_errorlist), sum(mse_ph_errorlist)/len(mse_ph_errorlist), sum(mse_nn_amp_errorlist)/len(mse_nn_amp_errorlist), sum(mse_nn_ph_errorlist)/len(mse_nn_ph_errorlist))
    print(len(mse_amp_errorlist), len(mse_ph_errorlist), len(mse_nn_amp_errorlist), len(mse_nn_ph_errorlist))
    return sum(mse_amp_errorlist)/len(mse_amp_errorlist), sum(mse_ph_errorlist)/len(mse_ph_errorlist), sum(mse_nn_amp_errorlist)/len(mse_nn_amp_errorlist), sum(mse_nn_ph_errorlist)/len(mse_nn_ph_errorlist)
