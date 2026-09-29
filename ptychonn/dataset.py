import os
import h5py
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from skimage.transform import resize
from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader

from . import parameters
from . import perf_model


def get_diffrdata(skip_line=0) -> np.ndarray:
    diffr_data = np.load(parameters.DATA_DIFFR_PATH)["arr_0"]

    diffr_data_red = np.zeros((diffr_data.shape[0]-skip_line,diffr_data.shape[1],64,64), np.float32)
    for i in range(skip_line, diffr_data.shape[0]):
        for j in range(diffr_data.shape[1]):
            diffr_data_red[i-skip_line,j] = resize(diffr_data[i, j,32:-32,32:-32],(64,64),preserve_range=True, anti_aliasing=True)
            diffr_data_red[i-skip_line,j] = np.where(diffr_data_red[i-skip_line,j]<3,0,diffr_data_red[i-skip_line,j])

    return diffr_data_red


def get_diffrdata_large(skip_line=0) -> np.ndarray:
    diffr_data = h5py.File(os.path.join(parameters.LARGE_DATASET_DIR, parameters.LARGE_DATASET_FILE))["data"]["reciprocal"][:]
    diffr_data = diffr_data.reshape(parameters.DIFFRLINE_L, parameters.SCANPOINT_L, parameters.H_L, parameters.W_L)

    diffr_data_red = np.zeros((parameters.DIFFRLINE_L-skip_line, parameters.SCANPOINT_L,64,64), np.float32)
    for i in range(skip_line, diffr_data.shape[0]):
        for j in range(diffr_data.shape[1]):
            diffr_data_red[i-skip_line,j] = diffr_data[i,j,32:-32,32:-32] # resize(diffr_data[i,j],(64,64),preserve_range=True, anti_aliasing=True)
            diffr_data_red[i-skip_line,j] = np.where(diffr_data_red[i-skip_line,j]<3,0,diffr_data_red[i-skip_line,j])

    return diffr_data_red


def get_dataset(datarate, deadline, overrideratio=None, IPR_throughput=None):
    if  IPR_throughput is None:
        IPR_throughput = parameters.IPR_THROUGHPUT
    
    if overrideratio is None:
        ratio = perf_model.estimate_train_test_ratio(
            IPR_throughput=IPR_throughput,
            datarate=datarate, deadline=deadline)
    else:
        ratio = overrideratio

    ntest_percentage = (1-ratio)*100
    nvalid_percentage = max(1, (100-ntest_percentage)*0.3)

    diffr_data = np.load(parameters.DATA_DIFFR_PATH)["arr_0"]

    diffr_data_red = np.zeros((diffr_data.shape[0],diffr_data.shape[1],64,64), float)
    for i in range(1, diffr_data.shape[0]):
        for j in range(diffr_data.shape[1]):
            diffr_data_red[i,j] = resize(diffr_data[i,j,32:-32,32:-32],(64,64),preserve_range=True, anti_aliasing=True)
            diffr_data_red[i,j] = np.where(diffr_data_red[i,j]<3,0,diffr_data_red[i,j])

    ground_truth_data = np.load(parameters.REAL_SPACE_PATH)
    ground_truth_amp = np.abs(ground_truth_data)
    ground_truth_ph = np.angle(ground_truth_data)

    total_line_count = parameters.DIFFRLINE
    test_start_line = int(total_line_count * ( 100 - ntest_percentage)/100)
    valid_start_line = int(total_line_count * ( 100 - (nvalid_percentage + ntest_percentage ))/100)
    print(diffr_data_red[:valid_start_line,:].shape)
    for i in range(parameters.IMGCOUNT):
        if i == 0:
            X_train = diffr_data_red[i, 0:valid_start_line].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]
            Y_I_train = ground_truth_amp[i, 0:valid_start_line].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]
            Y_phi_train = ground_truth_ph[i, 0:valid_start_line].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]

            X_valid = diffr_data_red[i, valid_start_line:test_start_line].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]
            Y_I_valid = ground_truth_amp[i, valid_start_line:test_start_line].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]
            Y_phi_valid = ground_truth_ph[i, valid_start_line:test_start_line].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]
            
            X_test = diffr_data_red[i, test_start_line:].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]
            Y_I_test = ground_truth_amp[i, test_start_line:].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]
            Y_phi_test = ground_truth_ph[i, test_start_line:].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]
        else:
            X_train = np.vstack((X_train, diffr_data_red[i, 0:valid_start_line].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]))
            Y_I_train = np.vstack((Y_I_train, ground_truth_amp[i, 0:valid_start_line].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]))
            Y_phi_train = np.vstack((Y_phi_train, ground_truth_ph[i, 0:valid_start_line].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]))

            X_valid = np.vstack((X_valid, diffr_data_red[i, valid_start_line:test_start_line].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]))
            Y_I_valid = np.vstack((Y_I_valid, ground_truth_amp[i, valid_start_line:test_start_line].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]))
            Y_phi_valid = np.vstack((Y_phi_valid, ground_truth_ph[i, valid_start_line:test_start_line].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]))
            
            X_test = np.vstack((X_test, diffr_data_red[i, test_start_line:].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]))
            Y_I_test = np.vstack((Y_I_test, ground_truth_amp[i, test_start_line:].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]))
            Y_phi_test = np.vstack((Y_phi_test, ground_truth_ph[i, test_start_line:].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]))

    print(X_train.shape, X_valid.shape, X_test.shape, test_start_line)

    return {"train": (X_train, Y_I_train, Y_phi_train),
            "valid": (X_valid, Y_I_valid, Y_phi_valid),
            "test": (X_test, Y_I_test, Y_phi_test)}


def get_gtdata(skip_line=0) -> np.ndarray:
    ground_truth_data = np.load(parameters.REAL_SPACE_PATH)
    ground_truth_amp = np.abs(ground_truth_data)
    ground_truth_ph = np.angle(ground_truth_data)

    # we will generate the data array by reading each line from ground truth amp. and phase data
    for i in range(skip_line, parameters.DIFFRLINE):
        if i == skip_line:
            Y_I = ground_truth_amp[i, :].reshape(-1,parameters.H,parameters.W)
            Y_phi = ground_truth_ph[i, :].reshape(-1,parameters.H,parameters.W)
        else:
            Y_I = np.vstack((Y_I, ground_truth_amp[i, :].reshape(-1,parameters.H,parameters.W)))
            Y_phi = np.vstack((Y_phi, ground_truth_ph[i, :].reshape(-1,parameters.H,parameters.W)))

    return Y_I, Y_phi


def get_large_gtdata(skip_line=0) -> np.ndarray:
    ground_truth_data = h5py.File(
        os.path.join(parameters.LARGE_DATASET_DIR, parameters.LARGE_DATASET_FILE)
    )["data"]["real"]
    ground_truth_amp = np.abs(ground_truth_data)
    ground_truth_ph = np.angle(ground_truth_data)
    
    Y_I = np.zeros((ground_truth_data.shape[0]-skip_line*parameters.SCANPOINT_L, 64, 64), dtype=np.float32)
    Y_phi = np.zeros((ground_truth_data.shape[0]-skip_line*parameters.SCANPOINT_L, 64, 64), dtype=np.float32)
    
    # we will generate the data array by reading each line from ground truth amp. and phase data
    for i in range(skip_line*parameters.SCANPOINT_L, ground_truth_data.shape[0]):
        # for NN feeding
        gt_amp = resize(ground_truth_amp[i],(64,64), preserve_range=True, anti_aliasing=True)
        gt_ph = resize(ground_truth_ph[i],(64,64), preserve_range=True, anti_aliasing=True)

        Y_I[i-skip_line*parameters.SCANPOINT_L] = gt_amp
        Y_phi[i-skip_line*parameters.SCANPOINT_L] = gt_ph

    return Y_I, Y_phi


def get_large_dataset(datarate, deadline, overrideratio=None, IPR_throughput=None):
    if  IPR_throughput is None:
        IPR_throughput = parameters.IPR_THROUGHPUT
    
    if overrideratio is None:
        ratio = perf_model.estimate_train_test_ratio(
            IPR_throughput=IPR_throughput,
            datarate=datarate, deadline=deadline)
    else:
        ratio = overrideratio

    ntest_percentage = 0
    nvalid_percentage = 0

    diffr_data = h5py.File(os.path.join(parameters.LARGE_DATASET_DIR, parameters.LARGE_DATASET_FILE))["data"]["reciprocal"][:]
    diffr_data = diffr_data.reshape(parameters.DIFFRLINE_L, parameters.SCANPOINT_L, parameters.H_L, parameters.W_L)

    diffr_data_red = np.zeros((diffr_data.shape[0],diffr_data.shape[1],64,64), float)
    for i in range(1, diffr_data.shape[0]):
        for j in range(diffr_data.shape[1]):
            diffr_data_red[i,j] = resize(diffr_data[i,j],(64,64),preserve_range=True, anti_aliasing=True)
            diffr_data_red[i,j] = np.where(diffr_data_red[i,j]<3,0,diffr_data_red[i,j])

    ground_truth_data = h5py.File(os.path.join(parameters.LARGE_DATASET_DIR, parameters.LARGE_DATASET_FILE))["data"]["real"]
    ground_truth_amp = np.abs(ground_truth_data)
    ground_truth_ph = np.angle(ground_truth_data)


    for i in range(parameters.IMGCOUNT):
        if i == 0:
            X_train = diffr_data_red[i, 0:valid_start_line].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]
            Y_I_train = ground_truth_amp[i, 0:valid_start_line].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]
            Y_phi_train = ground_truth_ph[i, 0:valid_start_line].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]

            X_valid = diffr_data_red[i, valid_start_line:test_start_line].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]
            Y_I_valid = ground_truth_amp[i, valid_start_line:test_start_line].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]
            Y_phi_valid = ground_truth_ph[i, valid_start_line:test_start_line].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]
            
            X_test = diffr_data_red[i, test_start_line:].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]
            Y_I_test = ground_truth_amp[i, test_start_line:].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]
            Y_phi_test = ground_truth_ph[i, test_start_line:].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]
        else:
            X_train = np.vstack((X_train, diffr_data_red[i, 0:valid_start_line].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]))
            Y_I_train = np.vstack((Y_I_train, ground_truth_amp[i, 0:valid_start_line].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]))
            Y_phi_train = np.vstack((Y_phi_train, ground_truth_ph[i, 0:valid_start_line].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]))

            X_valid = np.vstack((X_valid, diffr_data_red[i, valid_start_line:test_start_line].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]))
            Y_I_valid = np.vstack((Y_I_valid, ground_truth_amp[i, valid_start_line:test_start_line].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]))
            Y_phi_valid = np.vstack((Y_phi_valid, ground_truth_ph[i, valid_start_line:test_start_line].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]))
            
            X_test = np.vstack((X_test, diffr_data_red[i, test_start_line:].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]))
            Y_I_test = np.vstack((Y_I_test, ground_truth_amp[i, test_start_line:].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]))
            Y_phi_test = np.vstack((Y_phi_test, ground_truth_ph[i, test_start_line:].reshape(-1,parameters.H,parameters.W)[:,np.newaxis,:,:]))

    print(X_train.shape, X_valid.shape, X_test.shape, test_start_line)

    return {"train": (X_train, Y_I_train, Y_phi_train),
            "valid": (X_valid, Y_I_valid, Y_phi_valid),
            "test": (X_test, Y_I_test, Y_phi_test)}
