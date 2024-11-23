import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from skimage.transform import resize
from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader

from . import parameters

def get_dataset(nlines, nvalid_percentage, ntest_percentage):
    diffr_data = np.load(parameters.DATA_DIFFR_PATH)["arr_0"]

    diffr_data_red = np.zeros((diffr_data.shape[0],diffr_data.shape[1],64,64), float)
    for i in range(diffr_data.shape[0]):
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
