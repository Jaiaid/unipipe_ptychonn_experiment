"""
 We assume training will be done with full overhead
 
 Therefore, each interval window will have least amount of time to serve inferences

 Objectives:
 1. Collect mean inference accuracy, assume missed inference as 0 accuracy
"""

import argparse
import random
import copy
import torch
import numpy as np
import torch.utils
import torch.utils.data

import ptychonn.dataset
import ptychonn.parameters

from scipy.stats import ks_2samp

if __name__ == "__main__":
    # for reproducability
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    torch.backends.cudnn.deterministic = True

    # define arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--interval-duration", "-idur", type=int, required=True, help="length of interval in seconds")
    arg_parser.add_argument("--interval-count", "-icount", type=int, required=True, help="number of interval")
    # get the arguments
    args = arg_parser.parse_args()

    dataset_dict = ptychonn.dataset.get_dataset(nlines=161, nvalid_percentage=20, ntest_percentage=10)

    train_data = dataset_dict["train"]
    valid_data = dataset_dict["valid"]
    test_data = dataset_dict["test"]

    #Training data
    X_train_tensor = torch.Tensor(train_data[0]) 
    Y_I_train_tensor = torch.Tensor(train_data[1]) 
    Y_phi_train_tensor = torch.Tensor(train_data[2])

    #Validation data
    X_valid_tensor = torch.Tensor(valid_data[0]) 
    Y_I_valid_tensor = torch.Tensor(valid_data[1]) 
    Y_phi_valid_tensor = torch.Tensor(valid_data[2])

    #Test data
    X_test_tensor = torch.Tensor(test_data[0]) 
    Y_I_test_tensor = torch.Tensor(test_data[1]) 
    Y_phi_test_tensor = torch.Tensor(test_data[2])

    # print(X_train_tensor.shape, Y_I_train_tensor.shape, Y_phi_train_tensor.shape)

    train_data = torch.utils.data.TensorDataset(X_train_tensor,Y_I_train_tensor,Y_phi_train_tensor)
    valid_data = torch.utils.data.TensorDataset(X_valid_tensor,Y_I_valid_tensor,Y_phi_valid_tensor)
    test_data = torch.utils.data.TensorDataset(X_test_tensor, Y_I_test_tensor, Y_phi_test_tensor)

    # to create some datastructures beforehand
    # this is done to avoid some overhead when interval and make the scenario more realistic
    # in secnario we will just calculate the prediction and fill up a initiated array
    # error calculation will be done after all intervals are finished and we have result
    testloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1, shuffle=True)
    testloader_iter = iter(testloader)

    test_rep_data = []
    for interval_count in range(args.interval_count):
        result_list = [[], [], []]
        #same for test
        #download and load training data
        for j in range(100 * (args.interval_duration + 1)):
            try:
                batch = next(testloader_iter)
            except StopIteration:
                testloader = torch.utils.data.DataLoader(
                    test_data,
                    batch_size=1, shuffle=True)

            result_list[0].append(copy.deepcopy(batch[0].numpy()[0]))
            result_list[1].append(copy.deepcopy(batch[1].numpy()[0]))
            result_list[2].append(copy.deepcopy(batch[2].numpy()[0]))

        np_diff_array = np.array(result_list[0]).mean(axis=(0, 1)).flatten()
        np_gtamp_array = np.array(result_list[1]).mean(axis=(0, 1)).flatten()
        np_gtph_array = np.array(result_list[2]).mean(axis=(0, 1)).flatten()

        test_rep_data.append([np_diff_array, np_gtamp_array, np_gtph_array])

        # print("Test Interval {0}".format(interval_count))
        # print("diffraction data: ", np_diff_array.mean(), np_diff_array.std())
        # print("ground truth amp. data: ", np_gtamp_array.mean(), np_gtamp_array.std())
        # print("ground truth phase data: ", np_gtph_array.mean(), np_gtph_array.std())
        # stat_in = ks_2samp(np_diff_array, np_diff_array)
        # stat_out1 = ks_2samp(np_gtamp_array, np_gtamp_array)
        # stat_out2 = ks_2samp(np_gtamp_array, np_gtamp_array)
        # print("KS similarity with pretrain dataset input:", stat_in.pvalue, stat_out1.pvalue, stat_out2.pvalue)
        # print("KS similarity with pretrain dataset amp. output:", ks_2samp(np_gtamp_array, np_gtamp_array))

    train_rep_data = []
    for interval_count in range(args.interval_count):
        #download and load training data
        trainloader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(
                train_data,
                list(range(interval_count * len(train_data)//args.interval_count,
                           (interval_count + 1)* len(train_data)//args.interval_count)
                )
            ),
            batch_size=1, shuffle=False
        )

        result_list = [[], [], []]
        for batch in trainloader:
            result_list[0].append(copy.deepcopy(batch[0].numpy()[0]))
            result_list[1].append(copy.deepcopy(batch[1].numpy()[0]))
            result_list[2].append(copy.deepcopy(batch[2].numpy()[0]))

        np_diff_array = np.array(result_list[0]).mean(axis=(0, 1)).flatten()
        np_gtamp_array = np.array(result_list[1]).mean(axis=(0, 1)).flatten()
        np_gtph_array = np.array(result_list[2]).mean(axis=(0, 1)).flatten()

        train_rep_data.append([np_diff_array, np_gtamp_array, np_gtph_array])

    # compare each training part with different test part
    # input data distribution, output a matrix with row of each train part and column of each test part
    for i in range(args.interval_count):
        for j in range(args.interval_count):
            stat_in = ks_2samp(train_rep_data[i][0], test_rep_data[j][0])
            stat_out1 = ks_2samp(train_rep_data[i][1], test_rep_data[j][1])
            stat_out2 = ks_2samp(train_rep_data[i][2], test_rep_data[j][2])
            print("{0}".format(stat_in.pvalue), end="\t")
        print("")

    # input data distribution, output a matrix with row of each train part and column of each train part
    for i in range(args.interval_count):
        for j in range(args.interval_count):
            stat_in = ks_2samp(train_rep_data[i][0], train_rep_data[j][0])
            stat_out1 = ks_2samp(train_rep_data[i][1], train_rep_data[j][1])
            stat_out2 = ks_2samp(train_rep_data[i][2], train_rep_data[j][2])
            print("{0}".format(stat_in.pvalue), end="\t")
        print("")

    # input data distribution, output a matrix with row of each test part and column of each test part
    for i in range(args.interval_count):
        for j in range(args.interval_count):
            stat_in = ks_2samp(test_rep_data[i][0], test_rep_data[j][0])
            stat_out1 = ks_2samp(test_rep_data[i][1], test_rep_data[j][1])
            stat_out2 = ks_2samp(test_rep_data[i][2], test_rep_data[j][2])
            print("{0}".format(stat_in.pvalue), end="\t")
        print("")