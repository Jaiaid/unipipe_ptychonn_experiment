"""
 We assume training will be done with 0 overhead
 
 Therefore, each interval window will be able to serve
 maximum inference requests possible

 Objectives:
 1. Collect mean inference accuracy, assume missed inference as 0 accuracy
"""

import random
import torch
import numpy as np
import torch.utils
import torch.utils.data

import ptychonn.model
import ptychonn.dataset
import ptychonn.parameters
import ptychonn.process_funcs

# pre train + incremental training in 4 interval
INC_TRAIN_INTERVAL = 5


if __name__ == "__main__":
    # for reproducability
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    torch.backends.cudnn.deterministic = True

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

    test_metrics = []
    performance_metrics = {"train time": [], "inference time": []}

    # init the model
    model = ptychonn.model.recon_model()

    import time
    for interval_count in range(INC_TRAIN_INTERVAL):
        #download and load training data
        trainloader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(
            train_data, list(range(interval_count * len(train_data)//INC_TRAIN_INTERVAL,
                             (interval_count + 1)* len(train_data)//INC_TRAIN_INTERVAL))),
            batch_size=ptychonn.parameters.BATCH_SIZE, shuffle=True, num_workers=4
        )

        #same for test
        #download and load training data
        testloader = torch.utils.data.DataLoader(
            test_data,
            batch_size=ptychonn.parameters.BATCH_SIZE, shuffle=False, num_workers=4)

        # pretrain
        if interval_count == 0:
            validloader = torch.utils.data.DataLoader(
                valid_data,
                batch_size=ptychonn.parameters.BATCH_SIZE, shuffle=True, num_workers=4)
            # train and save
            start_time = time.time()
            train_metrics = ptychonn.process_funcs.train(
                model=model, trainloader=trainloader, chkpt_path="model_best_case/pretrained_bestmodel.pth",
                epoch=ptychonn.parameters.EPOCHS, bs=ptychonn.parameters.BATCH_SIZE, do_validate=True, validloader=validloader)
            performance_metrics["train time"].append(time.time() - start_time)
            # test
            # print(len(train_data), len(valid_data), len(test_data))
            start_time = time.time()
            test_metrics.append(ptychonn.process_funcs.test(model=model, testloader=testloader))
            performance_metrics["inference time"].append(time.time() - start_time)
            continue
        
        # incremental training
        # as best case no training overhead
        # so all inference can be served, we can think that the test will run completely
        start_time = time.time()
        train_metrics = ptychonn.process_funcs.train(
            model=model, trainloader=trainloader, chkpt_path="model_best_case/inctrained_interaval{0}_model.pth".format(interval_count),
            epoch=ptychonn.parameters.EPOCHS, bs=ptychonn.parameters.BATCH_SIZE, do_validate=False)
        performance_metrics["train time"].append(time.time() - start_time)
        # test
        start_time = time.time()
        test_metrics.append(ptychonn.process_funcs.test(model=model, testloader=testloader))
        performance_metrics["inference time"].append(time.time() - start_time)

    # average
    # performance_metrics["train time"] = performance_metrics["train time"]/(INC_TRAIN_INTERVAL * ptychonn.parameters.EPOCHS)
    # performance_metrics["inference time"] = performance_metrics["inference time"]/(INC_TRAIN_INTERVAL)

    print(test_metrics)
    print(performance_metrics)