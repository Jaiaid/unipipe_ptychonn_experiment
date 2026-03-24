import time
import math
import os
import numpy as np
import torch
import copy

from tqdm import tqdm

from . import parameters
from . import error_calculation
from . import ipc
from . import datastream
from . import perf_model
from . import shm_datareader
from . import multicontext_parameters
from . import model

from logfast import fastlogger


"""
This function is for torch multiprocess checkpoint restoration 
the checkpointing process to restore model from checkpoint


"""
def restore_checkpoint(
    result_queue:torch.multiprocessing.Queue,
    chkpt_dir:str
):
    next_model = 0
    # nn_model = model.recon_model()
    while True:
        if ipc.exist_shm(
            os.path.join(chkpt_dir, multicontext_parameters.MULTICONTEXT_SHM_MARKER_IPRITER_END.format(next_model))
        ):
            # nn_model.load_state_dict(
            #     torch.load(
            #         os.path.join(
            #             "/dev/shm/", chkpt_dir, 
            #             multicontext_parameters.MULTICONTEXT_IPRITER_MODELNAME_FMT.format(next_model)
            #         )
            #     )
            # )

            result_queue.put(
                # this is just a signal to restore checkpoint process
                # the actual data will be read from shm by the restore checkpoint process
                (chkpt_dir, next_model)
            )

            next_model += 1

    result_queue.close()
    


def write_inference_results(
    data_queue:torch.multiprocessing.Queue,
    result_queue:torch.multiprocessing.Queue,
    # logger:fastlogger.FastLogger=None
):
    while True:
        if not data_queue.empty():
            infer_delay_missed = 0
            inferidxlist, pred_amps, pred_phs, ipriteration_no = data_queue.get()
            success_list = []

            # sentinel value            
            if inferidxlist is None:
                break
            # inference is done so remove the data from shm
            # as inference will be done only once
            # so delete
            # do the delete first because this is essential claiming we have done within time
            for i, idx in enumerate(inferidxlist):
                try:
                    ipc.remove_shm(
                        parameters.SHM_DATA_DIFFR_NAMEFMT.format(idx)
                    )
                    success_list.append(i)
                except FileNotFoundError as ex:
                    # if logger is not None:
                    #     logger.log("MISSED DUE TO INFER DELAY", idx)
                    infer_delay_missed += 1

            for i in success_list:
                ipc.create_shm_data(
                    os.path.join(
                        parameters.SHM_MARKER_NNRES_FOLDER,
                        parameters.SHM_MARKER_NNRES_PHASE_NAMEFMT.format(inferidxlist[i])
                    ),
                    pred_phs[i]
                )
                ipc.create_shm_data(
                    os.path.join(
                        parameters.SHM_MARKER_NNRES_FOLDER,
                        parameters.SHM_MARKER_NNRES_AMP_NAMEFMT.format(inferidxlist[i])
                    ),
                    pred_amps[i]
                )

            do_forward = False       
            if infer_delay_missed > len(inferidxlist)/2:
                do_forward = True

            result_queue.put((infer_delay_missed, do_forward))

    result_queue.close()
    
    print("Exiting write inference result process")


#Function to update saved model if validation loss is minimum
def update_saved_model(model, path, name, logger=None):
    if not os.path.isdir(path):
        os.mkdir(path)
    if logger is None:
        pass
        # print("save model in ", os.path.join(path, name))
    else:
        logger.log("SAVE CHKPT ", os.path.join(path, name))
    torch.save(model.state_dict(), os.path.join(path, name))

# for checkpoint save offloading
def checkpointing_process_function(data_queue:torch.multiprocessing.Queue):
    # save model
    start_time = time.time()
    # max 150 seconds run
    while time.time() - start_time < 150:
        if not data_queue.empty():
            chkpt_dir, model, next_model = data_queue.get()

            if chkpt_dir is None:
                break

            update_saved_model(
                model=model,
                path=os.path.join(
                    "/dev/shm/", chkpt_dir
                ),
                name=multicontext_parameters.MULTICONTEXT_IPRITER_MODELNAME_FMT.format(next_model)
            )
            ipc.create_shm_marker(
                os.path.join(chkpt_dir, multicontext_parameters.MULTICONTEXT_SHM_MARKER_IPRITER_END.format(next_model)))
            ipc.create_shm_marker(
                os.path.join(chkpt_dir, multicontext_parameters.MULTICONTEXT_SHM_MARKER_IPRITER_END.format(next_model)+str(time.time())))
 
    return
