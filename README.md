# UniPipe: Inference Serving and Fine-Tuning on Single GPU for Stable Inference Latency in ML Surrogate Workflow

In this repository we present our prototype implementation of real-time DNN surrogate processing of a datastream using shared context inference and training and reproducability methods for the results published in IEEE eScience'26. The repository features:

* A prototype DNN surrogate workflow of processing and fine-tuning using single GPU context evaluated against Ptychographic phase image restoration Datasets

* A `C++` scheduling algorithm implementation and `Python` interface using `pybind11` which provides the batch compositions (how many training data and inference data in a DNN pass) which maintains the **Deadline** with best effort and reduces the **makespan** of processing. The batch composition are repeatitive over interval which is equal to the time the classical solver process (or any ground-truth generation process) takes to generate one labeled data.

* A `/dev/shm` based IPC mechanism and two (inference and training) `Python` datareader class implementation to act as data provider for the UniPipe based DNN deployment. These datareader classes assumes that both the data-acquisition process  and the ground-truth generation process puts the streaming and labeled data in `/dev/shm`

We evaluate our system on DNN surrogate based Ptychographic phase image restoration. The ground truth generation is done through classical computation which is slow. To maintain the deadline occasionally processing is handle over to the neural network inference. As the sample view changes over time neural network needs retraining. For this retraining, classical computational method generated result is used as ground truth.


## Environment Setup

We assume that it is inside Linux and ``/dev/shm`` is available for shared memory communication.

1. Clone the **PtychoNN** repository
  ```
  git clone https://github.com/mcherukara/PtychoNN
  cd PtychoNN
  git-lfs install
  git-lfs pull
  ```

2. Relative to **PtychoNN** repo. root, change directory to PyTorch and clone this repository in that folder. Change to the **stream_train_infer_impl** branch
  ```
  cd PyTorch
  git clone https://github.com/Jaiaid/unipipe_ptychonn_experiment
  cd unipipe_ptychonn_experiment
  git checkout stream_train_infer_impl
  mkdir result_logs
  ```

3. Install the requirements.
  ```
  pip install --no-cache-dir -r requirements.txt
  ```

**Note:** Step 1 is needed to download dataset 1 as described in the published paper and reproduce the results described. If the goal is to use **UniPipe** approach to deploy a DNN surrogate workflow, starting from Step 2 will suffice.


## UniPipe Usage

This is still at prototype phase. We request to look into `unipipe_dp.py` script and check into the `unipipe_dp_traininfer` function for the training loop (line 113-322). The forward pass is used to process both training and inference data (line 197-209). Then the training data's results are extracted and loss is calculated and `.backward()` is called (line 256-279).

We are working on to put more detail and a more user-friendly version.

## Reproducing eScience'26 Results

**Note:** Step 3 and 4 can be skipped by downloading the models from this google drive public [link](https://drive.google.com/drive/folders/12jLstRQpE0N8cBv51x8zuuNeQLEiHMOL?usp=sharing) and save them into `pretrained_model` subdirectory w.r.t repo. root.

Followings are the steps to reproduce eScience'26 published results.

1. Download the datasets from this google drive public [link](https://drive.google.com/drive/folders/1RqksLZj0ID2464iQCikKQgOA3_AVQTpi?usp=sharing). Put the contents inside `Ptychonn/data` directory where `Ptychonn` is the repository cloned in Step 1 of [Environment Setup](#environment-setup). **In the paper, we name one dataset as dataset 1 another (in a subdirectory) as dataset 2**.

2. To generate the neural network forward and backward pass performance model curve, (**Assumption:** CUDA device is present and no other CUDA context is running.)
  ```
  pushd ptychonn
  python3 exp_step_benchmarking_ptychonn.py 
  popd
  ```

3. To generate the pre-trained model trained on partial dataset, run
  ```
  bash pretrained_model_generation/pretrained_model_generation.sh
  ```

4. Save the generated models in `pretrained_model` directory. For dataset 1 the model will be saved in `pretrained_model_1.25M_small\best_model.pth`. Save it as `pretrained_model\pretrained_bestmodel_ptychonn_small`. For dataset 2, the model will be saved in `pretrained_model_1.25M_large\best_model.pth`. Save it as `pretrained_model\pretrained_bestmodel_ptychonn_large`.

5. To run all the experiments, run from git repo. root, (**Warning**: It will remove previous ```result_logs``` directory)
  ```
  bash exp_run_all.sh
  ```

  This script will generate logs and other necessary files inside ``result_logs/``. Each compared system's data will be under separate folder. For example **unipipe** data will be in ``result_logs/bulk/unipipe``. If want to run individual experiment, we request to look into the script. This script calls other scripts with self-explanatory names.

6. To generate the plots, run from git repo. root,
  ```
  bash exp_plot_all_escience.sh
  ```
  All generated plots will be in the root folder.


## Future Work
* Validation of UniPipe in more DNN surrogate workflow where datadrift shows up across the datastream when using a pretrained model.

* Investigate and incorporate training dataset exempler selection when fine-tuning for more general and stable training.

* A more user-friendly abstraction and implementation of UniPipe.


## Citation
If you use this work in your research, please consider citing our work:

```
@inproceedings{mobin2026unipipe,
  title={UniPipe: Unified Inference Serving and Fine-Tuning for Deadline-Constrained DNN Surrogate Workflows},
  author={Mobin, Jaiaid and Maurya, Avinash and Rafique, M Mustafa and Nicolae, Bogdan},
  year={2026},
  organization={In Proceedings of the 22nd IEEE International eScience Conference (eScience)}
}
```

## Contact
For query regarding usage, repository, and paper, please contact [jm5071@rit.edu](mailto:jm5071@rit.edu) 