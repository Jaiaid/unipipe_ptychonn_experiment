## Concurrent Training and Ground Truth Generation Performance Model

In this repository we present our system on shared context inference and training on a datastream . Different parameters of the system are evaluated based on performance model in a concurrent ground truth generation setup with data sample specific but constant length timelimit/deadline for processing. Our system has following features:

* Overlapping NN inference, trainnig and ground truth generation 
* Maximization of Ground Truth Generation for error minimization over the whole datastream
* Training epoch count estimation for intermittent training
* Batch size estimation to finish the NN inference and training within the intermittent NN processing timeline

We evaluate our system on Ptychographic phase image restoration. The ground truth generation is done through classical computation which is slow. To maintain the deadline occasionally processing is handle over to the neural network inference. As the sample view changes over time neural network needs retraining. For this retraining, classical computational method generated result is used as ground truth.


### Envirnoment Setup

We assume that it is inside Linux and ``/dev/shm`` is available for shared memory communication.

1. Clone the **PtychoNN** repository
```
git clone https://github.com/mcherukara/PtychoNN
```

2. Relative to **PtychoNN** repo. root, change directory to PyTorch and clone this repository in that folder. Change to the **stream_train_infer_impl** branch
```
cd PyTorch
git clone https://github.com/Jaiaid/unipipe_ptychonn_experiment
cd unipipe_ptychonn_experiment
git checkout -b stream_train_infer_impl
mkdir result_logs
```

3. Install the requirements.
```
pip install --no-cache-dir -r requirements.txt
```

### Experiment Run

Our two major performance metric are evaluated through following script
```
bash exp_perf_datacollection_bulk.sh
```
This script will generate logs and other necessary files inside ``result_logs/bulk``. Each compared system's data will be under separate folder. For example **unipipe** data will be in ``result_logs/bulk/unipipe``

To generate the plots,
```
bash exp_plot_generation.sh <result directory containing the system folders>
```
Example,
```
bash exp_plot_generation.sh result_logs/bulk
```
All generated plots will be in the result folder (e.g. ``result_logs/bulk``)


### Background Mock Phase Retrieval Process

To simulate the background load of phase retrival computation CPU load, we use [ptypy](https://ptycho.github.io/ptypy/) provided benchmarking code for Phase retrieval. Follow the [instruction](https://ptycho.github.io/ptypy/rst/getting_started.html#installation) to setup the repo. and necessary environment. The repo. is assumed to be setup in same ``PtychoNN/PyTorch`` folder.

Modify the script ``benchmark/mpi_allreduce_bench.sh`` to measure phase retrieval bandwidth in your system
```
#!/bin/bash

# Runs MPI all reduce benchmark with variable number of processes

echo "Processes,i08,i13,i14_1,i14_2"
for p in {1..8}
do
    echo -n $p
    #mpirun -np $p python3 mpi_allreduce_speed.py | \
    #awk -F, '$0 ~ /^i[0-9]/ {printf(",%s", $2)} END {print ""}'
    mpirun -np $p python3 ptypy_moonflower_script_numpy.py -i 2000 -s 64 -n 200 | tail -n 10
done
```

Here, 1..8 will be 1..<cpu count - 3>. CPU count mean physical core count. (3 cores will be used by data streamer, IPR mock and training inference)