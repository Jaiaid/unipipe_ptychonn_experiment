import math

from . import unipipe_scheduler


def estimate_T_IPR_pretrained(
        phase_retrieval_genrate: float, acquisition_rate: float,
        deadline_sec: float, nn_uf:float=0.0005, nn_ub:float=0.0015, epochs:int=1) -> float:

    T_IPR = max(
        1/phase_retrieval_genrate,
        min(
            deadline_sec, (deadline_sec + 1/acquisition_rate)/(
                1 +
                (acquisition_rate - phase_retrieval_genrate)*nn_uf
            )
        )
    )

    n_train = max(1, math.floor(phase_retrieval_genrate * T_IPR))
    print("Estimated Pretrained Interval :", T_IPR, n_train, math.floor((acquisition_rate-phase_retrieval_genrate) * T_IPR))

    return T_IPR


def estimate_T_IPR(
        phase_retrieval_genrate: float, acquisition_rate: float,
        deadline_sec: float, nn_uf:float=0.0005, nn_ub:float=0.0015, epochs:int=1) -> float:

    T_IPR = max(
        1/phase_retrieval_genrate,
        min(
            deadline_sec, (deadline_sec + 1/acquisition_rate)/(
                1 +
                (acquisition_rate - phase_retrieval_genrate)*nn_uf +
                phase_retrieval_genrate*(nn_uf + nn_ub)*epochs
            )
        )
    )

    n_train = max(1, math.floor(phase_retrieval_genrate * T_IPR))
    print("Estimated Interval :", T_IPR, n_train, math.floor((acquisition_rate-phase_retrieval_genrate) * T_IPR))

    return T_IPR

def estimate_T_IPR_unipipe(
        phase_retrieval_genrate: float, acquisition_rate: float,
        deadline_sec: float, nn_uf:float=0.0005, nn_ub:float=0.0015, epochs:int=1) -> float:

    inferbs = 64
    trainbs = 64

    # read profile data to get
    forward_time_per_sample = []
    backward_time_per_sample = []
    with open("ptychonn/benchmark_ptychonn_nn_step.csv") as f:
        for line in f.readlines()[1:]:
            tokens = line.split()
            bs = int(tokens[0])
            fwd_time = float(tokens[4])
            bwd_time = float(tokens[6])
            forward_time_per_sample.append(fwd_time/bs)
            backward_time_per_sample.append(bwd_time/bs)

    while True:
        nn_uf = forward_time_per_sample[inferbs-1]
        nn_ub = backward_time_per_sample[trainbs-1]
        T_IPR = (deadline_sec + 1/acquisition_rate)/(
            1 +
            (acquisition_rate - phase_retrieval_genrate)*nn_uf +
            phase_retrieval_genrate*(nn_uf + nn_ub)*epochs
        )

        n_train = max(1, math.floor(phase_retrieval_genrate * T_IPR))
        print("Estimated Interval :", T_IPR, n_train, math.floor((acquisition_rate-phase_retrieval_genrate) * T_IPR))
        if n_train >= trainbs or trainbs == 1:
            break

        trainbs = n_train
        inferbs = min(inferbs, math.floor((acquisition_rate-phase_retrieval_genrate) * T_IPR))

    return T_IPR, trainbs, inferbs

def estimate_T_IPR_unipipe(
        phase_retrieval_genrate: float, acquisition_rate: float,
        deadline_sec: float, nn_uf:float=0.0005, nn_ub:float=0.0015, epochs:int=1) -> float:

    inferbs = 64
    trainbs = 64

    # read profile data to get
    forward_time_per_sample = []
    backward_time_per_sample = []
    with open("ptychonn/benchmark_ptychonn_nn_step.csv") as f:
        for line in f.readlines()[1:]:
            tokens = line.split()
            bs = int(tokens[0])
            fwd_time = float(tokens[4])
            bwd_time = float(tokens[6])
            forward_time_per_sample.append(fwd_time/bs)
            backward_time_per_sample.append(bwd_time/bs)

    while True:
        nn_uf = forward_time_per_sample[inferbs-1]
        nn_ub = backward_time_per_sample[trainbs-1]
        T_IPR = (deadline_sec + 1/acquisition_rate)/(
            1 +
            (acquisition_rate - phase_retrieval_genrate)*nn_uf +
            phase_retrieval_genrate*(nn_uf + nn_ub)*epochs
        )

        n_train = max(1, math.floor(phase_retrieval_genrate * T_IPR))
        print("Estimated Interval :", T_IPR, n_train, math.floor((acquisition_rate-phase_retrieval_genrate) * T_IPR))
        if n_train >= trainbs or trainbs == 1:
            break

        trainbs = n_train
        inferbs = min(inferbs, math.floor((acquisition_rate-phase_retrieval_genrate) * T_IPR))

    return T_IPR, trainbs, inferbs

def estimate_unipipe_schedule(
        phase_retrieval_genrate: float, acquisition_rate: float,
        deadline_sec: float, ground_truth_count:int, epochs:int=1) -> float:

    interarrival_gap = 1 / acquisition_rate
    Tk = ground_truth_count/phase_retrieval_genrate # time to generate TEST_K ground truth data
    accum_while_gt_genereted = int(Tk / interarrival_gap)

    # read profile data to get
    forward_time = []
    backward_time = []
    with open("ptychonn/benchmark_ptychonn_nn_step.csv") as f:
        for line in f.readlines()[1:]:
            tokens = line.split()
            bs = int(tokens[0])
            fwd_time = float(tokens[4])
            bwd_time = float(tokens[6])
            forward_time.append(fwd_time)
            backward_time.append(bwd_time)

    solution = unipipe_scheduler.resolve_schedule(
        ground_truth_count, accum_while_gt_genereted, 0,  interarrival_gap,
        deadline_sec, forward_time, backward_time
    )

    # build schedule
    i = ground_truth_count
    j = accum_while_gt_genereted
    S = []
    # print(solution)
    # the schedule will be built in reverse order
    while i > 0 or j > 0:
        (k, t) = solution[i][j][0:2]
        q = solution[i][j][3]
        (k, t) = (int(k), int(t))
        # S.append((k, t, q))
        S.append((k, t))
        (i, j) = (i-k, j-t)
    S.reverse()

    T_IPR = ground_truth_count / phase_retrieval_genrate

    return T_IPR, S

    # return max(
    #     1/phase_retrieval_genrate,
    #     min(
    #         deadline_sec,
    #         deadline_sec / ((acquisition_rate - phase_retrieval_genrate)*nn_uf+phase_retrieval_genrate*(nn_uf + nn_ub))
    #     )
    # )

# def estimate_IPR_time(IPR_throughput, datarate, deadline):
#     return IPR_throughput/(datarate * (datarate - IPR_throughput))

# def estimate_train_test_ratio(IPR_throughput, datarate, deadline):
#     return (IPR_throughput**2)/(datarate * (datarate - IPR_throughput))

def estimate_infer_bs(datarate, deadline):
    return math.ceil((1 + deadline * datarate) / 2)

def estimate_train_bs(infer_bs, time_uf, time_ub, datarate):
    return math.floor(infer_bs * (1-datarate*time_uf) /\
                        ((time_uf + time_ub) * datarate))