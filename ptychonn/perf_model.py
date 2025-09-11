import math


def estimate_T_IPR(
        phase_retrieval_genrate: float, acquisition_rate: float,
        deadline_sec: float, nn_uf:float=0.0005, nn_ub:float=0.0015):

    return max(
        1/phase_retrieval_genrate,
        min(
            deadline_sec,
            deadline_sec / ((acquisition_rate - phase_retrieval_genrate)*nn_uf+phase_retrieval_genrate*(nn_uf + nn_ub))
        )
    )

# def estimate_IPR_time(IPR_throughput, datarate, deadline):
#     return IPR_throughput/(datarate * (datarate - IPR_throughput))

# def estimate_train_test_ratio(IPR_throughput, datarate, deadline):
#     return (IPR_throughput**2)/(datarate * (datarate - IPR_throughput))

def estimate_infer_bs(datarate, deadline):
    return math.ceil((1 + deadline * datarate) / 2)

def estimate_train_bs(infer_bs, time_uf, time_ub, datarate):
    return math.floor(infer_bs * (1-datarate*time_uf) /\
                        ((time_uf + time_ub) * datarate))