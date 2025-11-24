import unipipe_scheduler
import math
import time


TEST_K = 160
TEST_B = 160
TEST_D_RATE = 2000
TEST_DEADLINE = 0.2

BENCHMARK_FILE = "../ptychonn/benchmark_ptychonn_nn_step.csv"


if __name__ == "__main__":
    A = 1 / TEST_D_RATE
    Tk = TEST_K/TEST_B # time to generate TEST_K ground truth data
    accum_while_gt_genereted = int(Tk / A)

    print("loading benchmark data from ", BENCHMARK_FILE)
    bs_fw_benchmark = []
    bs_bw_benchmark = []
    with open(BENCHMARK_FILE, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            items = line.strip().split()
            bs_fw_benchmark.append(float(items[4]))
            bs_bw_benchmark.append(float(items[6]))


    print("Calling DP Scheduling from cpp backend implementation...")
    print(f"Parameters: K={TEST_K}, B={TEST_B}, D_RATE={TEST_D_RATE}, DEADLINE={TEST_DEADLINE}")
    print(f"A={A}, Tk={Tk}, Tk/A={Tk/A}")
    # calculate optimal schedule
    start_time = time.time()

    solution = unipipe_scheduler.resolve_schedule(
        TEST_K, accum_while_gt_genereted, 0,  A, TEST_DEADLINE, bs_fw_benchmark, bs_bw_benchmark
    )
    print(len(solution), len(solution[0]), len(solution[0][0]))

    if solution[TEST_K][int(Tk/A)][2] == math.inf:
        S = None # no solution
        print("No feasible schedule found.")
        exit(0)

    i = TEST_K
    j = accum_while_gt_genereted
    S = []
    while i > 0 or j > 0:
        (k, t) = solution[i][j][0:2]
        (k, t) = (int(k), int(t))
        S.append((k, t))
        (i, j) = (i-k, j-t)
    end_time = time.time()
    print("DP Scheduling Runtime: ", end_time - start_time)
    print("Optimal Duration: ", solution[TEST_K][int(Tk/A)][2])
    print("Optimal Schedule: ", S)
