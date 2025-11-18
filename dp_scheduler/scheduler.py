import math
import time

# DP memoaization array for scheduling problem
memo = [] # for memoaization
succ = [] # for reconstructing solution

def get_batch_time(i, j, nn_uf=0.00027, nn_ub=0.00035):
    # Placeholder function to return time taken for processing batch with i training sample and j inference sample
    # In a real scenario, this would be based on empirical data or a model
    return (i+j) * nn_uf + nn_ub * j  # Example: time increases with both i and j

def f(k, served, start, Tk, A, D):
    print("f({0}, {1}, {2}) recursive call".format(k, served, start))
    if memo[k][served] < math.inf:
        print("returning by exiting early as memoaization found")
        return memo[k][served]

    if served == int(Tk/A):
        if start + get_batch_time(k, 0) < memo[k][served]:
            memo[k][served] = start + get_batch_time(k, 0)
            succ[k][0] = (0, 0)
        print("returning served == int(Tk/A)")
        return memo[k][served]
    
    remaining = int(Tk/A) - served

    for i in range(k, -1, -1):
        for j in range(remaining, 0, -1):
            print(f"Considering k={k}, served={served}, i={i}, j={j}, start={start}")
            Tij = get_batch_time(i, j)

            if start + Tij > A * served + D: # oldest remaining will miss deadline
                continue

            print("f({0}, {1}, {2}) recursive call".format(k-i, served + j, start + Tij))
            duration = f(k - i, served + j, start + Tij, Tk, A, D)
            # this is done to select subsolution which gives the minimum duration
            if duration < memo[k][served]:
                memo[k][served] = duration
                succ[k][served] = (k - i, served + j)
                # break # no need to check smaller j, we do not thrive for smaller duration
    
    print("returning served after finishing loop")
    return memo[k][served]


TEST_K = 20
TEST_B = 16
TEST_D_RATE = 40
TEST_DEADLINE = 1


if __name__ == "__main__":
    A = 1 / TEST_D_RATE
    Tk = TEST_K/TEST_B # time to generate TEST_K ground truth data

    # initialize memoization arrays
    for i in range(TEST_K + 1):
        memo.append([math.inf] * (int(Tk / A) + 1))
        succ.append([(0, 0)] * (int(Tk / A) + 1))


    if memo[TEST_K][int(Tk/A)] == math.inf:
        S = None # no solution

    print("Starting DP Scheduling...")
    print(f"Parameters: K={TEST_K}, B={TEST_B}, D_RATE={TEST_D_RATE}, DEADLINE={TEST_DEADLINE}")
    print(f"A={A}, Tk={Tk}, Tk/A={Tk/A}")
    # calculate optimal schedule
    start_time = time.time()
    dur = f(TEST_K, 0, 0, Tk, A, TEST_DEADLINE)

    i = TEST_K
    j = int(Tk/A)
    S = []
    while i > 0 and j > 0:
        (k, t) = succ[i][j]
        S.append((i - k, j - t))
        (i, j) = (k, t)
    end_time = time.time()
    print("DP Scheduling Runtime: ", end_time - start_time)
    print("Optimal Duration: ", memo[TEST_K][int(Tk/A)])
    print("Optimal Schedule: ", S)
