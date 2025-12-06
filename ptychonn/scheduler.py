import math
import time

# DP memoaization array for scheduling problem
memo = [] # for memoaization
succ = [] # for reconstructing solution
quality_mem = [] # for memoaization of quality values for each (i,j) state

def get_batch_time(i, j, nn_uf=0.00027, nn_ub=0.00035):
    # Placeholder function to return time taken for processing batch with i training sample and j inference sample
    # In a real scenario, this would be based on empirical data or a model
    return (i+j) * nn_uf + nn_ub * j  # Example: time increases with both i and j

def get_forward_pass_time(i, j, nn_uf=0.00027, nn_ub=0.00035):
    return (i+j) * nn_uf

MAX_FEASIBLE_BATCHSIZE=128

# DP scheduling solver
#
# each cell memo[i][j] represents the minimum time to process i training samples and j inference samples
# start_timepoint: the time point when scheduling starts
# A: arrival interval of inference samples
# D: deadline for each inference sample
# each solution memo[i][j] is constructed by considering all feasible (k,l) batch sizes and memo[i-k][j-l] 
# to process while maintainitng deadlines
# 0<=k+l<=MAX_FEASIBLE_BATCHSIZE 
def schedule_solver(gt_count, accum_while_gt_genereted, start_timepoint, A, D):
    # to count the loop to have an idea of complexity
    loop_count = 0

    for i in range(0, gt_count+1):
        for j in range(0, accum_while_gt_genereted+1):
            # if the else block is not there, we will initiate
            if i+j <= MAX_FEASIBLE_BATCHSIZE:
                memo[i][j] = get_batch_time(i, j)  # initialize with processing all at once
                succ[i][j] = (i, j)
                quality_mem[i][j] = 0
            else:
                # initialize with some possibly feasible value, if we find better it will get replace
                memo[i][j] = math.inf  
                succ[i][j] = None
                quality_mem[i][j] = 0

            found_feasible_incorporating_new_request = False
            # extra index check, Python allows negative indexing
            # so if not min'ed it will create unwanted effect
            for k in range(0, min(MAX_FEASIBLE_BATCHSIZE, i)+1):    
                # extra index check, Python allows negative indexing
                # so if not min'ed it will create unwanted effect
                for l in range(0, min(MAX_FEASIBLE_BATCHSIZE, j)+1):
                    loop_count += 1
                    # in an iteration maximum batch size is limited realistically due to memory constraints 
                    if k+l > MAX_FEASIBLE_BATCHSIZE:
                        break

                    # conditions to update memoization: new duration is smaller
                    # Notice: There are three parts
                    # 1. NN processing overhead for processing `k` training samples and `l` inference samples
                    # 2. Time of finish for processing `i-k` training samples and `j-l` inference samples
                    # 3. Time to wait until `l` inference samples arrived 
                    time_to_wait_until_l_inference_arrived = max(0, (start_timepoint +  j * A) - memo[i-k][j-l])
                    dur_kl_newiteration = memo[i-k][j-l] + get_batch_time(k, l) + time_to_wait_until_l_inference_arrived
                    quality = 0 if j == 0 else (i-k)/j # quality is how many training samples per inference sample in the current state
                    
                    # we do not consider the case that deadline is missed
                    # every processing needs to finish before deadline D of the earliest request from l inference samples
                    if time_to_wait_until_l_inference_arrived + get_forward_pass_time(k, l) > D:
                        # taking a larger `l` will only increase the time to wait
                        # so deadline will be missed for larger `l` as well
                        break

                    # found at least one feasible way by incorporating new request
                    found_feasible_incorporating_new_request = True
                    # following condition will choose one solution but it will be part of pareto front
                    if dur_kl_newiteration <= memo[i][j] and quality > quality_mem[i][j]:
                        memo[i][j] = dur_kl_newiteration
                        succ[i][j] = (k, l)
                        quality_mem[i][j] = quality

            # if we do not find any feasible way by incorporating new request
            # we will not get a feasible schedule serving all requests
            if not found_feasible_incorporating_new_request:
                print("loop count:", loop_count)
                print("serving every request is not feasible for given parameters")
                print("Max feasible served requests while generating all GT data:", j-1)
                return memo[gt_count][accum_while_gt_genereted]

    print("loop count:", loop_count)
    print("All requests can be served while generating {gt_count} GT data")
    return memo[gt_count][accum_while_gt_genereted]


TEST_K = 160
TEST_B = 160
TEST_D_RATE = 2000
TEST_DEADLINE = 0.2


if __name__ == "__main__":
    A = 1 / TEST_D_RATE
    Tk = TEST_K/TEST_B # time to generate TEST_K ground truth data
    accum_while_gt_genereted = int(Tk / A)

    # initialize memoization arrays
    for i in range(TEST_K + 1):
        memo.append([math.inf] * (int(Tk / A) + 1))
        succ.append([(0, 0)] * (int(Tk / A) + 1))
        quality_mem.append([0] * (int(Tk / A) + 1))

    print("Starting DP Scheduling...")
    print(f"Parameters: K={TEST_K}, B={TEST_B}, D_RATE={TEST_D_RATE}, DEADLINE={TEST_DEADLINE}")
    print(f"A={A}, Tk={Tk}, Tk/A={Tk/A}")
    # calculate optimal schedule
    start_time = time.time()

    dur = schedule_solver(TEST_K, accum_while_gt_genereted, 0,  A, TEST_DEADLINE)

    if memo[TEST_K][int(Tk/A)] == math.inf:
        S = None # no solution
        print("No feasible schedule found.")
        exit(0)

    i = TEST_K
    j = accum_while_gt_genereted
    S = []
    while i > 0 or j > 0:
        (k, t) = succ[i][j]
        S.append((k, t))
        (i, j) = (i-k, j-t)
    end_time = time.time()
    print("DP Scheduling Runtime: ", end_time - start_time)
    print("Optimal Duration: ", memo[TEST_K][int(Tk/A)])
    print("Optimal Schedule: ", S)
