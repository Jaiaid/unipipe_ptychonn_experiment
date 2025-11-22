#include <cstdio>
#include <cstdlib>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// this complex type is to hold dp ara information
// the inner most vector just hold a 3 element vector
// 1. train minibatch size, 
// 2. infer mini batch size,
// 3. duration to cover upto i training samples and j inference samples
typedef std::vector<std::vector<std::vector<float>>> schedule_entry_matrix;


#define NN_UF 0.00027 // defined for now, for proper implementation, we can read from data file or some performance model
#define NN_UB 0.00036 // defined for now, for proper implementation, we can read from data file or some performance model
#define MAX_BATCH_SIZE 128 // maximum feasible batch size for iteration due to memory constraint
#define INF 1e10 // for memoization initialization

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))


float estimate_iteration_time(
    int train_size, int infer_size, float nn_uf, float nn_ub
)
{
    return (train_size + infer_size)*nn_uf + train_size*nn_ub;
}

float estimate_forward_pass_time(
    int train_size, int infer_size, float nn_uf, float nn_ub
)
{
    return (train_size + infer_size)*nn_uf;
}



/*
 * DP scheduling solver
 * each cell memo[i][j] represents the minimum time to process i training samples and j inference samples
 * start_timepoint: the time point when scheduling starts
 * traindatset_size: total training samples to be processed
 * accum_while_gt_genereted: total inference samples to be processed while generating all traindatset_size data
 * start_timepoint: the time point when scheduling starts
 * interarrival_time: arrival interval of inference samples
 * deadline_sec: deadline for each inference sample
 * min_dur_memo_ara: memoization array to store minimum duration to process i training samples and j inference samples
 * schedule_ara: to store the schedule solution
 *
 *
 * each solution memo[i][j] is constructed by considering all feasible (k,l) batch sizes and memo[i-k][j-l] 
 * to process while maintainitng deadlines
 * 0<=k, l<MAX_FEASIBLE_BATCHSIZE
 */
schedule_entry_matrix schedule_solver(
    int traindatset_size, int accum_while_gt_generated, float start_timepoint,
    const float interarrival_time, const float deadline_sec
)
{
    // to hold the dp solution
    // initiating the array with appropriate sizes
    schedule_entry_matrix dp_ara = schedule_entry_matrix(traindatset_size+1);
    for(int i=0;i<=traindatset_size;i++)
    {
        dp_ara[i] = std::vector<std::vector<float>>(accum_while_gt_generated+1, std::vector<float>(3));
    }

    int upto_train_size, upto_infer_size;
    for(upto_train_size=0;upto_train_size<=traindatset_size;upto_train_size++)
    {
        for(upto_infer_size=0;upto_infer_size<=accum_while_gt_generated;upto_infer_size++)
        {
            if (upto_train_size + upto_infer_size <= MAX_BATCH_SIZE) {
                dp_ara[upto_train_size][upto_infer_size][2] = estimate_iteration_time(upto_train_size, upto_infer_size, NN_UF, NN_UB);
                dp_ara[upto_train_size][upto_infer_size][0] = upto_train_size;
                dp_ara[upto_train_size][upto_infer_size][1] = upto_infer_size;
            }
            else {
                // initiate with a large value
                dp_ara[upto_train_size][upto_infer_size][2] = INF;
            }

            // for subsolution indexing
            int t_idx, i_idx;
            // we assume subsolution won't be found
            // this has some implication, check the limit always in loop and starts index incrementally
            // otherwise no solution may be found although there is, because we break the loop due to negative idx
            bool solution_found = false;

            int t_idx_limit = MIN(MAX_BATCH_SIZE, upto_train_size);
            for(t_idx=0;t_idx<=t_idx_limit;t_idx++)
            {
                int i_idx_limit = MIN(MAX_BATCH_SIZE, upto_infer_size);
                for(i_idx=0;i_idx<=i_idx_limit;i_idx++)
                {
                    // not acceptable condition due to memory constraint
                    if(t_idx + i_idx > MAX_BATCH_SIZE) {
                        break;
                    }

                    float iteration_time = estimate_iteration_time(
                        t_idx, i_idx, NN_UF, NN_UB
                    );
                    // conditions to update memoization: new duration is smaller
                    // Notice: There are three parts
                    // 1. NN processing overhead for processing `k` training samples and `l` inference samples
                    // 2. Time of finish for processing `i-k` training samples and `j-l` inference samples
                    // 3. Time to wait until `l` inference samples arrived 
                    float time_to_wait_until_l_inference_arrived = MAX(
                        0, (start_timepoint +  upto_infer_size * interarrival_time) - dp_ara[upto_train_size-t_idx][upto_infer_size-i_idx][2]
                    );
                    
                    float dur_kl_newiteration = 
                    dp_ara[upto_train_size-t_idx][upto_infer_size-i_idx][2] +
                    iteration_time +
                    time_to_wait_until_l_inference_arrived;

                    // we do not consider the case that deadline is missed
                    // every processing needs to finish before deadline D of the earliest request from l inference samples
                    if (time_to_wait_until_l_inference_arrived + estimate_forward_pass_time(t_idx, i_idx, NN_UF, NN_UB) > deadline_sec) {
                        break;
                    }

                    // found at least one feasible way by incorporating new request
                    solution_found = true;
                    if (dur_kl_newiteration < dp_ara[upto_train_size][upto_infer_size][2]) {
                        dp_ara[upto_train_size][upto_infer_size][2] = dur_kl_newiteration;
                        dp_ara[upto_train_size][upto_infer_size][0] = t_idx;
                        dp_ara[upto_train_size][upto_infer_size][1] = i_idx; 
                    }
                }
            }
            
            // if we do not find any feasible way by incorporating new request
            // we will not get a feasible schedule serving all requests
            if (!solution_found) {
                printf("No feasible subsolution found for traindatset_size=%d, served_count=%d\n", traindatset_size, accum_while_gt_generated);
                printf("Serving every request is not feasible for given parameters\n");
                printf("Max feasible served requests while generating all GT data: %d\n", upto_infer_size - 1);
                return dp_ara;
            }
        }
    }

    return dp_ara;
}

PYBIND11_MODULE(unipipe_scheduler, m, pybind11::mod_gil_not_used()) {
    m.doc() = "pybind11 plugin for fast unipipe schedule solver";

    m.def(
        "resolve_schedule", &schedule_solver,
        "Function to determine the schedule. The schedule is returned as [traindataset size][inference request count][3] list"
    );
}
