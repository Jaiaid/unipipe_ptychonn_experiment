#include <stdio.h>
#include <stdlib.h>


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


typedef struct {
    int train_minibatch_size;
    int infer_minibatch_size;
} schedule_entry;

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
float schedule_solver(
    int traindatset_size, int accum_while_gt_generated, float start_timepoint,
    const float interarrival_time, const float deadline_sec,
    float **min_dur_memo_ara, schedule_entry** schedule_ara
)
{
    int upto_train_size, upto_infer_size;
    for(upto_train_size=0;upto_train_size<=traindatset_size;upto_train_size++)
    {
        for(upto_infer_size=1;upto_infer_size<=accum_while_gt_generated;upto_infer_size++)
        {
            // initiate with a large value
            min_dur_memo_ara[upto_train_size][upto_infer_size] = INF;
            schedule_entry tmp = {upto_train_size, upto_infer_size};

            // for subsolution indexing
            int t_idx, i_idx;
            // we assume subsolution won't be found
            // this has some implication, check the limit always in loop and starts index incrementally
            // otherwise no solution may be found although there is, because we break the loop due to negative idx
            int solution_found = 0;

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
                        0, (start_timepoint +  i_idx * interarrival_time) - min_dur_memo_ara[upto_train_size-t_idx][upto_infer_size-i_idx]
                    );
                    
                    float dur_kl_newiteration = 
                    min_dur_memo_ara[upto_train_size-t_idx][upto_infer_size-i_idx] +
                    iteration_time +
                    time_to_wait_until_l_inference_arrived;

                    // we do not consider the case that deadline is missed
                    // every processing needs to finish before deadline D of the earliest request from l inference samples
                    // 0 for training samples, because we get the inference response after forward pass
                    if (time_to_wait_until_l_inference_arrived + estimate_iteration_time(0, i_idx, NN_UF, NN_UB) > deadline_sec) {
                        break;
                    }

                    // found at least one feasible way by incorporating new request
                    solution_found = 1;
                    if (dur_kl_newiteration < min_dur_memo_ara[upto_train_size][upto_infer_size]) {
                        min_dur_memo_ara[upto_train_size][upto_infer_size] = dur_kl_newiteration;
                        schedule_entry tmp = {t_idx, i_idx};
                        schedule_ara[upto_train_size][upto_infer_size] = tmp; 
                    }
                }
            }
            
            // if we do not find any feasible way by incorporating new request
            // we will not get a feasible schedule serving all requests
            if (!solution_found) {
                printf("No feasible subsolution found for traindatset_size=%d, served_count=%d\n", traindatset_size, accum_while_gt_generated);
                printf("Serving every request is not feasible for given parameters\n");
                printf("Max feasible served requests while generating all GT data: %d\n", upto_infer_size - 1);
                return INF;
            }
        }
    }

    return min_dur_memo_ara[traindatset_size][accum_while_gt_generated];
}


int main()
{
    int K = 20;
    float d_rate = 20;
    float D = 1;
    float B = 16;

    scanf("%d%f%f%f", &K, &d_rate, &D, &B);

    float Tk = K/B;

    float **memo_ptr;
    schedule_entry **schedule_solution_ptr;

    int i, j; // for loop variable
    int accum_infer_count = (int)(Tk*d_rate);

    // initialize the array
    printf("Initializing Memoaization Arrays\n");
    memo_ptr = (float **)malloc(sizeof(float *)*K);
    schedule_solution_ptr = (schedule_entry **)malloc(sizeof(schedule_entry *) * K);
    for (i = 0;i <= K;i++)
    {
        // initiaization not needed as the algorithm will fill all entries
        memo_ptr[i] = (float *)malloc(sizeof(float) * (accum_infer_count+1));
        schedule_solution_ptr[i] = (schedule_entry *)malloc(sizeof(schedule_entry) * (accum_infer_count+1));
    }

    
    printf("Initiating Schedule Calculation\n");
    // calculate the schedule
    schedule_solver(
        K, accum_infer_count, 0, 1.0/d_rate, D, memo_ptr, schedule_solution_ptr
    );

    if (memo_ptr[K][accum_infer_count] > INF - 10) {
        printf("No Solution Found!!!!!\n");
        return 0;
    }

    printf("Optimal Schedule Run Duration: %f second\n\nOptimal Schedule: ", memo_ptr[K][accum_infer_count]);
    i=K;
    j=accum_infer_count;
    while (i > 0 && j > 0)
    {
        schedule_entry tmp = schedule_solution_ptr[i][j];

        printf("{%d, %d}, ", tmp.train_minibatch_size, tmp.infer_minibatch_size);
        i = i - tmp.train_minibatch_size;
        j = j - tmp.infer_minibatch_size;
    }
    printf("\n");

    return 0;
}