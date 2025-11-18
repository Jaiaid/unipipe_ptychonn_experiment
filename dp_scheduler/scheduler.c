#include <stdio.h>
#include <stdlib.h>


#define NN_UF 0.00027
#define NN_UB 0.00031
#define INF 1e10


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


float schedule_solver(
    int traindatset_size, int served_count, float passed_time,
    float traindatset_gentime, const float interarrival_time, const float deadline_sec,
    float **min_dur_memo_ara, schedule_entry** schedule_ara
)
{
    int accumulated = (int)(traindatset_gentime/interarrival_time);

    if (min_dur_memo_ara[traindatset_size][served_count] < INF) {
        return min_dur_memo_ara[traindatset_size][served_count];
    }

    if (served_count == accumulated) {
        float iteration_time = estimate_iteration_time(
            traindatset_size, 0, NN_UF, NN_UB
        );

        if (passed_time + iteration_time < min_dur_memo_ara[traindatset_size][served_count]) {
            min_dur_memo_ara[traindatset_size][served_count] = passed_time + iteration_time;
            schedule_entry tmp = {0, 0};
            schedule_ara[traindatset_size][0] = tmp;
        }

        return min_dur_memo_ara[traindatset_size][served_count];
    }

    int i, j;
    int remaining = accumulated - served_count;
    for (i=traindatset_size;i>=0;i--)
    {
        for(j=remaining;j>=1;j--)
        {
            float iteration_time = estimate_iteration_time(
                i, j, NN_UF, NN_UB
            );

            // oldest remaining will miss the deadline
            if (passed_time + iteration_time > interarrival_time*served_count + deadline_sec) {
                continue;
            }

            float duration = schedule_solver(
                traindatset_size - i, served_count + j, passed_time + iteration_time,
                traindatset_gentime, interarrival_time, deadline_sec,
                min_dur_memo_ara, schedule_ara
            );

            // this is done to select subsolution which gives the minimum duration
            if (duration < min_dur_memo_ara[traindatset_size][served_count]) {
                min_dur_memo_ara[traindatset_size][served_count] = duration;
                schedule_entry tmp = {traindatset_size - i, served_count + j};
                schedule_ara[traindatset_size][served_count] = tmp; 
            }
        }
    }

    return min_dur_memo_ara[traindatset_size][served_count];
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
        memo_ptr[i] = (float *)malloc(sizeof(float) * (accum_infer_count+1));
        schedule_solution_ptr[i] = (schedule_entry *)malloc(sizeof(schedule_entry) * (accum_infer_count+1));
    
        int j;
        for(j=0;j<=accum_infer_count;j++)
        {
            memo_ptr[i][j] = INF;
        }
    }

    
    printf("Initiating Schedule Calculation\n");
    // calculate the schedule
    schedule_solver(
        K, 0, 0, Tk, 1.0/d_rate, D, memo_ptr, schedule_solution_ptr
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

        printf("{%d, %d}, ", i - tmp.train_minibatch_size, j - tmp.infer_minibatch_size);
        i = tmp.train_minibatch_size;
        j = tmp.infer_minibatch_size;
    }
    printf("\n");

    return 0;
}