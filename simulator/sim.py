from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import pandas as pd
from functools import lru_cache
import time

CLASSIC = 0
ML_TRAIN = 1
ML_INFER = 2
FAIL = -1
modes = { CLASSIC: "CLASSIC", ML_TRAIN: "ML_TRAIN", ML_INFER: "ML_INFER", FAIL: "FAIL"}

@dataclass
class Sample:
    def __init__(self, sid: int, arrival_ts: float, deadline_s: float):
        self.sid: int = sid
        self.arrival_ts: float = arrival_ts
        self.deadline_ts: float = deadline_s
        self.completed_ts: float = 0.0
        self.chosen_mode: int = FAIL
        self.batch_id: int = -1

    def get_info(self):
        return {"sid": self.sid, "arrival_ts": self.arrival_ts, "deadline_ts": self.deadline_ts,
                "completed_ts": self.completed_ts, "chosen_mode": modes[self.chosen_mode], "batch_id": self.batch_id}


class Simulator:
    def __init__(self, arrival_rate: int, classic_compute_rate: int, ml_batch_time_map: Dict[Tuple[int, int], float], deadline_s: int):
        
        assert arrival_rate > 0
        assert classic_compute_rate > 0
        assert deadline_s > 0
        assert len(ml_batch_time_map) > 0
        assert 1/deadline_s < arrival_rate, "Arrival rate must be less than the maximum processing rate to have any feasible schedule"
        
        self.num_samples = arrival_rate # assume that we simulate for 1 second
        self.arrival_rate = arrival_rate
        self.classic_rate = classic_compute_rate
        self.ml_batch_time_map = ml_batch_time_map
        self.deadline_s = deadline_s
        self.missed_sample_penalty = -1
        self.trained_sample_bonus = 0.1
        self.samples: List[Sample] = []
        for i in range(self.num_samples):
            arrival_ts = i / self.arrival_rate
            deadline_ts = arrival_ts + self.deadline_s
            self.samples.append(Sample(sid=i+1, arrival_ts=arrival_ts, deadline_s=deadline_ts))
        self.classic_per_sample_time = 1.0 / self.classic_rate
        num_classic_samples = self.deadline_s / self.classic_per_sample_time
        self.train_samples = int(num_classic_samples)    # number of samples for which we have ground truth and will train with
        self.infer_samples = self.num_samples - self.train_samples        # number of samples which we need to run inference on        
        self.sim_results = []

    def print_summary(self):
        for s in self.samples:
            print(s.get_info())

    def simulate(self):
        
        # The first m samples will be completed using CLASSIC mode (ground truth)
        for i in range(self.train_samples):
            s = self.samples[i]
            s.completed_ts = s.arrival_ts + self.classic_per_sample_time
            s.chosen_mode = CLASSIC
            # Now reset the arrival time of these samples; which will be used for ML training scheduling
            s.arrival_ts = s.completed_ts
        print(f"Classic mode: Processed {self.train_samples} samples with classic compute rate {self.classic_rate} samples/s, each taking {self.classic_per_sample_time:.4f}s")

        T = sorted(self.samples[:self.train_samples], key=lambda s: (s.arrival_ts, s.deadline_ts))
        I = sorted(self.samples[self.train_samples:], key=lambda s: (s.arrival_ts, s.deadline_ts))

        # Pre-extract arrays for speed
        T_arr = [s.arrival_ts for s in T]
        I_arr = [s.arrival_ts for s in I]
        I_dead = [s.deadline_ts for s in I]

        time_map = self.ml_batch_time_map
        allowed_pairs = sorted(time_map.keys(), reverse=False)  # list of (t,i) we allow

        results = []

        # -------- core algorithm: exhaustive search per batch size --------

        # Earliest time we can start a batch that takes t_take train and i_take infer samples,
        # beginning at indices t_idx / i_idx (contiguously), respecting arrivals.
        def feasible_start(now, t_take, i_take, t_idx, i_idx):
            latest_needed_arrival = now
            if t_take > 0:
                # arrival of the last train sample included
                latest_needed_arrival = max(latest_needed_arrival, T_arr[t_idx + t_take - 1])
            if i_take > 0:
                # arrival of the last inference sample included
                latest_needed_arrival = max(latest_needed_arrival, I_arr[i_idx + i_take - 1])
            return latest_needed_arrival

        # Number of inference samples among I[i_idx : i_idx+i_take] whose deadlines are < end_time
        def count_infer_misses(end_time, i_idx, i_take):
            miss = 0
            for k in range(i_idx, i_idx + i_take):
                if I_dead[k] < end_time + 1e-12: # tiny epsilon for numerical stability when I_deadline == end_time
                    miss += 1
            return miss

        def compute_score(missed_infers, num_inferred, num_trained):
            missed_score = missed_infers * self.missed_sample_penalty
            inferred_score = num_inferred * (num_trained * self.trained_sample_bonus)
            return missed_score + inferred_score
    

        # Return minimal missed inference deadlines from state:
        # - rem_batches batches remaining to schedule (must be >=1),
        # - we have already assigned T[:t_idx] and I[:i_idx],
        # - current time is 'now'.
        # We must consume all remaining (m - t_idx, n - i_idx) using exactly rem_batches batches.
        # 'now' is discretized for memoization by rounding to 0.1 ms to avoid float-key explosion while remaining accurate for scheduling.
        unfavorable_result = (float("inf"), 0.0, tuple(), -float("inf"))  # (missed_infers, num_trained, batch_list, score)
        @lru_cache(maxsize=None)
        def dp(rem_batches, t_idx, i_idx, now):
            prev_trained_on = t_idx # Previously, we trained on these many samples
            # Base case: last batch must consume everything
            if rem_batches == 1:
                t_take = self.train_samples - t_idx
                i_take = self.infer_samples - i_idx
                if (t_take, i_take) not in time_map: # impossible packing at this batch size
                    return unfavorable_result
                start = feasible_start(now, t_take, i_take, t_idx, i_idx)
                end = start + time_map[(t_take, i_take)]
                missed_here = count_infer_misses(end, i_idx, i_take)
                return missed_here, t_take, ((t_take, i_take),), compute_score(missed_here, i_take, prev_trained_on)

            best_miss = float("inf")
            best_trained = -1
            best_batches = tuple()
            best_score = -float("inf")
            t_rem = self.train_samples - t_idx
            i_rem = self.infer_samples - i_idx
            possible_best = compute_score(0, i_rem, self.train_samples)  # cannot do better than this

            for (t_take, i_take) in allowed_pairs:
                if (t_take == 0 and i_take == 0) or (t_take + i_take > self.num_samples):
                    continue
                if t_take > t_rem or i_take > i_rem:
                    continue

                start = feasible_start(now, t_take, i_take, t_idx, i_idx)
                end   = start + time_map[(t_take, i_take)]
                miss_here = count_infer_misses(end, i_idx, i_take)
                curr_score = compute_score(miss_here, i_take, prev_trained_on)
                tail_upper_bound_score = compute_score(0, i_rem - i_take, self.train_samples - (t_idx + t_take))
                if curr_score + tail_upper_bound_score < best_score:
                    continue

                tail_miss, tail_trained, tail_batches, tail_score = dp(
                    rem_batches - 1,
                    t_idx + t_take,
                    i_idx + i_take,
                    round(end, 4)  # discretize time for memoization
                )
                if tail_score == -float("inf"):
                    continue  # infeasible tail

                total_miss = miss_here + tail_miss
                total_trained = t_take + tail_trained
                total_score = curr_score + tail_score
                
                if (total_score > best_score) or (total_score == best_score and (total_miss < best_miss or (total_miss == best_miss and total_trained > best_trained))):
                    best_score = total_score
                    best_miss = total_miss
                    best_trained = total_trained
                    best_batches = ((t_take, i_take),) + tail_batches
                
                if total_score >= possible_best:
                    break  # cannot do better than this
            if best_score == -float("inf"):
                return unfavorable_result
            return best_miss, best_trained, best_batches, best_score
        
        # Sweep num_batches from 1 to (m+n)
        latest_arrival = min(I[0].arrival_ts, T[0].arrival_ts)
        overall_best_score = -float("inf")
        max_num_zero_missed_batches = 5 # stop after this many batch sizes with zero misses, to avoid long runtimes
        num_batch = 1
        while num_batch <= (self.train_samples + self.infer_samples):
            # dp.cache_clear()
            start_time = time.time()
            best_missed, best_num_trained, batches, best_score = dp(num_batch, 0, 0, latest_arrival)
            end_time = time.time()
            if best_score >= overall_best_score:
                overall_best_score = best_score
            if best_missed == 0:
                max_num_zero_missed_batches -= 1

            print(f"Num batch: {num_batch}, Time taken: {end_time - start_time:.4f}s, Inferences missed: {best_missed}, Best trained: {best_num_trained}, Score: {best_score:.4f}, Overall best score: {overall_best_score:.4f}, Possible best score: {compute_score(0, self.infer_samples, self.train_samples):.4f}")

            if best_score < overall_best_score or max_num_zero_missed_batches <= 0:
                # No need to continue with larger num_batches if score is degrading
                # because larger num_batches take long for simulation
                print("Stopping further batch size exploration as score is degrading.")
                break
            res = {"batches": num_batch, "batches": batches, "min_missed_infers": (0 if best_missed == float("inf") else int(best_missed)), "num_trained": best_num_trained, "score": round(best_score, 4)}
            results.append(res)
            if best_score == compute_score(0, self.infer_samples, self.train_samples):
                break  # cannot do better than this
            num_batch += 1
        self.sim_results = results
        return results
    
    def get_outcome(self):
        t_idx = 0
        i_idx = 0
        now = 0
        best_result = max(self.sim_results, key=lambda r: r['score'])
        for batch_id, (ntrain, ninfer) in enumerate(best_result['batches']):
            now += self.ml_batch_time_map[(ntrain, ninfer)]
            # Process training samples
            for _ in range(ntrain):
                s = self.samples[t_idx]
                s.completed_ts = now    
                s.chosen_mode = ML_TRAIN
                s.batch_id = batch_id + 1
                t_idx += 1
            # Process inference samples
            for _ in range(ninfer):
                s = self.samples[self.train_samples + i_idx]
                s.completed_ts = now
                s.chosen_mode = ML_INFER
                s.batch_id = batch_id + 1
                i_idx += 1
        p = pd.DataFrame([s.get_info() for s in self.samples])
        print(p.to_string(index=False))
        return p


def build_ml_batch_time_map(df: pd.DataFrame):
    # base series from CSV
    fwd = dict(zip(df["Batch Size"], df["Forward"]))
    bwd = dict(zip(df["Batch Size"], df["Backward"]))
    inf = dict(zip(df["Batch Size"], df["Inference"]))

    # sorted keys for interpolation/extrapolation
    keys = sorted(fwd.keys())
    if keys != sorted(bwd.keys()) or keys != sorted(inf.keys()):
        raise ValueError("Forward/Backward/Inference must share the same batch-size grid.")

    xgrid = keys
    xmin, xmax = xgrid[0], xgrid[-1]

    @lru_cache(None)
    def interp(series_name: str, k: int) -> float:
        """
        Return value for batch size k from series 'Forward'/'Backward'/'Inference':
        - exact if present
        - linear interpolation if xmin < k < xmax
        - linear extrapolation using edge slope if k < xmin or k > xmax
        """
        serie = {"Forward": fwd, "Backward": bwd, "Inference": inf}[series_name]
        if k in serie:
            return float(serie[k])

        # find neighbors
        if k < xmin:
            x1, x2 = xgrid[0], xgrid[1]
        elif k > xmax:
            x1, x2 = xgrid[-2], xgrid[-1]
        else:
            # inside range: find bracketing points
            # xgrid is small; linear scan is fine
            x1 = max(x for x in xgrid if x <= k)
            x2 = min(x for x in xgrid if x >= k)
            if x1 == x2:
                return float(serie[x1])

        y1, y2 = float(serie[x1]), float(serie[x2])
        # linear interpolation/extrapolation
        return y1 + (y2 - y1) * ((k - x1) / (x2 - x1))

    # helper to enumerate all (t,i) with t+i = power of two
    def ti_pairs_for_sum(z: int):
        return [(t, z - t) for t in range(0, z + 1)]

    # powers of two up to 256 inclusive
    powers_of_two = [1 << p for p in range(0, 9)]  # 1,2,4,...,256
    ml_map = {}
    rows = []

    for z in powers_of_two:
        for (t, i) in ti_pairs_for_sum(z):
            if t == 0 and i == 0:
                continue

            # compute times via interpolation/extrapolation
            if t == 0:
                # pure inference
                time = interp("Inference", i)
            elif i == 0:
                # pure training
                time = interp("Forward", t) + interp("Backward", t)
            else:
                # mixed: forward over (t+i), backward over t
                time = interp("Forward", t + i) + interp("Backward", t)

            ml_map[(t, i)] = float(time)
            rows.append({"train": t, "infer": i, "sum": t + i, "time": float(time)})

    table = pd.DataFrame(rows).sort_values(["sum", "train", "infer"]).reset_index(drop=True)
    return ml_map, table

if __name__ == "__main__":
    ml_times = pd.read_csv("resnet-50-batch-times.csv", delimiter=',')
    ml_map, table = build_ml_batch_time_map(ml_times)
    max_arrival_rate = 64 # samples/s
    deadline_s = 0.6  # seconds
    classical_compute_rate = 4  # samples/s

    assert 1/deadline_s < max_arrival_rate, "Arrival rate must be less than the maximum processing rate to have any feasible schedule"
    assert classical_compute_rate >= 1/deadline_s, "Classical compute rate must be at least the maximum processing rate to have any feasible schedule"
    
    print("----------------------------------------")
    print("Arrival rate limit (samples/s):", max_arrival_rate)
    print("ML batch time map:")
    print(table) #.to_string(index=False))

    simulator = Simulator(arrival_rate=max_arrival_rate, classic_compute_rate=classical_compute_rate, ml_batch_time_map=ml_map, deadline_s=deadline_s)
    print("---------------------------------------- DP version ----------------------------------------")
    results = simulator.simulate()
    for res in results:
        print(res)
    print("---------------------------------------- Outcome ----------------------------------------")
    simulator.get_outcome()