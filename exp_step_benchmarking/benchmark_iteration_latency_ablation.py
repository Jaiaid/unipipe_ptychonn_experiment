import argparse
import csv
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class GenerativeNet64(nn.Module):
    """~1.25M parameter conv generator: 64x64x1 -> 64x64x1."""

    def __init__(self, channels: int = 62):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 4, channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 4, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_eval_batch_schedule(name: str, eval_iters: int, max_bs: int, rng: random.Random):
    if name == "constant_32":
        return [max_bs] * eval_iters

    if name == "constant_16":
        return [16] * eval_iters

    if name == "constant_1":
        return [1] * eval_iters

    if name == "increasing_1_to_32":
        # Maps 100 eval iterations over 32 batch sizes (~3.125 iterations each).
        return [1 + (i * max_bs) // eval_iters for i in range(eval_iters)]

    if name == "decreasing_32_to_1":
        return [max_bs - (i * max_bs) // eval_iters for i in range(eval_iters)]

    if name == "random_1_to_32":
        return [rng.randint(1, max_bs) for _ in range(eval_iters)]

    if name == "random_pow2_set_1_2_4_8_16_32":
        batch_candidates = [1, 2, 4, 8, 16, 32]
        return [rng.choice(batch_candidates) for _ in range(eval_iters)]

    raise ValueError(f"Unknown schedule: {name}")


def run_schedule(
    model: nn.Module,
    schedule_name: str,
    batch_sizes_eval: list[int],
    warmup_iters: int,
    device: torch.device,
    seed: int,
):
    rng = random.Random(seed + hash(schedule_name) % 10_000)

    if schedule_name in {
        "random_1_to_32",
        "random_pow2_set_1_2_4_8_16_32",
    }:
        warmup_batches = [rng.randint(1, 32) for _ in range(warmup_iters)]
    else:
        warmup_batches = [batch_sizes_eval[0]] * warmup_iters

    full_schedule = warmup_batches + batch_sizes_eval
    measured_times_ms = []
    backward_measured_iters = 10

    model.train()  # keep grad-enabled behavior

    for i, bs in enumerate(full_schedule):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        # Whole-iteration timing starts before tensor creation (dataload surrogate).
        x_cpu = torch.randn(bs, 1, 64, 64, dtype=torch.float32)
        x = x_cpu.to(device, non_blocking=False).requires_grad_(True)

        out = model(x)

        # Run backward on first 10 measured iterations for every schedule.
        do_backward = warmup_iters <= i < (warmup_iters + backward_measured_iters)
        if do_backward:
            loss = (out * out).mean()
            loss.backward()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        if i >= warmup_iters:
            measured_times_ms.append((t1 - t0) * 1000.0)

    return measured_times_ms, full_schedule


def save_raw_csv(path: Path, all_results: dict, warmup_iters: int):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["schedule", "iter_idx", "batch_size", "latency_ms", "is_warmup"])
        for schedule_name, data in all_results.items():
            full_schedule = data["full_schedule"]
            measured = data["measured_times_ms"]

            for i in range(warmup_iters):
                writer.writerow([schedule_name, i, full_schedule[i], "", 1])

            for j, latency in enumerate(measured):
                idx = warmup_iters + j
                writer.writerow([schedule_name, idx, full_schedule[idx], f"{latency:.6f}", 0])


def plot_boxplot(path: Path, all_results: dict):
    schedule_order = [
        "constant_32",
        "constant_16",
        "constant_1",
        "increasing_1_to_32",
        "decreasing_32_to_1",
        "random_1_to_32",
        "random_pow2_set_1_2_4_8_16_32",
    ]
    labels = [
        "constant 32",
        "constant 16",
        "constant 1",
        "1 -> 32",
        "32 -> 1",
        "random [1,32]",
        "random {1,2,4,8,16,32}",
    ]
    data = [all_results[name]["measured_times_ms"] for name in schedule_order]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    box = ax.boxplot(
        data,
        patch_artist=True,
        showfliers=True,
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": "#333333",
            "markeredgecolor": "#333333",
            "alpha": 0.7,
        },
    )

    colors = [
        "#2E86AB",
        "#5D9CEC",
        "#A0D468",
        "#F6C85F",
        "#6F4E7C",
        "#9FD356",
        "#FF9F1C",
    ]
    for patch, c in zip(box["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Iteration latency (ms)", fontsize=11)
    ax.set_title(
        "Iteration Latency Distribution (64x64 -> 64x64, ~1.25M params, first 10 iters with backward)",
        fontsize=12,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description="Benchmark iteration latency for batch schedules")
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=1000)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("benchmark_iteration_latency_ablation_raw.csv"),
    )
    parser.add_argument(
        "--out-plot",
        type=Path,
        default=Path("fig_iteration_latency_ablation_boxplot.png"),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GenerativeNet64(channels=62).to(device)
    nparams = count_params(model)
    print(f"Device: {device}")
    print(f"Model parameters: {nparams} ({nparams / 1e6:.3f}M)")

    schedules = [
        "constant_32",
        "constant_16",
        "constant_1",
        "increasing_1_to_32",
        "decreasing_32_to_1",
        "random_1_to_32",
        "random_pow2_set_1_2_4_8_16_32",
    ]

    all_results = {}
    for schedule_name in schedules:
        batch_sizes_eval = build_eval_batch_schedule(
            schedule_name, args.eval_iters, args.max_batch, random.Random(args.seed)
        )
        measured_times_ms, full_schedule = run_schedule(
            model=model,
            schedule_name=schedule_name,
            batch_sizes_eval=batch_sizes_eval,
            warmup_iters=args.warmup_iters,
            device=device,
            seed=args.seed,
        )

        all_results[schedule_name] = {
            "measured_times_ms": measured_times_ms,
            "full_schedule": full_schedule,
        }

        arr = np.array(measured_times_ms, dtype=np.float64)
        print(
            f"{schedule_name}: n={len(arr)} mean={arr.mean():.3f}ms "
            f"p50={np.percentile(arr, 50):.3f}ms p95={np.percentile(arr, 95):.3f}ms"
        )

    save_raw_csv(args.out_csv, all_results, args.warmup_iters)
    plot_boxplot(args.out_plot, all_results)

    print(f"Saved raw timings to: {args.out_csv}")
    print(f"Saved box plot to: {args.out_plot}")


if __name__ == "__main__":
    main()
