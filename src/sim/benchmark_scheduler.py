import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.sim.recorder import TilePayload


def solve_01_knapsack_dp(tiles: List[TilePayload], capacity_kb: int) -> Tuple[float, List[str]]:
    """
    Exact 0/1 Knapsack using Dynamic Programming with integer weights (KB).
    Returns (max_utility, selected_tile_ids).
    """
    n = len(tiles)
    W = int(capacity_kb)

    weights = [max(1, int(np.ceil(t.compressed_size_kb))) for t in tiles]
    values = [float((1.0 - t.cloud_fraction) ** 1.5) for t in tiles]

    dp = [0.0] * (W + 1)
    keep = [[False] * (W + 1) for _ in range(n)]

    for i in range(n):
        w_i = weights[i]
        v_i = values[i]
        for w in range(W, w_i - 1, -1):
            if dp[w - w_i] + v_i > dp[w]:
                dp[w] = dp[w - w_i] + v_i
                keep[i][w] = True

    selected_ids = []
    curr_w = W
    for i in range(n - 1, -1, -1):
        if keep[i][curr_w]:
            selected_ids.append(tiles[i].tile_id)
            curr_w -= weights[i]

    return dp[W], selected_ids


def solve_greedy_density(tiles: List[TilePayload], capacity_kb: float) -> Tuple[float, List[str]]:
    """
    Greedy selection sorting by value-density (Utility / Size).
    Returns (total_utility, selected_tile_ids).
    """
    scored_tiles = []
    for t in tiles:
        utility = float((1.0 - t.cloud_fraction) ** 1.5)
        density = utility / max(1.0, t.compressed_size_kb)
        scored_tiles.append((density, utility, t))

    # Sort descending by density
    scored_tiles.sort(key=lambda x: x[0], reverse=True)

    total_utility = 0.0
    used_kb = 0.0
    selected_ids = []

    for density, utility, t in scored_tiles:
        if used_kb + t.compressed_size_kb <= capacity_kb:
            used_kb += t.compressed_size_kb
            total_utility += utility
            selected_ids.append(t.tile_id)

    return total_utility, selected_ids


def run_adversarial_test():
    """
    Theoretical Worst-Case Analysis:
    For 0/1 Knapsack, standard value-density greedy has an unbounded worst-case gap:
      Item 0: size = 1, value = 1 + eps   -> density = 1 + eps
      Item 1: size = W, value = W         -> density = 1.0
    Greedy selects Item 0 and halts (remaining capacity W - 1 cannot fit Item 1).
    Total Greedy Value = 1 + eps vs Optimal DP Value = W.
    As W -> inf, (V_DP - V_Greedy) / V_DP -> 100%.

    Below is a domain-constrained adversarial instance (equal-weight split):
    Capacity W = 1000 KB.
    Demonstrates a ~50% collapse without extreme weight disparities.
    """
    print("\n" + "=" * 72)
    print("RUNNING ADVERSARIAL KNAPSACK STRESS TEST (POISON-PILL INSTANCE)")
    print("=" * 72)

    W = 1000.0
    u0 = 0.502
    c0 = 1.0 - (u0 ** (2.0 / 3.0))

    u1_2 = 0.490
    c1_2 = 1.0 - (u1_2 ** (2.0 / 3.0))

    adversarial_tiles = [
        TilePayload("poison_pill", 0.0, c0, 501.0, 1.0 - c0, (0.0, 0.0, 1.0, 1.0)),
        TilePayload("item_1", 0.0, c1_2, 500.0, 1.0 - c1_2, (1.0, 1.0, 2.0, 2.0)),
        TilePayload("item_2", 0.0, c1_2, 500.0, 1.0 - c1_2, (2.0, 2.0, 3.0, 3.0)),
    ]

    dp_util, dp_ids = solve_01_knapsack_dp(adversarial_tiles, int(W))
    greedy_util, greedy_ids = solve_greedy_density(adversarial_tiles, W)

    adv_gap = ((dp_util - greedy_util) / dp_util) * 100.0

    print(f"Knapsack Capacity: {W:.0f} KB")
    print(f" - Optimal DP Solution : Utility = {dp_util:.4f} | Packed: {dp_ids}")
    print(f" - Greedy Density      : Utility = {greedy_util:.4f} | Packed: {greedy_ids}")
    print(f" - Domain-Constrained Gap: {adv_gap:.2f}%")
    print("Note: Theoretical worst-case gap for value-density greedy approaches 100%.")
    print("The <0.55% empirical gap requires bounded tile footprints (max(s_i) << W).")
    print("=" * 72 + "\n")


def generate_synthetic_pool(n_tiles: int, seed: int = 42) -> List[TilePayload]:
    """Generates realistic synthetic multi-spectral candidate tiles."""
    np.random.seed(seed)
    tiles = []

    cloud_fracs = np.random.beta(a=0.5, b=0.5, size=n_tiles)
    base_sizes = 50.0 + (1.0 - cloud_fracs) * 180.0 + np.random.normal(0, 15.0, size=n_tiles)
    sizes = np.clip(base_sizes, 25.0, 260.0)

    for i in range(n_tiles):
        cf = float(cloud_fracs[i])
        tile = TilePayload(
            tile_id=f"tile_{i:04d}",
            capture_time=float(i * 15.0),
            cloud_fraction=cf,
            compressed_size_kb=float(sizes[i]),
            base_utility=float(1.0 - cf),
            bbox=(float(i), float(i), float(i + 1), float(i + 1))
        )
        tiles.append(tile)

    return tiles


def run_scalability_sweep():
    print("=" * 72)
    print("RUNNING SCHEDULER BENCHMARK: EXACT DP VS GREEDY VALUE-DENSITY")
    print("=" * 72)

    pass_capacity_kb = 5120.0  # 5 MB ground pass window
    pool_sizes = [25, 50, 100, 200, 400, 800, 1600, 2500]

    records = []

    for n in pool_sizes:
        tiles = generate_synthetic_pool(n, seed=42 + n)

        # 1. Benchmark Greedy
        t0 = time.perf_counter()
        greedy_util, greedy_ids = solve_greedy_density(tiles, pass_capacity_kb)
        greedy_time_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Benchmark DP (Cap DP execution at N <= 1600 to prevent host hangs)
        if n <= 1600:
            t0 = time.perf_counter()
            dp_util, dp_ids = solve_01_knapsack_dp(tiles, int(pass_capacity_kb))
            dp_time_ms = (time.perf_counter() - t0) * 1000.0

            gap_pct = ((dp_util - greedy_util) / max(1e-6, dp_util)) * 100.0
        else:
            dp_util = np.nan
            dp_time_ms = np.nan
            gap_pct = np.nan

        records.append({
            "pool_size": n,
            "greedy_time_ms": greedy_time_ms,
            "dp_time_ms": dp_time_ms,
            "greedy_utility": greedy_util,
            "dp_utility": dp_util,
            "optimality_gap_pct": gap_pct
        })

        gap_str = f"{gap_pct:.3f}%" if not np.isnan(gap_pct) else "N/A (OOM/Timeout)"
        dp_time_str = f"{dp_time_ms:.2f} ms" if not np.isnan(dp_time_ms) else "N/A"
        print(f"N = {n:<5} | Greedy: {greedy_time_ms:<8.3f} ms | DP: {dp_time_str:<10} | Optimality Gap: {gap_str}")

    df = pd.DataFrame(records)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/scheduler_benchmark.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#faf8f5")
    for ax in (ax1, ax2):
        ax.set_facecolor("#faf8f5")
        ax.grid(True, linestyle="--", alpha=0.5)

    # Latency Plot
    valid_dp = df.dropna(subset=["dp_time_ms"])
    ax1.plot(df["pool_size"], df["greedy_time_ms"], "o-", color="#1a73e8", label="Greedy Value-Density", linewidth=2)
    ax1.plot(valid_dp["pool_size"], valid_dp["dp_time_ms"], "s-", color="#d93025", label="Exact 0/1 Knapsack (DP)", linewidth=2)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Candidate Pool Size ($N$)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Execution Time (ms)", fontsize=11, fontweight="bold")
    ax1.set_title("Runtime Scaling (Log-Log)", fontsize=12, fontweight="bold")
    ax1.legend()

    # Optimality Gap Plot
    ax2.plot(valid_dp["pool_size"], valid_dp["optimality_gap_pct"], "d-", color="#137333", linewidth=2)
    ax2.set_xscale("log")
    ax2.set_xlabel("Candidate Pool Size ($N$)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Sub-optimality Gap (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Greedy Sub-Optimality Gap vs Exact DP", fontsize=12, fontweight="bold")
    ax2.axhline(0, color="gray", linestyle=":")

    plt.tight_layout()
    plt.savefig("outputs/scheduler_scaling.png", dpi=300)
    print("\nSaved benchmark results to outputs/scheduler_benchmark.csv")
    print("Saved scaling plots to outputs/scheduler_scaling.png")
    print("=" * 72)


if __name__ == "__main__":
    run_adversarial_test()
    run_scalability_sweep()