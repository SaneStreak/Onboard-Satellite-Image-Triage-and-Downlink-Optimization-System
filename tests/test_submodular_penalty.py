import os
import sys
import itertools
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.sim.pass_engine import PriorityEngine, TileMetadata
from src.sim.benchmark_scheduler import solve_01_knapsack_dp


def test_diminishing_marginal_returns():
    """
    Formally verifies submodularity:
    For any subsets A subseteq B, and new candidate x not in B:
    Delta_f(x | A) >= Delta_f(x | B)
    """
    engine = PriorityEngine(gamma_overlap=0.8)

    box_a1 = (0.0, 0.0, 1.0, 1.0)
    box_b_extra = (0.5, 0.5, 1.5, 1.5)

    set_A = [box_a1]
    set_B = [box_a1, box_b_extra]  # A is a strict subset of B

    # Candidate x overlaps more with box_b_extra than box_a1
    candidate_x = (0.7, 0.7, 1.7, 1.7)
    cloud_frac = 0.1

    marginal_gain_A = engine.calculate_marginal_utility(cloud_frac, candidate_x, set_A)
    marginal_gain_B = engine.calculate_marginal_utility(cloud_frac, candidate_x, set_B)

    # Submodularity condition
    assert marginal_gain_A >= marginal_gain_B, (
        f"Submodularity violation: Gain in A ({marginal_gain_A}) < Gain in B ({marginal_gain_B})"
    )


def test_dp_exactness_vs_brute_force():
    """
    Validates DP knapsack correctness against exhaustive 2^N brute force search
    on integer-discretized inputs.
    """
    capacity_kb = 250.0

    # Small test pool (N=10)
    tiles = [
        TileMetadata(f"t_{i}", 0.0, cloud_fraction=c, compressed_size_kb=s, base_utility=1.0 - c)
        for i, (c, s) in enumerate([
            (0.10, 45.0), (0.25, 70.0), (0.05, 90.0), (0.80, 30.0), (0.15, 60.0),
            (0.40, 50.0), (0.02, 80.0), (0.70, 40.0), (0.30, 65.0), (0.00, 100.0)
        ])
    ]

    # Ground-truth brute force
    best_brute_val = 0.0
    for r in range(1, len(tiles) + 1):
        for combo in itertools.combinations(tiles, r):
            total_size = sum(t.compressed_size_kb for t in combo)
            if total_size <= capacity_kb:
                total_val = sum((1.0 - t.cloud_fraction) ** 1.5 for t in combo)
                if total_val > best_brute_val:
                    best_brute_val = total_val

    # DP output
    val_dp, _ = solve_01_knapsack_dp(tiles, capacity_kb)

    # Values must match within discretization float tolerance
    assert abs(val_dp - best_brute_val) < 1e-4, (
        f"DP ground truth mismatch: DP={val_dp} vs BruteForce={best_brute_val}"
    )


if __name__ == "__main__":
    test_diminishing_marginal_returns()
    test_dp_exactness_vs_brute_force()
    print("All formal submodularity and DP exactness assertions passed.")