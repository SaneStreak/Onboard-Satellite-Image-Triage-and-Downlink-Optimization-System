import os
import sys
from pathlib import Path
import numpy as np

# Robust root resolution: finds the directory containing 'src'
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent if (FILE_PATH.parent / "src").exists() else FILE_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sim.benchmark_scheduler import generate_synthetic_pool, solve_01_knapsack_dp


def solve_01_knapsack_numpy(tiles, capacity_kb: int):
    n = len(tiles)
    W = int(capacity_kb)

    weights = [max(1, int(np.ceil(t.compressed_size_kb))) for t in tiles]
    values = [float((1.0 - t.cloud_fraction) ** 1.5) for t in tiles]

    keep = np.zeros((n, W + 1), dtype=bool)
    dp = np.zeros(W + 1, dtype=np.float64)

    for i in range(n):
        w_i = weights[i]
        v_i = values[i]

        prev_dp = dp[:W + 1 - w_i].copy()
        candidates = prev_dp + v_i
        target_slice = dp[w_i:]
        mask = candidates > target_slice

        target_slice[mask] = candidates[mask]
        keep[i, w_i:][mask] = True

    selected_ids = []
    curr_w = W
    for i in range(n - 1, -1, -1):
        if keep[i, curr_w]:
            selected_ids.append(tiles[i].tile_id)
            curr_w -= weights[i]

    return dp[W], selected_ids


def test_dp_vectorized_equivalence():
    W = 5120
    for n in [25, 100, 400, 1600]:
        tiles = generate_synthetic_pool(n, seed=42 + n)

        val_py, ids_py = solve_01_knapsack_dp(tiles, W)
        val_np, ids_np = solve_01_knapsack_numpy(tiles, W)

        val_match = np.isclose(val_py, val_np, atol=1e-5)
        print(f"Pool N={n:<4} | Python DP: {val_py:.4f} | NumPy DP: {val_np:.4f} | Match: {val_match}")
        assert val_match, f"Discrepancy detected at N={n}: {val_py} != {val_np}"

    print("\nVerification Passed: Vectorized DP is numerically identical to reference DP across all N.")


if __name__ == "__main__":
    verify_equivalence()