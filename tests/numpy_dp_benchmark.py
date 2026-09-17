import os
import sys
import time
import tracemalloc
from pathlib import Path
import numpy as np

# Robust root resolution: finds the directory containing 'src'
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent if (FILE_PATH.parent / "src").exists() else FILE_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sim.benchmark_scheduler import generate_synthetic_pool


def benchmark_numpy_vectorized_dp(n: int = 2500, capacity_kb: int = 5120):
    tiles = generate_synthetic_pool(n, seed=42 + n)
    W = int(capacity_kb)

    weights = [max(1, int(np.ceil(t.compressed_size_kb))) for t in tiles]
    values = [float((1.0 - t.cloud_fraction) ** 1.5) for t in tiles]

    tracemalloc.start()
    t0 = time.perf_counter()

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

    # Backtracking selected items
    selected_ids = []
    curr_w = W
    for i in range(n - 1, -1, -1):
        if keep[i, curr_w]:
            selected_ids.append(tiles[i].tile_id)
            curr_w -= weights[i]

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"NumPy Vectorized DP (N={n}, W={W}):")
    print(f" - Execution Time : {elapsed_ms:.2f} ms")
    print(f" - Peak Allocation : {peak_mem / (1024 * 1024):.2f} MB")
    print(f" - Total Utility   : {dp[W]:.4f}")
    print(f" - Items Selected  : {len(selected_ids)}")


if __name__ == "__main__":
    benchmark_numpy_vectorized_dp()