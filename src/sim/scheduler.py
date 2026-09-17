import os
import sys
from typing import List, Dict, Tuple

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.sim.pass_engine import TileMetadata
from src.sim.recorder import OnboardBufferManager


class DownlinkScheduler:
    def __init__(self, data_rate_kbps: float = 1000.0, algorithm: str = "greedy"):
        """
        data_rate_kbps: Downlink transmission rate in kilobits per second (kbps).
        Note: 8 kilobits = 1 Kilobyte (KB).
        algorithm: 'greedy' or 'knapsack' / 'dp'
        """
        self.data_rate_kbps = float(data_rate_kbps)
        self.algorithm = algorithm.lower()

    def calculate_pass_capacity_kb(self, pass_duration_s: float) -> float:
        """
        Computes total bytes deliverable during a ground station pass.
        Capacity (KB) = (Rate (kbps) * Duration (s)) / 8.0
        """
        return (self.data_rate_kbps * pass_duration_s) / 8.0

    def select_tiles_greedy(
        self,
        buffer_manager: OnboardBufferManager,
        pass_time: float,
        pass_duration_s: float
    ) -> Tuple[List[str], float, float]:
        """
        Greedy selection using priority-to-size ratio (density).
        Returns: (selected_tile_ids, total_downlinked_kb, total_priority_value)
        """
        downlink_budget_kb = self.calculate_pass_capacity_kb(pass_duration_s)
        
        # Calculate current dynamic priorities
        candidates = []
        for t_id, tile in buffer_manager.buffer.items():
            p_score = tile.compute_priority(
                pass_time,
                alpha=buffer_manager.alpha,
                beta=buffer_manager.beta,
                decay_lambda=buffer_manager.decay_lambda
            )
            # Density = priority / size
            density = p_score / max(tile.compressed_size_kb, 1.0)
            candidates.append((t_id, p_score, tile.compressed_size_kb, density))

        # Sort by density descending
        candidates.sort(key=lambda x: x[3], reverse=True)

        selected_ids = []
        total_kb = 0.0
        total_value = 0.0

        for t_id, p_score, size_kb, _ in candidates:
            if total_kb + size_kb <= downlink_budget_kb:
                selected_ids.append(t_id)
                total_kb += size_kb
                total_value += p_score

        return selected_ids, total_kb, total_value

    def select_tiles_knapsack(
        self,
        buffer_manager: OnboardBufferManager,
        pass_time: float,
        pass_duration_s: float
    ) -> Tuple[List[str], float, float]:
        """
        Exact 0/1 Knapsack solution via dynamic programming.
        Discretizes capacity into 1 KB integer units.
        Returns: (selected_tile_ids, total_downlinked_kb, total_priority_value)
        """
        budget_kb = int(self.calculate_pass_capacity_kb(pass_duration_s))
        tiles = list(buffer_manager.buffer.values())
        n = len(tiles)

        if n == 0 or budget_kb <= 0:
            return [], 0.0, 0.0

        # Weights (sizes in rounded integer KB) and values
        weights = [max(1, int(round(t.compressed_size_kb))) for t in tiles]
        values = [
            t.compute_priority(
                pass_time,
                alpha=buffer_manager.alpha,
                beta=buffer_manager.beta,
                decay_lambda=buffer_manager.decay_lambda
            )
            for t in tiles
        ]

        # DP table: (n + 1) x (budget_kb + 1)
        dp = [[0.0] * (budget_kb + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            w = weights[i - 1]
            v = values[i - 1]
            for c in range(budget_kb + 1):
                if w <= c:
                    dp[i][c] = max(dp[i - 1][c], dp[i - 1][c - w] + v)
                else:
                    dp[i][c] = dp[i - 1][c]

        # Backtrack to identify selected tiles
        selected_ids = []
        curr_c = budget_kb
        for i in range(n, 0, -1):
            if dp[i][curr_c] != dp[i - 1][curr_c]:
                selected_ids.append(tiles[i - 1].tile_id)
                curr_c -= weights[i - 1]

        total_kb = sum(buffer_manager.buffer[t_id].compressed_size_kb for t_id in selected_ids)
        total_value = dp[n][budget_kb]

        return selected_ids, total_kb, total_value

    def select_tiles(
        self,
        buffer_manager: OnboardBufferManager,
        pass_time: float,
        pass_duration_s: float
    ) -> Tuple[List[str], float, float]:
        """
        Unified dispatch according to configured selection algorithm.
        """
        if self.algorithm in ("knapsack", "dp"):
            return self.select_tiles_knapsack(buffer_manager, pass_time, pass_duration_s)
        return self.select_tiles_greedy(buffer_manager, pass_time, pass_duration_s)

    def execute_downlink(
        self,
        buffer_manager: OnboardBufferManager,
        selected_tile_ids: List[str]
    ):
        """
        Purges successfully downlinked tiles from the onboard buffer.
        """
        for t_id in selected_tile_ids:
            if t_id in buffer_manager.buffer:
                buffer_manager.current_usage_kb -= buffer_manager.buffer[t_id].compressed_size_kb
                del buffer_manager.buffer[t_id]


if __name__ == "__main__":
    scheduler = DownlinkScheduler(data_rate_kbps=1000.0, algorithm="greedy")
    pass_duration = 2.0
    budget_kb = scheduler.calculate_pass_capacity_kb(pass_duration)

    manager = OnboardBufferManager(capacity_kb=1000.0)
    manager.ingest_tile(TileMetadata("tile_A", 0.0, 0.05, 90.0, 0.95), current_time=0.0)
    manager.ingest_tile(TileMetadata("tile_B", 10.0, 0.40, 85.0, 0.60), current_time=10.0)
    manager.ingest_tile(TileMetadata("tile_C", 50.0, 0.02, 95.0, 0.98), current_time=50.0)
    manager.ingest_tile(TileMetadata("tile_D", 80.0, 0.80, 70.0, 0.20), current_time=80.0)

    sim_time = 100.0
    print(f"Pass Contact Window: {pass_duration}s | Bandwidth Ceiling: {budget_kb:.1f} KB")

    selected_tiles, used_kb, total_val = scheduler.select_tiles(manager, sim_time, pass_duration)
    print(f"\n{scheduler.algorithm.upper()} Selection:")
    print(f" - Selected Tiles : {selected_tiles}")
    print(f" - Total Data     : {used_kb:.1f} / {budget_kb:.1f} KB")
    print(f" - Total Value    : {total_val:.6f}")

    scheduler.execute_downlink(manager, selected_tiles)
    print(f" - Post-Downlink Buffer Usage: {manager.current_usage_kb:.1f} KB")
    print(f" - Remaining Tiles in Buffer  : {list(manager.buffer.keys())}")