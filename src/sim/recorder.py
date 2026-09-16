import os
import sys
from typing import List, Dict, Tuple, Optional

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.sim.pass_engine import TileMetadata


class OnboardBufferManager:
    def __init__(
        self,
        capacity_kb: float,
        alpha: float = 1.0,
        beta: float = 0.5,
        decay_lambda: float = 1e-4
    ):
        self.capacity_kb = float(capacity_kb)
        self.alpha = alpha
        self.beta = beta
        self.decay_lambda = decay_lambda

        # In-memory storage: tile_id -> TileMetadata
        self.buffer: Dict[str, TileMetadata] = {}
        self.current_usage_kb: float = 0.0

        # Telemetry audit counters
        self.admitted_count: int = 0
        self.rejected_count: int = 0
        self.evicted_count: int = 0

    def get_ranked_tiles(self, current_time: float) -> List[Tuple[str, float]]:
        """
        Returns buffered tiles sorted by priority descending: [(tile_id, priority), ...]
        """
        scored = [
            (
                t_id,
                tile.compute_priority(
                    current_time,
                    alpha=self.alpha,
                    beta=self.beta,
                    decay_lambda=self.decay_lambda
                )
            )
            for t_id, tile in self.buffer.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def ingest_tile(self, tile: TileMetadata, current_time: float) -> bool:
        """
        Attempts to store an incoming tile.
        Returns True if admitted (with or without evicting older tiles), False if rejected.
        """
        tile_size = tile.compressed_size_kb

        # Edge case: Single tile exceeds total satellite capacity
        if tile_size > self.capacity_kb:
            self.rejected_count += 1
            return False

        # If sufficient room exists, admit immediately
        if self.current_usage_kb + tile_size <= self.capacity_kb:
            self.buffer[tile.tile_id] = tile
            self.current_usage_kb += tile_size
            self.admitted_count += 1
            return True

        # Buffer full: evaluate dynamic priority against current contents
        candidate_priority = tile.compute_priority(
            current_time,
            alpha=self.alpha,
            beta=self.beta,
            decay_lambda=self.decay_lambda
        )

        ranked = self.get_ranked_tiles(current_time)
        
        # Check eviction viability: simulate dropping lowest-priority items
        eviction_candidates = []
        reclaimed_kb = 0.0
        space_needed = (self.current_usage_kb + tile_size) - self.capacity_kb

        # Iterate from lowest priority to highest
        for t_id, p_score in reversed(ranked):
            if candidate_priority <= p_score:
                # Candidate is lower priority than remaining items; cannot justify further evictions
                break

            eviction_candidates.append(t_id)
            reclaimed_kb += self.buffer[t_id].compressed_size_kb

            if reclaimed_kb >= space_needed:
                break

        if reclaimed_kb < space_needed:
            # Candidate tile does not possess enough relative value to displace existing tiles
            self.rejected_count += 1
            return False

        # Execute evictions
        for t_id in eviction_candidates:
            self.current_usage_kb -= self.buffer[t_id].compressed_size_kb
            del self.buffer[t_id]
            self.evicted_count += 1

        # Admit candidate
        self.buffer[tile.tile_id] = tile
        self.current_usage_kb += tile_size
        self.admitted_count += 1
        return True


if __name__ == "__main__":
    # Smoke test: Initialize a small 250 KB onboard buffer (~2-3 tiles capacity)
    manager = OnboardBufferManager(capacity_kb=250.0, alpha=1.0, beta=0.5, decay_lambda=1e-3)

    # Ingest Tile 1 (High clarity, captured at t=0)
    t1 = TileMetadata("tile_001", capture_time=0.0, cloud_fraction=0.1, compressed_size_kb=90.0, base_utility=0.9)
    manager.ingest_tile(t1, current_time=0.0)

    # Ingest Tile 2 (Moderate clarity, captured at t=10)
    t2 = TileMetadata("tile_002", capture_time=10.0, cloud_fraction=0.3, compressed_size_kb=90.0, base_utility=0.7)
    manager.ingest_tile(t2, current_time=10.0)

    # Ingest Tile 3 (High clarity, captured at t=20) -> Reaches ~270 KB, triggers eviction evaluation
    t3 = TileMetadata("tile_003", capture_time=20.0, cloud_fraction=0.05, compressed_size_kb=90.0, base_utility=0.95)
    admitted = manager.ingest_tile(t3, current_time=20.0)

    print("Onboard Buffer Manager Operational:")
    print(f" - Buffer Capacity  : {manager.capacity_kb:.1f} KB")
    print(f" - Current Usage    : {manager.current_usage_kb:.1f} KB")
    print(f" - Stored Tiles     : {list(manager.buffer.keys())}")
    print(f" - Admitted Count   : {manager.admitted_count}")
    print(f" - Evicted Count    : {manager.evicted_count}")
    print(f" - Rejected Count   : {manager.rejected_count}")