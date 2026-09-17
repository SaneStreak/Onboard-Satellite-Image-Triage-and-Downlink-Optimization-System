import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from src.sim.pass_engine import PriorityEngine, TileMetadata

# Maintain TilePayload as an alias for TileMetadata
TilePayload = TileMetadata


class OnboardBuffer:
    def __init__(self, capacity_kb: float, priority_engine: PriorityEngine):
        self.capacity_kb = float(capacity_kb)
        self.engine = priority_engine
        self.storage: Dict[str, TileMetadata] = {}
        self.current_usage_kb: float = 0.0

    @property
    def buffer(self) -> Dict[str, TileMetadata]:
        """Provides compatibility with scheduler access."""
        return self.storage

    @property
    def alpha(self) -> float:
        return self.engine.alpha

    @property
    def beta(self) -> float:
        return self.engine.beta

    @property
    def decay_lambda(self) -> float:
        return self.engine.decay_lambda

    def get_buffered_bboxes(self, exclude_id: Optional[str] = None) -> List[Tuple[float, float, float, float]]:
        return [
            t.bbox for tid, t in self.storage.items()
            if exclude_id is None or tid != exclude_id
        ]

    def get_current_priorities(self, current_time_s: float) -> Dict[str, float]:
        priorities = {}
        all_bboxes = self.get_buffered_bboxes()

        for tid, tile in self.storage.items():
            other_bboxes = [b for b in all_bboxes if b != tile.bbox]
            dt = current_time_s - tile.capture_time
            p = self.engine.calculate_priority(
                cloud_fraction=tile.cloud_fraction,
                compressed_size_kb=tile.compressed_size_kb,
                elapsed_time_s=dt,
                candidate_bbox=tile.bbox,
                buffered_bboxes=other_bboxes
            )
            priorities[tid] = p
        return priorities

    def admit_tile(self, tile: TileMetadata, current_time_s: float) -> bool:
        """
        Attempts to admit a new tile. If memory overflows, evicts the lowest marginal
        priority tile, provided the new tile provides higher marginal value.
        """
        if tile.compressed_size_kb > self.capacity_kb:
            return False

        existing_bboxes = self.get_buffered_bboxes()
        candidate_priority = self.engine.calculate_priority(
            cloud_fraction=tile.cloud_fraction,
            compressed_size_kb=tile.compressed_size_kb,
            elapsed_time_s=0.0,
            candidate_bbox=tile.bbox,
            buffered_bboxes=existing_bboxes
        )

        needed_kb = (self.current_usage_kb + tile.compressed_size_kb) - self.capacity_kb

        if needed_kb <= 0.0:
            self.storage[tile.tile_id] = tile
            self.current_usage_kb += tile.compressed_size_kb
            return True

        eviction_candidates = []
        priorities = self.get_current_priorities(current_time_s)
        sorted_tiles = sorted(self.storage.keys(), key=lambda k: priorities[k])

        freed_kb = 0.0
        for victim_id in sorted_tiles:
            if priorities[victim_id] >= candidate_priority:
                return False

            eviction_candidates.append(victim_id)
            freed_kb += self.storage[victim_id].compressed_size_kb
            if freed_kb >= needed_kb:
                break

        if freed_kb < needed_kb:
            return False

        for victim_id in eviction_candidates:
            self.current_usage_kb -= self.storage[victim_id].compressed_size_kb
            del self.storage[victim_id]

        self.storage[tile.tile_id] = tile
        self.current_usage_kb += tile.compressed_size_kb
        return True

    def ingest_tile(self, tile: TileMetadata, current_time: float) -> bool:
        """Alias for admit_tile to maintain simulate.py compatibility."""
        return self.admit_tile(tile, current_time)

    def remove_tiles(self, tile_ids: List[str]) -> None:
        """Removes downlinked tiles from storage."""
        for tid in tile_ids:
            if tid in self.storage:
                self.current_usage_kb -= self.storage[tid].compressed_size_kb
                del self.storage[tid]


class OnboardBufferManager(OnboardBuffer):
    """Convenience wrapper accepting individual priority hyperparameters."""
    def __init__(
        self,
        capacity_kb: float,
        alpha: float = 1.5,
        beta: float = 0.25,
        decay_lambda: float = 0.0001,
        gamma_overlap: float = 0.8
    ):
        engine = PriorityEngine(
            alpha=alpha,
            beta=beta,
            decay_lambda=decay_lambda,
            gamma_overlap=gamma_overlap
        )
        super().__init__(capacity_kb=capacity_kb, priority_engine=engine)