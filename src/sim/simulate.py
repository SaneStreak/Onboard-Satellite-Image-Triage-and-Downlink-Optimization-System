import os
import sys
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cv.dataset import Cloud38Dataset
from src.utils.metrics import estimate_compressed_size
from src.sim.pass_engine import TileMetadata, CloudInferenceEngine
from src.sim.recorder import OnboardBufferManager
from src.sim.scheduler import DownlinkScheduler


class BaselineFIFOBuffer:
    """Standard unmanaged satellite buffer: drops incoming data when full."""
    def __init__(self, capacity_kb: float):
        self.capacity_kb = float(capacity_kb)
        self.buffer: Dict[str, TileMetadata] = {}
        self.current_usage_kb: float = 0.0

    def ingest_tile(self, tile: TileMetadata) -> bool:
        if self.current_usage_kb + tile.compressed_size_kb <= self.capacity_kb:
            self.buffer[tile.tile_id] = tile
            self.current_usage_kb += tile.compressed_size_kb
            return True
        return False  # Drop tail on overflow

    def downlink_fifo(self, budget_kb: float) -> list:
        downlinked = []
        used_kb = 0.0
        keys_to_remove = []

        for t_id, tile in self.buffer.items():
            if used_kb + tile.compressed_size_kb <= budget_kb:
                downlinked.append(tile)
                used_kb += tile.compressed_size_kb
                keys_to_remove.append(t_id)
            else:
                break

        for t_id in keys_to_remove:
            self.current_usage_kb -= self.buffer[t_id].compressed_size_kb
            del self.buffer[t_id]

        return downlinked


def run_simulation(config_path: str = "tests/base_config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    np.random.seed(cfg["simulation"]["random_seed"])

    # Initialize subsystems
    engine = CloudInferenceEngine(cfg["paths"]["onnx_model"])
    triage_buffer = OnboardBufferManager(
        capacity_kb=cfg["storage"]["buffer_capacity_kb"],
        alpha=cfg["triage"]["alpha"],
        beta=cfg["triage"]["beta"],
        decay_lambda=cfg["triage"]["decay_lambda"]
    )
    baseline_buffer = BaselineFIFOBuffer(capacity_kb=cfg["storage"]["buffer_capacity_kb"])
    
    # Ingest test dataset
    dataset = Cloud38Dataset(cfg["paths"]["data_root"], cfg["paths"]["manifest_csv"])
    val_indices = list(range(int(0.8 * len(dataset)), len(dataset)))
    np.random.shuffle(val_indices)
    tile_cursor = 0

    # Telemetry logging
    triage_downlinked = []
    baseline_downlinked = []

    total_duration = cfg["simulation"]["total_duration_s"]
    dt = cfg["simulation"]["time_step_s"]
    capture_interval = cfg["payload"]["capture_interval_s"]
    orbit_period = cfg["orbit"]["orbital_period_s"]
    daylight_window = cfg["payload"]["daylight_duty_cycle_s"]
    passes = cfg["orbit"]["passes"]

    last_capture_time = -capture_interval

    print("=" * 60)
    print(f"Running LEO Triage Simulation: {total_duration}s ({total_duration / 3600:.1f}h)")
    print(f"Buffer Limit: {cfg['storage']['buffer_capacity_kb']:.1f} KB | Capture Step: {capture_interval}s")
    print("=" * 60)

    for current_time in range(0, total_duration, dt):
        in_daylight = (current_time % orbit_period) < daylight_window

        # 1. Optical Tile Acquisition
        if in_daylight and (current_time - last_capture_time >= capture_interval):
            last_capture_time = current_time
            idx = val_indices[tile_cursor % len(val_indices)]
            tile_cursor += 1

            sample = dataset[idx]
            tensor_np = sample["image"].numpy()  # (4, 384, 384)
            tile_id = f"patch_{idx}_{current_time}s"

            # Measure real compressed footprint s_i & inference cloud fraction
            cloud_frac = engine.predict_cloud_fraction(tensor_np, threshold=cfg["triage"]["cloud_threshold"])
            compressed_kb = estimate_compressed_size(sample["image"], quality=cfg["payload"]["webp_quality"]) / 1024.0
            base_util = 1.0 - cloud_frac

            tile = TileMetadata(
                tile_id=tile_id,
                capture_time=float(current_time),
                cloud_fraction=cloud_frac,
                compressed_size_kb=compressed_kb,
                base_utility=base_util
            )

            # Parallel ingestion
            triage_buffer.ingest_tile(tile, float(current_time))
            baseline_buffer.ingest_tile(tile)

        # 2. Ground Station Contact Passes
        for p in passes:
            if current_time == p["start_time_s"]:
                duration = p["duration_s"]
                scheduler = DownlinkScheduler(data_rate_kbps=p["data_rate_kbps"])
                budget_kb = scheduler.calculate_pass_capacity_kb(duration)

                print(f"\n[t={current_time:5d}s] ENTERING PASS: {p['contact_id']}")
                print(f" -> Pass Budget: {budget_kb:.1f} KB | Triage Usage: {triage_buffer.current_usage_kb:.1f} KB")

                # Triage Track: Knapsack Selection
                selected_ids, used_kb, val = scheduler.select_tiles_knapsack(triage_buffer, float(current_time), duration)
                for tid in selected_ids:
                    triage_downlinked.append(triage_buffer.buffer[tid])
                scheduler.execute_downlink(triage_buffer, selected_ids)

                # Baseline Track: FIFO Selection
                down_base = baseline_buffer.downlink_fifo(budget_kb)
                baseline_downlinked.extend(down_base)

                print(f" -> Triage downlinked {len(selected_ids)} tiles | Baseline downlinked {len(down_base)} tiles")

    # Final Telemetry Summary
    def summarize(tiles, name):
        if not tiles:
            return {"name": name, "count": 0, "mean_cloud": 0.0, "total_utility": 0.0, "data_mb": 0.0}
        clouds = [t.cloud_fraction for t in tiles]
        utils = [t.base_utility for t in tiles]
        data_kb = sum(t.compressed_size_kb for t in tiles)
        return {
            "name": name,
            "count": len(tiles),
            "mean_cloud": float(np.mean(clouds)),
            "total_utility": float(np.sum(utils)),
            "data_mb": data_kb / 1024.0
        }

    res_triage = summarize(triage_downlinked, "Triage (Active Eviction + Knapsack)")
    res_base = summarize(baseline_downlinked, "Baseline (Unmanaged FIFO)")

    print("\n" + "=" * 60)
    print("ORBITAL SIMULATION COMPLETE - PERFORMANCE BENCHMARK")
    print("=" * 60)
    print(f"{'Metric':<30} | {'Baseline (FIFO)':<18} | {'Triage Engine':<18}")
    print("-" * 72)
    print(f"{'Downlinked Tiles':<30} | {res_base['count']:<18d} | {res_triage['count']:<18d}")
    print(f"{'Downlinked Data (MB)':<30} | {res_base['data_mb']:<18.2f} | {res_triage['data_mb']:<18.2f}")
    print(f"{'Mean Cloud Coverage':<30} | {res_base['mean_cloud'] * 100:<17.1f}% | {res_triage['mean_cloud'] * 100:<17.1f}%")
    print(f"{'Cumulative Utility Yield':<30} | {res_base['total_utility']:<18.2f} | {res_triage['total_utility']:<18.2f}")
    print("=" * 60)


if __name__ == "__main__":
    run_simulation()