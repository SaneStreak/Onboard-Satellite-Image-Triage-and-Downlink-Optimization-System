import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cv.dataset import Cloud38Dataset
from src.utils.metrics import estimate_compressed_size
from src.sim.pass_engine import TileMetadata, CloudInferenceEngine
from src.sim.recorder import OnboardBufferManager
from src.sim.scheduler import DownlinkScheduler
from src.sim.simulate import BaselineFIFOBuffer


def run_tracked_simulation(cfg: dict, alpha: float = None, decay_lambda: float = None):
    """Runs a single simulation while recording continuous time-series telemetry."""
    np.random.seed(cfg["simulation"]["random_seed"])

    use_alpha = alpha if alpha is not None else cfg["triage"]["alpha"]
    use_lambda = decay_lambda if decay_lambda is not None else cfg["triage"]["decay_lambda"]

    engine = CloudInferenceEngine(cfg["paths"]["onnx_model"])
    triage_buf = OnboardBufferManager(
        capacity_kb=cfg["storage"]["buffer_capacity_kb"],
        alpha=use_alpha,
        beta=cfg["triage"]["beta"],
        decay_lambda=use_lambda
    )
    baseline_buf = BaselineFIFOBuffer(capacity_kb=cfg["storage"]["buffer_capacity_kb"])

    dataset = Cloud38Dataset(cfg["paths"]["data_root"], cfg["paths"]["manifest_csv"])
    val_indices = list(range(int(0.8 * len(dataset)), len(dataset)))
    np.random.shuffle(val_indices)
    tile_cursor = 0

    total_duration = cfg["simulation"]["total_duration_s"]
    dt = cfg["simulation"]["time_step_s"]
    capture_interval = cfg["payload"]["capture_interval_s"]
    orbit_period = cfg["orbit"]["orbital_period_s"]
    daylight_window = cfg["payload"]["daylight_duty_cycle_s"]
    passes = cfg["orbit"]["passes"]

    last_capture_time = -capture_interval

    # Time-series telemetry logs
    history = {
        "time": [],
        "triage_usage_kb": [],
        "baseline_usage_kb": [],
        "triage_cum_util": [],
        "baseline_cum_util": []
    }

    triage_downlinked = []
    baseline_downlinked = []
    cum_util_triage = 0.0
    cum_util_base = 0.0

    for current_time in range(0, total_duration, dt):
        in_daylight = (current_time % orbit_period) < daylight_window

        # Optical Capture
        if in_daylight and (current_time - last_capture_time >= capture_interval):
            last_capture_time = current_time
            idx = val_indices[tile_cursor % len(val_indices)]
            tile_cursor += 1

            sample = dataset[idx]
            tensor_np = sample["image"].numpy()
            tile_id = f"patch_{idx}_{current_time}s"

            cloud_frac = engine.predict_cloud_fraction(tensor_np, threshold=cfg["triage"]["cloud_threshold"])
            compressed_kb = estimate_compressed_size(sample["image"], quality=cfg["payload"]["webp_quality"]) / 1024.0

            tile = TileMetadata(
                tile_id=tile_id,
                capture_time=float(current_time),
                cloud_fraction=cloud_frac,
                compressed_size_kb=compressed_kb,
                base_utility=1.0 - cloud_frac
            )

            triage_buf.ingest_tile(tile, float(current_time))
            baseline_buf.ingest_tile(tile)

        # Ground Station Passes
        for p in passes:
            if current_time == p["start_time_s"]:
                duration = p["duration_s"]
                scheduler = DownlinkScheduler(data_rate_kbps=p["data_rate_kbps"])
                budget_kb = scheduler.calculate_pass_capacity_kb(duration)

                # Triage downlink
                sel_ids, _, _ = scheduler.select_tiles_knapsack(triage_buf, float(current_time), duration)
                for tid in sel_ids:
                    tile_obj = triage_buf.buffer[tid]
                    triage_downlinked.append(tile_obj)
                    cum_util_triage += tile_obj.base_utility
                scheduler.execute_downlink(triage_buf, sel_ids)

                # Baseline downlink
                down_base = baseline_buf.downlink_fifo(budget_kb)
                for b_tile in down_base:
                    baseline_downlinked.append(b_tile)
                    cum_util_base += b_tile.base_utility

        # Log continuous state
        history["time"].append(current_time)
        history["triage_usage_kb"].append(triage_buf.current_usage_kb)
        history["baseline_usage_kb"].append(baseline_buf.current_usage_kb)
        history["triage_cum_util"].append(cum_util_triage)
        history["baseline_cum_util"].append(cum_util_base)

    summary = {
        "alpha": use_alpha,
        "lambda": use_lambda,
        "triage_count": len(triage_downlinked),
        "baseline_count": len(baseline_downlinked),
        "triage_cloud": float(np.mean([t.cloud_fraction for t in triage_downlinked])) if triage_downlinked else 0.0,
        "baseline_cloud": float(np.mean([t.cloud_fraction for t in baseline_downlinked])) if baseline_downlinked else 0.0,
        "triage_util": cum_util_triage,
        "baseline_util": cum_util_base
    }

    return history, summary


def generate_telemetry_plots(history: dict, cfg: dict, save_path: str):
    """Renders buffer occupancy and cumulative delivered utility curves."""
    t_axis = np.array(history["time"]) / 60.0  # Minutes
    cap_limit = cfg["storage"]["buffer_capacity_kb"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    plt.subplots_adjust(hspace=0.25)

    # Top Panel: Buffer Capacity Fill State
    ax1.plot(t_axis, history["triage_usage_kb"], label="Triage Managed Buffer", color="#1f77b4", lw=1.8)
    ax1.plot(t_axis, history["baseline_usage_kb"], label="FIFO Buffer", color="#ff7f0e", linestyle="--", lw=1.8)
    ax1.axhline(cap_limit, color="black", linestyle=":", label=f"Buffer Ceiling ({cap_limit:.0f} KB)")

    # Ground Pass Contact Windows
    for p in cfg["orbit"]["passes"]:
        p_start = p["start_time_s"] / 60.0
        p_end = (p["start_time_s"] + p["duration_s"]) / 60.0
        ax1.axvspan(p_start, p_end, color="green", alpha=0.2, label="Ground Station LOS" if p == cfg["orbit"]["passes"][0] else "")
        ax2.axvspan(p_start, p_end, color="green", alpha=0.2)

    ax1.set_title("Onboard Memory Occupancy & Downlink Drain Profile", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Buffer Storage (KB)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right")

    # Bottom Panel: Cumulative Utility Delivered
    ax2.plot(t_axis, history["triage_cum_util"], label="Triage Delivered Utility", color="#1f77b4", lw=2.0)
    ax2.plot(t_axis, history["baseline_cum_util"], label="FIFO Delivered Utility", color="#ff7f0e", linestyle="--", lw=2.0)

    ax2.set_title("Cumulative Usable Scientific/Tactical Utility ($U_i = 1 - \text{Cloud}$)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Elapsed Orbit Time (minutes)", fontsize=10)
    ax2.set_ylabel("Cumulative Utility ($U$)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Simulation telemetry visualization saved to: {save_path}")


def run_parametric_sweep(cfg: dict):
    """Executes a 2D parameter grid across alpha and lambda."""
    alphas = [0.5, 1.0, 1.5, 2.0]
    lambdas = [0.0, 1e-4, 5e-4]

    print("\n" + "=" * 76)
    print("PARAMETRIC SWEEP: Cloud Penalty (alpha) vs Latency Decay (lambda)")
    print("=" * 76)
    print(f"{'Alpha':<8} | {'Lambda':<10} | {'Triage Util':<13} | {'Mean Cloud %':<14} | {'Util Gain %':<12}")
    print("-" * 76)

    records = []
    for a in alphas:
        for lam in lambdas:
            _, summary = run_tracked_simulation(cfg, alpha=a, decay_lambda=lam)
            gain = ((summary["triage_util"] - summary["baseline_util"]) / max(summary["baseline_util"], 1e-3)) * 100.0
            records.append(summary)
            print(f"{a:<8.1f} | {lam:<10.5f} | {summary['triage_util']:<13.2f} | {summary['triage_cloud'] * 100:<13.1f}% | +{gain:<10.1f}%")

    print("=" * 76)


if __name__ == "__main__":
    config_file = "tests/base_config_2.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # 1. Generate high-resolution telemetry plot under base_config_2 conditions
    hist, _ = run_tracked_simulation(config)
    output_plot = os.path.join("outputs", "sim_runs", "simulation_telemetry.png")
    generate_telemetry_plots(hist, config, output_plot)

    # 2. Run parametric trade-off sweep
    run_parametric_sweep(config)