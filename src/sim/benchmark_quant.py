import os
import sys
import time
import yaml
import numpy as np
import onnxruntime as ort
from typing import Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cv.dataset import Cloud38Dataset


def compute_numpy_iou(pred_mask: np.ndarray, true_mask: np.ndarray, smooth: float = 1e-6) -> float:
    """Calculates Intersection over Union directly on NumPy binary masks."""
    pred_b = pred_mask.astype(bool)
    true_b = true_mask.astype(bool)
    intersection = np.logical_and(pred_b, true_b).sum()
    union = np.logical_or(pred_b, true_b).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float((intersection + smooth) / (union + smooth))


def create_session(model_path: str, intra_threads: int = 1) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra_threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def evaluate_model(
    session: ort.InferenceSession,
    dataset: Cloud38Dataset,
    indices: list,
    threshold: float = 0.5
) -> Tuple[float, float, float]:
    """Evaluates mIoU, Mean Cloud Fraction, and average latency per tile in ms."""
    ious = []
    cloud_fractions = []
    latencies = []

    input_name = session.get_inputs()[0].name

    for idx in indices:
        sample = dataset[idx]
        image_np = sample["image"].unsqueeze(0).numpy().astype(np.float32)
        true_mask = sample["mask"].squeeze(0).numpy().astype(np.uint8)

        t0 = time.perf_counter()
        logits = session.run(None, {input_name: image_np})[0]
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

        probs = sigmoid(logits).squeeze()
        pred_mask = (probs >= threshold).astype(np.uint8)

        iou = compute_numpy_iou(pred_mask, true_mask)
        ious.append(iou)
        cloud_fractions.append(float(np.mean(pred_mask)))

    mean_iou = float(np.mean(ious))
    mean_cloud = float(np.mean(cloud_fractions))
    mean_lat = float(np.mean(latencies[5:]))  # Drop warmup runs

    return mean_iou, mean_cloud, mean_lat


def run_benchmark(
    config_path: str = "tests/base_config.yaml",
    int8_path: str = "checkpoints/tiny_cloud_unet_int8.onnx",
    num_eval_samples: int = 150
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    fp32_path = cfg["paths"]["onnx_model"]
    data_root = cfg["paths"]["data_root"]
    manifest_csv = cfg["paths"]["manifest_csv"]
    threshold = cfg["triage"]["cloud_threshold"]
    capture_interval_s = cfg["payload"]["capture_interval_s"]

    print("=" * 68)
    print("RUNNING FP32 VS INT8 QUANTIZATION AUDIT & CPU LATENCY PROFILING")
    print("=" * 68)
    print(f"FP32 Checkpoint : {fp32_path}")
    print(f"INT8 Checkpoint : {int8_path}")

    dataset = Cloud38Dataset(data_root, manifest_csv)
    np.random.seed(cfg["simulation"]["random_seed"])
    val_indices = list(range(int(0.8 * len(dataset)), len(dataset)))
    sample_indices = np.random.choice(val_indices, size=min(num_eval_samples, len(val_indices)), replace=False)

    print(f"Evaluating on {len(sample_indices)} validation tiles...\n")

    # 1. Profile FP32 (Single Thread)
    sess_fp32_1t = create_session(fp32_path, intra_threads=1)
    iou_fp32, cloud_fp32, lat_fp32_1t = evaluate_model(sess_fp32_1t, dataset, sample_indices, threshold)

    # 2. Profile INT8 (Single Thread)
    sess_int8_1t = create_session(int8_path, intra_threads=1)
    iou_int8, cloud_int8, lat_int8_1t = evaluate_model(sess_int8_1t, dataset, sample_indices, threshold)

    # 3. Profile INT8 (4 Threads)
    sess_int8_4t = create_session(int8_path, intra_threads=4)
    _, _, lat_int8_4t = evaluate_model(sess_int8_4t, dataset, sample_indices, threshold)

    delta_iou = iou_fp32 - iou_int8
    cloud_mae = abs(cloud_fp32 - cloud_int8)

    print("-" * 68)
    print(f"{'Metric':<32} | {'FP32 Model':<15} | {'INT8 Model':<15}")
    print("-" * 68)
    print(f"{'Validation mIoU':<32} | {iou_fp32:<15.4f} | {iou_int8:<15.4f}")
    print(f"{'mIoU Degradation (Delta)':<32} | {'-':<15} | {delta_iou:<15.4f}")
    print(f"{'Mean Cloud Coverage':<32} | {cloud_fp32 * 100:<14.2f}% | {cloud_int8 * 100:<14.2f}%")
    print(f"{'Cloud Fraction MAE':<32} | {'-':<15} | {cloud_mae:<15.4f}")
    print(f"{'CPU Latency (1 Thread)':<32} | {lat_fp32_1t:<12.2f} ms | {lat_int8_1t:<12.2f} ms")
    print(f"{'CPU Latency (4 Threads)':<32} | {'-':<15} | {lat_int8_4t:<12.2f} ms")
    print("-" * 68)

    budget_ms = capture_interval_s * 1000.0
    print(f"Orbital Budget Margin ({capture_interval_s}s frame rate): {budget_ms / lat_int8_1t:.1f}x real-time margin")
    print("=" * 68)


if __name__ == "__main__":
    run_benchmark()