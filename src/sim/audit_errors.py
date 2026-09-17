import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cv.dataset import Cloud38Dataset
from src.sim.benchmark_quant import compute_numpy_iou, sigmoid, create_session


def compute_ndsi(image_4band: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Computes Normalized Difference Snow Index (NDSI):
    Bands: 0: Red, 1: Green, 2: Blue, 3: NIR
    NDSI = (Green - NIR) / (Green + NIR)
    """
    green = image_4band[1].astype(np.float32)
    nir = image_4band[3].astype(np.float32)
    ndsi = (green - nir) / (green + nir + eps)
    return ndsi


def run_error_audit(
    config_path: str = "tests/base_config.yaml",
    int8_path: str = "checkpoints/tiny_cloud_unet_int8.onnx",
    top_k: int = 3
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_root = cfg["paths"]["data_root"]
    manifest_csv = cfg["paths"]["manifest_csv"]
    threshold = cfg["triage"]["cloud_threshold"]

    print("=" * 70)
    print("RUNNING QUALITATIVE AUDIT & FALSE ALBEDO TRIAGE ANALYSIS")
    print("=" * 70)

    dataset = Cloud38Dataset(data_root, manifest_csv)
    session = create_session(int8_path, intra_threads=4)
    input_name = session.get_inputs()[0].name

    np.random.seed(cfg["simulation"]["random_seed"])
    val_indices = list(range(int(0.8 * len(dataset)), len(dataset)))
    sample_indices = np.random.choice(val_indices, size=min(300, len(val_indices)), replace=False)

    records = []

    print(f"Scanning {len(sample_indices)} validation tiles for triage discrepancies...")

    for idx in sample_indices:
        sample = dataset[idx]
        img_tensor = sample["image"]
        img_np = img_tensor.unsqueeze(0).numpy().astype(np.float32)
        true_mask = sample["mask"].squeeze(0).numpy().astype(np.uint8)

        logits = session.run(None, {input_name: img_np})[0]
        probs = sigmoid(logits).squeeze()
        pred_mask = (probs >= threshold).astype(np.uint8)

        true_cloud_frac = float(np.mean(true_mask))
        pred_cloud_frac = float(np.mean(pred_mask))
        iou = compute_numpy_iou(pred_mask, true_mask)
        cloud_error = pred_cloud_frac - true_cloud_frac  # >0: False Cloud, <0: Missed Cloud

        # Compute mean NDSI on ground/snow regions
        raw_4band = img_tensor.numpy()
        ndsi = compute_ndsi(raw_4band)
        mean_ndsi = float(np.mean(ndsi))

        records.append({
            "idx": idx,
            "true_frac": true_cloud_frac,
            "pred_frac": pred_cloud_frac,
            "cloud_error": cloud_error,
            "iou": iou,
            "mean_ndsi": mean_ndsi,
            "raw_img": raw_4band,
            "true_mask": true_mask,
            "pred_mask": pred_mask
        })

    # Sort to isolate failure modes
    # False Positives: Model thought it was heavily clouded, but ground truth was clear
    false_positives = sorted(records, key=lambda x: x["cloud_error"], reverse=True)[:top_k]
    # False Negatives: Model thought it was clear, but ground truth had heavy cloud cover
    false_negatives = sorted(records, key=lambda x: x["cloud_error"])[:top_k]

    print("\nAuditing Critical Edge Cases:")
    print(f" - Top False-Positive Over-Penalty: Tile {false_positives[0]['idx']} (Error: +{false_positives[0]['cloud_error']*100:.1f}%)")
    print(f" - Top False-Negative Cloud Leakage: Tile {false_negatives[0]['idx']} (Error: {false_negatives[0]['cloud_error']*100:.1f}%)")

    # Plotting Gallery
    total_cases = top_k * 2
    fig, axes = plt.subplots(total_cases, 4, figsize=(16, 3.2 * total_cases))
    fig.patch.set_facecolor("#faf8f5")

    cases = [("FP (Erroneous Eviction Risk)", c) for c in false_positives] + \
            [("FN (Leakage / Wasted Bandwidth)", c) for c in false_negatives]

    for row_idx, (case_type, data) in enumerate(cases):
        # 1. RGB Render
        rgb = np.transpose(data["raw_img"][:3], (1, 2, 0))
        rgb = np.clip(rgb / np.percentile(rgb, 98), 0, 1)  # Adaptive contrast stretch

        # 2. Ground Truth Mask
        gt = data["true_mask"]

        # 3. Model Prediction
        pred = data["pred_mask"]

        # 4. Error Heatmap (Red: False Positive, Blue: False Negative)
        err_map = np.zeros((*gt.shape, 3), dtype=np.float32)
        err_map[(pred == 1) & (gt == 0)] = [0.9, 0.2, 0.2]  # Red: Over-predicted cloud
        err_map[(pred == 0) & (gt == 1)] = [0.2, 0.4, 0.9]  # Blue: Missed cloud

        axes[row_idx, 0].imshow(rgb)
        axes[row_idx, 0].set_title(f"Tile {data['idx']} - RGB\n{case_type}", fontsize=10, fontweight="bold")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(gt, cmap="gray")
        axes[row_idx, 1].set_title(f"True Mask (Cloud: {data['true_frac']*100:.1f}%)", fontsize=10)
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(pred, cmap="gray")
        axes[row_idx, 2].set_title(f"Pred Mask (Cloud: {data['pred_frac']*100:.1f}%)\nIoU: {data['iou']:.3f}", fontsize=10)
        axes[row_idx, 2].axis("off")

        axes[row_idx, 3].imshow(err_map)
        axes[row_idx, 3].set_title(f"Error (Red=FP, Blue=FN)\nMean NDSI: {data['mean_ndsi']:.2f}", fontsize=10)
        axes[row_idx, 3].axis("off")

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/triage_error_audit.png"
    plt.savefig(out_path, dpi=250)
    print(f"\nAudit visual diagnostics successfully saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_error_audit()