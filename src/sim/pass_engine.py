import os
import sys
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def compute_bbox_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes:
    box = (xmin, ymin, xmax, ymax) in planar ground coordinates (km or degrees).
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0.0, xB - xA)
    inter_height = max(0.0, yB - yA)
    inter_area = inter_width * inter_height

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union_area = boxA_area + boxB_area - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


class TileMetadata:
    """Represents a captured multi-spectral tile payload."""
    def __init__(
        self,
        tile_id: str,
        capture_time: float,
        cloud_fraction: float,
        compressed_size_kb: float,
        base_utility: float,
        bbox: Optional[Tuple[float, float, float, float]] = None
    ):
        self.tile_id = tile_id
        self.capture_time = float(capture_time)
        self.cloud_fraction = float(cloud_fraction)
        self.compressed_size_kb = float(compressed_size_kb)
        self.base_utility = float(base_utility)
        # Default placeholder footprint if none provided: (0, 0, 1, 1)
        self.bbox = bbox if bbox is not None else (0.0, 0.0, 1.0, 1.0)

    def compute_priority(
        self,
        current_time: float,
        alpha: float = 1.5,
        beta: float = 0.25,
        decay_lambda: float = 0.0001
    ) -> float:
        """Computes dynamic priority without submodular overlap penalty."""
        dt = max(0.0, current_time - self.capture_time)
        size_term = max(1.0, self.compressed_size_kb) ** beta
        decay_term = np.exp(-decay_lambda * dt)
        return float(((self.base_utility ** alpha) / size_term) * decay_term)


class PriorityEngine:
    def __init__(
        self,
        alpha: float = 1.5,
        beta: float = 0.25,
        decay_lambda: float = 0.0001,
        gamma_overlap: float = 0.8
    ):
        self.alpha = alpha
        self.beta = beta
        self.decay_lambda = decay_lambda
        self.gamma_overlap = gamma_overlap

    def calculate_marginal_utility(
        self,
        candidate_cloud_frac: float,
        candidate_bbox: Tuple[float, float, float, float],
        buffered_bboxes: List[Tuple[float, float, float, float]]
    ) -> float:
        base_utility = max(0.0, 1.0 - candidate_cloud_frac)
        if not buffered_bboxes or self.gamma_overlap <= 0.0:
            return float(base_utility)

        max_iou = 0.0
        for b in buffered_bboxes:
            iou = compute_bbox_iou(candidate_bbox, b)
            if iou > max_iou:
                max_iou = iou

        redundancy_discount = max(0.0, 1.0 - (self.gamma_overlap * max_iou))
        return float(base_utility * redundancy_discount)

    def calculate_priority(
        self,
        cloud_fraction: float,
        compressed_size_kb: float,
        elapsed_time_s: float,
        candidate_bbox: Optional[Tuple[float, float, float, float]] = None,
        buffered_bboxes: Optional[List[Tuple[float, float, float, float]]] = None
    ) -> float:
        if candidate_bbox is not None and buffered_bboxes is not None:
            effective_utility = self.calculate_marginal_utility(
                cloud_fraction, candidate_bbox, buffered_bboxes
            )
        else:
            effective_utility = max(0.0, 1.0 - cloud_fraction)

        size_term = max(1.0, compressed_size_kb) ** self.beta
        time_decay = np.exp(-self.decay_lambda * max(0.0, elapsed_time_s))
        priority = ((effective_utility ** self.alpha) / size_term) * time_decay
        return float(priority)


class CloudInferenceEngine:
    """Runs ONNX inference for cloud detection on incoming optical tiles."""
    def __init__(self, model_path: str, intra_threads: int = 4):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def predict_cloud_fraction(self, image_np: np.ndarray, threshold: float = 0.5) -> float:
        """
        Runs model inference on an input tile and returns estimated cloud fraction.
        image_np: shape (4, H, W) float32 in [0, 1].
        """
        if image_np.ndim == 3:
            image_np = np.expand_dims(image_np, axis=0)

        logits = self.session.run(None, {self.input_name: image_np.astype(np.float32)})[0]
        probs = self._sigmoid(logits).squeeze()
        binary_mask = (probs >= threshold).astype(np.uint8)
        return float(np.mean(binary_mask))