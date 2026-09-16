import numpy as np
from dataclasses import dataclass

@dataclass
class TileMetadata:
    tile_id: str
    capture_time: float      # Simulation timestamp in seconds
    cloud_fraction: float    # Fraction in [0.0, 1.0] from ONNX inference
    compressed_size_kb: float# Empirical footprint s_i in KB
    base_utility: float      # U_i in [0.0, 1.0]

    def compute_priority(
        self,
        current_time: float,
        alpha: float = 1.0,
        beta: float = 0.5,
        decay_lambda: float = 1e-4
    ) -> float:
        """
        Computes the dynamic priority P_i(t) = (U_i^alpha / s_i^beta) * exp(-lambda * delta_t).
        """
        delta_t = max(0.0, current_time - self.capture_time)
        time_decay = np.exp(-decay_lambda * delta_t)
        
        # Guard against zero-division or zero-size edge cases
        size_term = max(self.compressed_size_kb, 1.0) ** beta
        utility_term = max(self.base_utility, 1e-3) ** alpha
        
        return float((utility_term / size_term) * time_decay)


class CloudInferenceEngine:
    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict_cloud_fraction(self, tile_tensor: np.ndarray, threshold: float = 0.5) -> float:
        """
        Runs inference on a (1, 4, 384, 384) numpy array.
        Returns the scalar cloud fraction across the 384x384 patch.
        """
        if tile_tensor.ndim == 3:
            tile_tensor = np.expand_dims(tile_tensor, axis=0)

        # Output shape: (1, 1, 384, 384)
        logits = self.session.run(None, {self.input_name: tile_tensor.astype(np.float32)})[0]
        
        # Logistic sigmoid mapping: 1 / (1 + exp(-logits))
        probs = 1.0 / (1.0 + np.exp(-logits))
        cloud_mask = probs > threshold
        
        return float(np.mean(cloud_mask))


if __name__ == "__main__":
    import os
    onnx_path = os.path.join("checkpoints", "tiny_cloud_unet.onnx")
    
    engine = CloudInferenceEngine(onnx_path)
    dummy_tile = np.random.rand(1, 4, 384, 384).astype(np.float32)
    cloud_frac = engine.predict_cloud_fraction(dummy_tile)
    
    sample_tile = TileMetadata(
        tile_id="patch_sample_001",
        capture_time=100.0,
        cloud_fraction=cloud_frac,
        compressed_size_kb=89.2,
        base_utility=1.0 - cloud_frac
    )
    
    p_t0 = sample_tile.compute_priority(current_time=100.0)
    p_t1 = sample_tile.compute_priority(current_time=5000.0)
    
    print("Pass Engine Initialized:")
    print(f" - Predicted Cloud Fraction : {cloud_frac:.4f}")
    print(f" - Base Utility (U_i)       : {sample_tile.base_utility:.4f}")
    print(f" - Priority at t=100s       : {p_t0:.6f}")
    print(f" - Priority at t=5000s      : {p_t1:.6f} (Decayed)")