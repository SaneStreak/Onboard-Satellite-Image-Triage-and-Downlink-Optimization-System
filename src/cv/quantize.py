import os
import sys
import numpy as np
import onnxruntime
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cv.dataset import Cloud38Dataset


class CloudCalibrationDataReader(CalibrationDataReader):
    def __init__(self, dataset: Cloud38Dataset, num_samples: int = 100, input_name: str = "input_tile"):
        super().__init__()
        self.dataset = dataset
        self.num_samples = min(num_samples, len(dataset))
        self.input_name = input_name
        self.current_idx = 0
        
        # Sample representative patches across the dataset
        np.random.seed(42)
        self.sample_indices = np.random.choice(len(dataset), size=self.num_samples, replace=False)

    def get_next(self):
        if self.current_idx >= self.num_samples:
            return None
        
        idx = int(self.sample_indices[self.current_idx])
        self.current_idx += 1
        
        sample = self.dataset[idx]
        # (4, 384, 384) -> (1, 4, 384, 384)
        tensor_np = sample["image"].unsqueeze(0).numpy().astype(np.float32)
        
        return {self.input_name: tensor_np}

    def rewind(self):
        self.current_idx = 0


def run_int8_quantization(
    fp32_model_path: str = "checkpoints/tiny_cloud_unet.onnx",
    int8_model_path: str = "checkpoints/tiny_cloud_unet_int8.onnx",
    data_root: str = "data/raw/38-Cloud",
    manifest_csv: str = "data/raw/38-Cloud/training_patches_38-cloud_nonempty.csv",
    calibration_samples: int = 150
):
    os.makedirs(os.path.dirname(int8_model_path), exist_ok=True)
    
    print("=" * 65)
    print("INITIALIZING STATIC INT8 POST-TRAINING QUANTIZATION (PTQ)")
    print("=" * 65)
    print(f"FP32 Source Model : {fp32_model_path}")
    print(f"INT8 Target Model : {int8_model_path}")
    print(f"Calibration Set   : {calibration_samples} multi-spectral tiles")
    
    # Cloud38Dataset takes (data_root, csv_path) positionally
    dataset = Cloud38Dataset(data_root, manifest_csv)
    calibration_reader = CloudCalibrationDataReader(
        dataset=dataset,
        num_samples=calibration_samples,
        input_name="input_tile"
    )

    print("\nCalculating activation dynamic ranges and quantizing weights...")
    quantize_static(
        model_input=fp32_model_path,
        model_output=int8_model_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False
    )

    fp32_bytes = os.path.getsize(fp32_model_path)
    int8_bytes = os.path.getsize(int8_model_path)

    print("\nQuantization Complete:")
    print(f" - FP32 Footprint : {fp32_bytes / 1024.0:.2f} KB ({fp32_bytes / (1024.0**2):.2f} MB)")
    print(f" - INT8 Footprint : {int8_bytes / 1024.0:.2f} KB ({int8_bytes / (1024.0**2):.2f} MB)")
    print(f" - Compression    : {(1.0 - int8_bytes / fp32_bytes) * 100:.2f}% memory reduction")


if __name__ == "__main__":
    run_int8_quantization()