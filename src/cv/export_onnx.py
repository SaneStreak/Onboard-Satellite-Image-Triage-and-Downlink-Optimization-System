import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import onnx
import onnxruntime as ort
import numpy as np
from src.cv.models import TinyCloudUNet

def export_model():
    checkpoint_path = os.path.join("checkpoints", "tiny_cloud_unet_best.pth")
    onnx_path = os.path.join("checkpoints", "tiny_cloud_unet.onnx")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Initialize model and load weights
    model = TinyCloudUNet(in_channels=4, num_classes=1)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    # Define dummy input: (batch_size=1, channels=4, H=384, W=384)
    dummy_input = torch.randn(1, 4, 384, 384, requires_grad=False)

    # Export graph
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_tile"],
        output_names=["cloud_logits"],
        dynamic_axes={
            "input_tile": {0: "batch_size"},
            "cloud_logits": {0: "batch_size"}
        }
    )
    print(f"Model successfully exported to: {onnx_path}")

    # Validate ONNX graph integrity
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure checked and verified.")

    # Validate inference through ONNX Runtime
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = session.run(None, ort_inputs)

    # Verify output shape
    print("ONNX Runtime verification output shape:", ort_outs[0].shape)
    
    # Calculate file size
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"ONNX Model Binary Size: {size_mb:.2f} MB")

if __name__ == "__main__":
    export_model()