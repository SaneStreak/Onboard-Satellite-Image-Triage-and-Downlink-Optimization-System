import io
import torch
import numpy as np
from PIL import Image

def compute_iou(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7) -> float:
    """
    Computes Intersection over Union (IoU) for binary segmentation masks.
    preds: Tensor of raw logits or sigmoid probabilities.
    targets: Tensor of binary targets (0.0 or 1.0).
    """
    if preds.shape != targets.shape:
        raise ValueError(f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}")
    
    # Convert logits to binary predictions
    probs = torch.sigmoid(preds) if preds.min() < 0.0 or preds.max() > 1.0 else preds
    bin_preds = (probs > threshold).float()
    bin_targets = (targets > threshold).float()

    intersection = (bin_preds * bin_targets).sum().item()
    union = bin_preds.sum().item() + bin_targets.sum().item() - intersection

    if union == 0:
        return 1.0  # Perfect agreement on empty mask
    return (intersection + eps) / (union + eps)

def estimate_compressed_size(image_tensor: torch.Tensor, quality: int = 80) -> int:
    """
    Simulates onboard lossy/lossless compression (WebP proxy) on a (4, H, W) float32 tensor.
    Returns: Footprint in bytes (s_i).
    """
    # Detach and scale back to uint8 RGB representation for compression proxy
    arr = image_tensor[:3].detach().cpu().numpy()  # Take RGB channels
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))  # Convert (3, H, W) -> (H, W, 3)

    img = Image.fromarray(arr)
    buffer = io.BytesIO()
    # WebP provides compression characteristics similar to modern CCSDS 122.0 standards
    img.save(buffer, format="WEBP", quality=quality)
    
    compressed_bytes = buffer.tell()
    return compressed_bytes

if __name__ == "__main__":
    # Smoke test verification
    dummy_mask_pred = torch.randn(1, 1, 384, 384)
    dummy_mask_gt = torch.zeros(1, 1, 384, 384)
    dummy_mask_gt[:, :, 100:200, 100:200] = 1.0

    iou = compute_iou(dummy_mask_pred, dummy_mask_gt)
    
    dummy_tile = torch.rand(4, 384, 384)
    byte_size = estimate_compressed_size(dummy_tile, quality=75)
    
    print("Metrics Module Operational:")
    print(" - Sample mIoU       :", round(iou, 4))
    print(" - Uncompressed Tile :", f"{dummy_tile.nelement() * 4 / 1024:.2f} KB (FP32)")
    print(" - Compressed Size   :", f"{byte_size / 1024:.2f} KB (s_i)")