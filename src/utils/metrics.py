import io
import zlib
import numpy as np
import torch
from PIL import Image


def compute_iou(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """Computes mIoU for binary segmentation on PyTorch tensors."""
    probs = torch.sigmoid(preds)
    bin_preds = (probs > threshold).float()
    
    intersection = (bin_preds * targets).sum().item()
    union = (bin_preds + targets).clamp(0, 1).sum().item()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
        
    return (intersection + smooth) / (union + smooth)


def estimate_compressed_size(tensor: torch.Tensor, level: int = 6) -> int:
    """
    Estimates the true compressed storage footprint of a 4-band multi-spectral tile.
    
    Compresses all 4 channels (R, G, B, NIR) using zlib (Deflate level 6) on 
    quantized 8-bit band buffers. This captures entropy differences: complex ground 
    textures yield larger footprints, while homogeneous cloud cover yields smaller ones.
    
    Args:
        tensor: Tensor of shape (4, H, W) normalized to [0, 1].
        level: Compression level (1-9).
        
    Returns:
        Exact byte size of the compressed multi-spectral tile payload.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
        
    # Scale normalized float32 to standard 8-bit digital numbers (DN)
    tensor_uint8 = (tensor.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
    
    # Interleave bands (H, W, 4) for spatial-spectral delta redundancy
    interleaved = np.transpose(tensor_uint8, (1, 2, 0)).tobytes()
    
    # Compress the byte stream directly
    compressed_bytes = zlib.compress(interleaved, level=level)
    return len(compressed_bytes)