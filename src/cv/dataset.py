import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class Cloud38Dataset(Dataset):
    def __init__(self, data_root: str, manifest_csv: str, transform=None):
        """
        data_root: Path to 'data/raw/38-Cloud'
        manifest_csv: Path to 'training_patches_38-cloud_nonempty.csv'
        """
        self.data_root = data_root
        self.df = pd.read_csv(manifest_csv)
        self.transform = transform
        
        # Paths nested under '38-Cloud_training'
        train_dir = os.path.join(data_root, "38-Cloud_training")
        self.red_dir = os.path.join(train_dir, "train_red")
        self.green_dir = os.path.join(train_dir, "train_green")
        self.blue_dir = os.path.join(train_dir, "train_blue")
        self.nir_dir = os.path.join(train_dir, "train_nir")
        self.gt_dir = os.path.join(train_dir, "train_gt")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Extract patch ID from manifest (column: 'name')
        patch_name = self.df.iloc[idx, 0]

        # File names follow pattern: '<band>_<patch_name>.TIF'
        red_path = os.path.join(self.red_dir, f"red_{patch_name}.TIF")
        green_path = os.path.join(self.green_dir, f"green_{patch_name}.TIF")
        blue_path = os.path.join(self.blue_dir, f"blue_{patch_name}.TIF")
        nir_path = os.path.join(self.nir_dir, f"nir_{patch_name}.TIF")
        gt_path = os.path.join(self.gt_dir, f"gt_{patch_name}.TIF")

        # Load 16-bit TIFFs, normalize uint16 (0-65535) -> float32 (0.0-1.0)
        r = np.array(Image.open(red_path), dtype=np.float32) / 65535.0
        g = np.array(Image.open(green_path), dtype=np.float32) / 65535.0
        b = np.array(Image.open(blue_path), dtype=np.float32) / 65535.0
        nir = np.array(Image.open(nir_path), dtype=np.float32) / 65535.0

        # Stack into (4, H, W)
        img_tensor = torch.from_numpy(np.stack([r, g, b, nir], axis=0))

        # Binary cloud mask (0 = clear, 255 = cloud) -> float32 (1, H, W)
        gt = np.array(Image.open(gt_path), dtype=np.float32)
        gt_mask = np.where(gt > 128, 1.0, 0.0).astype(np.float32)
        gt_tensor = torch.from_numpy(gt_mask).unsqueeze(0)

        cloud_fraction = float(gt_mask.mean())

        return {
            "image": img_tensor,
            "mask": gt_tensor,
            "cloud_fraction": cloud_fraction,
            "patch_id": patch_name
        }

if __name__ == "__main__":
    DATA_ROOT = os.path.join("data", "raw", "38-Cloud")
    CSV_PATH = os.path.join(DATA_ROOT, "training_patches_38-cloud_nonempty.csv")
    
    if os.path.exists(CSV_PATH):
        ds = Cloud38Dataset(DATA_ROOT, CSV_PATH)
        print(f"Dataset successfully indexed {len(ds)} non-empty patches.")
        sample = ds[0]
        print("Sample verification:")
        print(" - Image Tensor Shape:", sample["image"].shape)
        print(" - Mask Tensor Shape :", sample["mask"].shape)
        print(" - Patch ID          :", sample["patch_id"])
        print(" - Cloud Fraction    :", round(sample["cloud_fraction"], 4))
    else:
        print(f"Error: Manifest not found at {CSV_PATH}")