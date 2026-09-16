import os
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.cv.dataset import Cloud38Dataset
from src.cv.models import TinyCloudUNet
from src.utils.metrics import compute_iou

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()
        
        return 0.5 * bce_loss + 0.5 * dice_loss

def train_epoch(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad()
        
        # Mixed precision forward pass via Tensor Cores
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits = model(images)
            loss = criterion(logits, masks)

        if device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_iou = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, masks)
            running_loss += loss.item() * images.size(0)

            total_iou += compute_iou(logits, masks) * images.size(0)

    val_loss = running_loss / len(dataloader.dataset)
    val_iou = total_iou / len(dataloader.dataset)
    return val_loss, val_iou

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    if device.type == "cuda":
        print(f"Device Name: {torch.cuda.get_device_name(0)}")

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # Dataset setup
    data_root = os.path.join("data", "raw", "38-Cloud")
    manifest_csv = os.path.join(data_root, "training_patches_38-cloud_nonempty.csv")
    full_ds = Cloud38Dataset(data_root, manifest_csv)

    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # Using batch_size=32 for GPU utilization
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    model = TinyCloudUNet(in_channels=4, num_classes=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = DiceBCELoss()

    os.makedirs("checkpoints", exist_ok=True)
    best_iou = 0.0
    epochs = 5

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_iou = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), "checkpoints/tiny_cloud_unet_best.pth")
            print(f" -> Saved new best model checkpoint with mIoU: {best_iou:.4f}")

if __name__ == "__main__":
    main()