import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """[Conv2d -> BatchNorm -> ReLU] x 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class TinyCloudUNet(nn.Module):
    """
    Lightweight 4-channel U-Net for onboard cloud screening.
    Footprint: ~450k parameters.
    """
    def __init__(self, in_channels=4, num_classes=1):
        super().__init__()
        # Encoder
        self.inc = DoubleConv(in_channels, 16)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))

        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(64, 32)

        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(32, 16)

        # Pixel mask output (raw logits)
        self.outc = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # Forward pass down
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Forward pass up with skip connections
        u1 = self.up1(x4)
        u1 = torch.cat([x3, u1], dim=1)
        u1 = self.conv_up1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([x2, u2], dim=1)
        u2 = self.conv_up2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([x1, u3], dim=1)
        u3 = self.conv_up3(u3)

        logits = self.outc(u3)
        return logits

if __name__ == "__main__":
    model = TinyCloudUNet(in_channels=4, num_classes=1)
    dummy_input = torch.randn(1, 4, 384, 384)
    out = model(dummy_input)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("TinyCloudUNet Instantiated Successfully:")
    print(" - Input Shape       :", dummy_input.shape)
    print(" - Output Logits     :", out.shape)
    print(" - Trainable Params  :", f"{total_params:,}")