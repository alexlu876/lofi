"""CycleGAN U-Net generator with 9-block residual bottleneck (spec §3.1)."""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, 1, 0),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, 1, 0),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x):
        return x + self.block(x)


class UNetGenerator(nn.Module):
    """U-Net encoder-decoder with skip connections and residual bottleneck.

    Input/Output: [B, 1, 80, 432]
    """

    def __init__(self, base_channels: int = 64, n_resblocks: int = 9):
        super().__init__()
        c = base_channels  # 64

        self.stem = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(1, c, 7, 1, 0),
            nn.InstanceNorm2d(c, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.down1 = self._down_block(c, c * 2)        # 64 -> 128
        self.down2 = self._down_block(c * 2, c * 4)    # 128 -> 256
        self.down3 = self._down_block(c * 4, c * 8)    # 256 -> 512
        self.down4 = self._down_block(c * 8, c * 8)    # 512 -> 512

        self.bottleneck = nn.Sequential(
            *[ResBlock(c * 8) for _ in range(n_resblocks)]
        )

        self.up4 = nn.ConvTranspose2d(c * 8, c * 8, 4, 2, 1)
        self.up4_norm = nn.Sequential(nn.InstanceNorm2d(c * 8, affine=True), nn.ReLU(inplace=True))
        self.up4_conv = nn.Sequential(
            nn.Conv2d(c * 16, c * 8, 3, 1, 1),
            nn.InstanceNorm2d(c * 8, affine=True),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 4, 2, 1)
        self.up3_norm = nn.Sequential(nn.InstanceNorm2d(c * 4, affine=True), nn.ReLU(inplace=True))
        self.up3_conv = nn.Sequential(
            nn.Conv2d(c * 8, c * 4, 3, 1, 1),
            nn.InstanceNorm2d(c * 4, affine=True),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1)
        self.up2_norm = nn.Sequential(nn.InstanceNorm2d(c * 2, affine=True), nn.ReLU(inplace=True))
        self.up2_conv = nn.Sequential(
            nn.Conv2d(c * 4, c * 2, 3, 1, 1),
            nn.InstanceNorm2d(c * 2, affine=True),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(c * 2, c, 4, 2, 1)
        self.up1_norm = nn.Sequential(nn.InstanceNorm2d(c, affine=True), nn.ReLU(inplace=True))
        self.up1_conv = nn.Sequential(
            nn.Conv2d(c * 2, c, 3, 1, 1),
            nn.InstanceNorm2d(c, affine=True),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(c, 1, 7, 1, 0),
            nn.Tanh(),
        )

    @staticmethod
    def _down_block(in_ch: int, out_ch: int):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 2, 1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        s0 = self.stem(x)    # [B, 64, 80, 432]
        s1 = self.down1(s0)  # [B, 128, 40, 216]
        s2 = self.down2(s1)  # [B, 256, 20, 108]
        s3 = self.down3(s2)  # [B, 512, 10, 54]
        s4 = self.down4(s3)  # [B, 512, 5, 27]

        b = self.bottleneck(s4)  # [B, 512, 5, 27]

        d4 = self.up4_norm(self.up4(b))
        d4 = self.up4_conv(torch.cat([d4, s3], dim=1))

        d3 = self.up3_norm(self.up3(d4))
        d3 = self.up3_conv(torch.cat([d3, s2], dim=1))

        d2 = self.up2_norm(self.up2(d3))
        d2 = self.up2_conv(torch.cat([d2, s1], dim=1))

        d1 = self.up1_norm(self.up1(d2))
        d1 = self.up1_conv(torch.cat([d1, s0], dim=1))

        return self.head(d1)
