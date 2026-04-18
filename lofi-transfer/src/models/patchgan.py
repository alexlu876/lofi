"""PatchGAN discriminator (spec §3.2). 70x70-equivalent receptive field.

Uses weight clipping as a Lipschitz constraint for MPS compatibility,
with proper spectral normalization available for CUDA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_discriminator_norm(device_type: str):
    """Return the appropriate normalization for the device."""
    if device_type == "cuda":
        from torch.nn.utils.parametrizations import spectral_norm
        return spectral_norm
    return None  # No spectral norm on MPS; use weight clipping instead


class PatchGANDiscriminator(nn.Module):
    """Input: [B, 1, 80, 432] -> Output: [B, 1, 8, 52]"""

    def __init__(self, base_channels: int = 64):
        super().__init__()
        c = base_channels

        self.model = nn.Sequential(
            nn.Conv2d(1, c, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(c, c * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(c * 2, c * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(c * 4, c * 8, 4, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(c * 8, 1, 4, 1, 1),
        )

    def apply_spectral_norm(self):
        """Apply spectral norm to all conv layers. Call after .to(device) for CUDA."""
        from torch.nn.utils.parametrizations import spectral_norm
        for i, m in enumerate(self.model):
            if isinstance(m, nn.Conv2d):
                self.model[i] = spectral_norm(m)
        return self

    def clip_weights(self, clip_value: float = 0.01):
        """WGAN-style weight clipping as MPS-compatible Lipschitz constraint."""
        for p in self.parameters():
            p.data.clamp_(-clip_value, clip_value)

    def forward(self, x):
        return self.model(x)
