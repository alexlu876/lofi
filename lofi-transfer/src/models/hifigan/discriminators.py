"""HiFi-GAN Multi-Period and Multi-Scale Discriminators (spec §5.2, §5.3)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm, spectral_norm


LRELU_SLOPE = 0.1


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        channels = [1, 32, 128, 512, 1024, 1024]
        self.convs = nn.ModuleList()
        for i in range(len(channels) - 1):
            stride = (3, 1) if i < 4 else 1
            padding = (2, 0)
            self.convs.append(
                weight_norm(nn.Conv2d(channels[i], channels[i + 1], (5, 1), stride, padding))
            )
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, (1, 0)))

    def forward(self, x):
        fmaps = []
        B, C, T = x.shape
        pad = (self.period - T % self.period) % self.period
        if pad > 0:
            x = F.pad(x, (0, pad), "reflect")
            T = T + pad
        x = x.view(B, C, T // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmaps.append(x)
        x = self.conv_post(x)
        fmaps.append(x)
        return x.flatten(1, -1), fmaps


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: list[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, y, y_hat):
        y_outs, y_hat_outs = [], []
        y_fmaps, y_hat_fmaps = [], []
        for d in self.discriminators:
            y_out, y_fmap = d(y)
            y_hat_out, y_hat_fmap = d(y_hat)
            y_outs.append(y_out)
            y_hat_outs.append(y_hat_out)
            y_fmaps.append(y_fmap)
            y_hat_fmaps.append(y_hat_fmap)
        return y_outs, y_hat_outs, y_fmaps, y_hat_fmaps


class ScaleDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, 7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, 20, groups=4)),
            norm_f(nn.Conv1d(128, 256, 41, 2, 20, groups=16)),
            norm_f(nn.Conv1d(256, 512, 41, 4, 20, groups=16)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, 20, groups=16)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, 20, groups=16)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, 2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, 1))

    def forward(self, x):
        fmaps = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmaps.append(x)
        x = self.conv_post(x)
        fmaps.append(x)
        return x.flatten(1, -1), fmaps


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=True),
            ScaleDiscriminator(),
            ScaleDiscriminator(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, y, y_hat):
        y_outs, y_hat_outs = [], []
        y_fmaps, y_hat_fmaps = [], []
        for i, d in enumerate(self.discriminators):
            if i > 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_out, y_fmap = d(y)
            y_hat_out, y_hat_fmap = d(y_hat)
            y_outs.append(y_out)
            y_hat_outs.append(y_hat_out)
            y_fmaps.append(y_fmap)
            y_hat_fmaps.append(y_hat_fmap)
        return y_outs, y_hat_outs, y_fmaps, y_hat_fmaps
