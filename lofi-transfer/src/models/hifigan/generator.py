"""HiFi-GAN V1 Generator (spec §5.1). Matches jik876/hifi-gan checkpoint layout."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


LRELU_SLOPE = 0.1


class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: list[int]):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for d in dilations:
            pad1 = (kernel_size - 1) * d // 2
            pad2 = (kernel_size - 1) // 2
            self.convs1.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, pad1, dilation=d)))
            self.convs2.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, pad2, dilation=1)))

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = x + xt
        return x


class HiFiGANGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 80,
        upsample_initial_channel: int = 512,
        upsample_rates: list[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: list[int] = [16, 16, 4, 4],
        resblock_kernel_sizes: list[int] = [3, 7, 11],
        resblock_dilation_sizes: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    ):
        super().__init__()
        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)

        self.conv_pre = weight_norm(nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, 3))

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        ch = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            out_ch = ch // 2
            self.ups.append(
                weight_norm(nn.ConvTranspose1d(ch, out_ch, k, u, padding=(k - u) // 2))
            )
            for j, (rk, rd) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock1(out_ch, rk, rd))
            ch = out_ch

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, 3))

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = 0
            for j in range(self.num_kernels):
                xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x
