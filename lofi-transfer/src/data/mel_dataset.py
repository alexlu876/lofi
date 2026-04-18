"""Mel spectrogram dataset for CycleGAN training (loads .npy, applies normalization)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MelDataset(Dataset):
    def __init__(self, file_list: list[str] | list[Path], stats_path: str | Path):
        self.files = [Path(f) for f in file_list]
        with open(stats_path) as f:
            stats = json.load(f)
        self.mean = stats["mean"]
        self.std = stats["std"]

    @classmethod
    def from_split_file(cls, split_path: str | Path, stats_path: str | Path):
        with open(split_path) as f:
            files = [line.strip() for line in f if line.strip()]
        return cls(files, stats_path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mel = np.load(self.files[idx]).astype(np.float32)  # [80, 432]
        mel_norm = (mel - self.mean) / self.std
        return torch.from_numpy(mel_norm).unsqueeze(0)  # [1, 80, 432]
