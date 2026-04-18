"""WAV dataset for HiFi-GAN fine-tuning (loads WAV, random-crops segment_size samples)."""
from __future__ import annotations

import random
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset

SAMPLE_RATE = 22050


class WavDataset(Dataset):
    def __init__(self, wav_dir: str | Path, segment_size: int = 8192):
        self.wav_dir = Path(wav_dir)
        self.files = sorted(self.wav_dir.glob("*.wav"))
        self.segment_size = segment_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.files[idx])
        assert sr == SAMPLE_RATE
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)

        if wav.shape[1] >= self.segment_size:
            start = random.randint(0, wav.shape[1] - self.segment_size)
            wav = wav[:, start : start + self.segment_size]
        else:
            wav = torch.nn.functional.pad(wav, (0, self.segment_size - wav.shape[1]))

        return wav  # [1, segment_size]
