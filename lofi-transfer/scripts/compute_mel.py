#!/usr/bin/env python3
"""Compute log-mel spectrograms from 5-second clips.

Output: .npy files of shape [80, 432], float32, log-compressed, UN-normalized.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80
F_MIN = 0
F_MAX = 8000
N_FRAMES = 432
LOG_MEL_FLOOR = 1e-5
PAD_VALUE = np.log(LOG_MEL_FLOOR)  # ~ -11.512


def build_mel_transform(device: str = "cpu") -> torchaudio.transforms.MelSpectrogram:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        f_min=F_MIN,
        f_max=F_MAX,
        norm="slaney",
        mel_scale="slaney",
    ).to(device)


def wav_to_log_mel(wav_path: Path, mel_transform, device: str = "cpu") -> np.ndarray:
    wav, sr = torchaudio.load(wav_path)
    assert sr == SAMPLE_RATE
    wav = wav.to(device)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)

    mel = mel_transform(wav)  # [1, 80, T]
    log_mel = torch.log(torch.clamp(mel, min=LOG_MEL_FLOOR))
    log_mel = log_mel.squeeze(0)  # [80, T]

    T = log_mel.shape[1]
    if T > N_FRAMES:
        start = (T - N_FRAMES) // 2
        log_mel = log_mel[:, start : start + N_FRAMES]
    elif T < N_FRAMES:
        pad_amount = N_FRAMES - T
        log_mel = torch.nn.functional.pad(log_mel, (0, pad_amount), value=PAD_VALUE)

    return log_mel.cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Compute log-mel spectrograms")
    parser.add_argument("--domain", required=True, choices=["clean", "lofi"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default="cpu", help="cpu or mps or cuda")
    args = parser.parse_args()

    base = Path(args.data_dir) if args.data_dir else Path(__file__).resolve().parent.parent / "data"
    src_dir = base / "clips_5s" / args.domain
    out_dir = base / "mel" / args.domain
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        print(f"Source not found: {src_dir}", file=sys.stderr)
        sys.exit(1)

    wav_files = sorted(src_dir.glob("*.wav"))
    if not wav_files:
        print(f"No clips found in {src_dir}", file=sys.stderr)
        sys.exit(1)

    mel_transform = build_mel_transform(args.device)
    count = 0
    for wav_path in tqdm(wav_files, desc=f"Mel {args.domain}"):
        npy_path = out_dir / f"{wav_path.stem}.npy"
        if npy_path.exists():
            continue
        log_mel = wav_to_log_mel(wav_path, mel_transform, args.device)
        assert log_mel.shape == (N_MELS, N_FRAMES), f"Bad shape {log_mel.shape}"
        np.save(npy_path, log_mel)
        count += 1

    print(f"Wrote {count} mel files to {out_dir}")


if __name__ == "__main__":
    main()
