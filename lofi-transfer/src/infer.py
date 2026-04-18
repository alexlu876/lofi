"""Phase 3: End-to-end inference pipeline (spec §7).

WAV -> log-mel -> CycleGAN G(clean->lofi) -> lofi log-mel -> HiFi-GAN -> WAV
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.cyclegan_generator import UNetGenerator
from src.models.hifigan import HiFiGANGenerator

SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80
F_MIN = 0
F_MAX = 8000
N_FRAMES = 432
LOG_MEL_FLOOR = 1e-5
PAD_VALUE = np.log(LOG_MEL_FLOOR)


def build_mel_transform(device="cpu"):
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


def load_stats(path):
    with open(path) as f:
        s = json.load(f)
    return s["mean"], s["std"]


def wav_to_chunks(wav, chunk_samples, hop_samples):
    T = wav.shape[-1]
    chunks = []
    pos = 0
    while pos < T:
        end = pos + chunk_samples
        chunk = wav[..., pos:end]
        if chunk.shape[-1] < chunk_samples:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - chunk.shape[-1]))
        chunks.append(chunk)
        pos += hop_samples
    return torch.stack(chunks)  # [N, 1, chunk_samples]


def mel_domain_overlap_add(mels, n_overlap_frames):
    """Stitch mel chunks with cosine crossfade in the overlap region."""
    if mels.shape[0] == 1:
        return mels[0]

    N, C, F, T = mels.shape
    hop_frames = T - n_overlap_frames
    total_frames = hop_frames * (N - 1) + T
    out = torch.zeros(C, F, total_frames, device=mels.device)
    weight = torch.zeros(1, 1, total_frames, device=mels.device)

    t = torch.linspace(0, 1, n_overlap_frames, device=mels.device)
    fade_in = (0.5 * (1.0 - torch.cos(torch.pi * t))).view(1, 1, -1)
    fade_out = 1.0 - fade_in

    for i in range(N):
        start = i * hop_frames
        w = torch.ones(1, 1, T, device=mels.device)
        if i > 0:
            w[..., :n_overlap_frames] = fade_in
        if i < N - 1:
            w[..., -n_overlap_frames:] = fade_out
        out[:, :, start:start + T] += mels[i] * w
        weight[:, :, start:start + T] += w

    return out / weight.clamp(min=1e-8)


def convert_to_lofi(
    in_wav_path: str,
    out_wav_path: str,
    G: UNetGenerator,
    hifigan: HiFiGANGenerator,
    clean_mean: float,
    clean_std: float,
    lofi_mean: float,
    lofi_std: float,
    mel_transform,
    device: str = "cpu",
    chunk_sec: float = 5.0,
    overlap_sec: float = 0.5,
):
    wav, sr = torchaudio.load(in_wav_path)
    wav = wav.mean(0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

    original_len = wav.shape[-1]
    chunk_samples = int(chunk_sec * SAMPLE_RATE)
    overlap_samples = int(overlap_sec * SAMPLE_RATE)
    hop_samples = chunk_samples - overlap_samples
    overlap_frames = int(overlap_sec * SAMPLE_RATE / HOP_LENGTH)

    chunks = wav_to_chunks(wav, chunk_samples, hop_samples).to(device)  # [N, 1, chunk]

    all_lofi_mels = []
    for i in range(chunks.shape[0]):
        chunk = chunks[i:i+1]  # [1, 1, chunk]
        mel = mel_transform(chunk.squeeze(0))  # [1, 80, T]
        log_mel = torch.log(torch.clamp(mel, min=LOG_MEL_FLOOR))

        T = log_mel.shape[-1]
        if T > N_FRAMES:
            start = (T - N_FRAMES) // 2
            log_mel = log_mel[:, :, start:start + N_FRAMES]
        elif T < N_FRAMES:
            log_mel = torch.nn.functional.pad(log_mel, (0, N_FRAMES - T), value=PAD_VALUE)

        mel_norm = (log_mel - clean_mean) / clean_std  # [1, 80, 432]

        with torch.inference_mode():
            lofi_norm = G(mel_norm.unsqueeze(0)).squeeze(0)  # [1, 80, 432]

        lofi_mel = lofi_norm * lofi_std + lofi_mean
        all_lofi_mels.append(lofi_mel.unsqueeze(0))

    all_lofi_mels = torch.cat(all_lofi_mels, dim=0)  # [N, 1, 80, 432]
    stitched_mel = mel_domain_overlap_add(all_lofi_mels, overlap_frames)  # [1, 80, total_frames]

    max_vocoder_frames = 2000
    total_mel_frames = stitched_mel.shape[-1]
    with torch.inference_mode():
        if total_mel_frames <= max_vocoder_frames:
            wav_out = hifigan(stitched_mel)
        else:
            wav_chunks = []
            for start in range(0, total_mel_frames, max_vocoder_frames):
                end = min(start + max_vocoder_frames, total_mel_frames)
                wav_chunks.append(hifigan(stitched_mel[:, :, start:end]))
            wav_out = torch.cat(wav_chunks, dim=-1)

    wav_out = wav_out.squeeze(0)  # [1, T_wav]
    wav_out = wav_out[:, :original_len]

    peak = wav_out.abs().max()
    if peak > 0:
        wav_out = wav_out / peak * 0.891  # -1 dBFS

    torchaudio.save(out_wav_path, wav_out.cpu(), SAMPLE_RATE)


def main():
    parser = argparse.ArgumentParser(description="Convert audio to lofi style")
    parser.add_argument("--input", required=True, help="Input WAV path")
    parser.add_argument("--output", required=True, help="Output WAV path")
    parser.add_argument("--cyclegan-ckpt", required=True, help="CycleGAN checkpoint .pt")
    parser.add_argument("--hifigan-ckpt", required=True, help="HiFi-GAN checkpoint dir or .pt")
    parser.add_argument("--clean-stats", required=True, help="clean_mel_stats.json")
    parser.add_argument("--lofi-stats", required=True, help="lofi_mel_stats.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--chunk-sec", type=float, default=5.0)
    parser.add_argument("--overlap-sec", type=float, default=0.5)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    G = UNetGenerator().to(device)
    ckpt = torch.load(args.cyclegan_ckpt, map_location=device, weights_only=True)
    if "ema_G" in ckpt:
        G.load_state_dict(ckpt["ema_G"])
    elif "G" in ckpt:
        G.load_state_dict(ckpt["G"])
    else:
        G.load_state_dict(ckpt)
    G.eval()

    hifigan_path = Path(args.hifigan_ckpt)
    hifigan = HiFiGANGenerator().to(device)
    if hifigan_path.is_dir():
        from src.models.hifigan.utils import load_hifigan_generator
        hifigan = load_hifigan_generator(str(hifigan_path), device)
    else:
        hifi_ckpt = torch.load(hifigan_path, map_location=device, weights_only=True)
        if "generator" in hifi_ckpt:
            hifigan.load_state_dict(hifi_ckpt["generator"])
        else:
            hifigan.load_state_dict(hifi_ckpt)
    hifigan.eval()

    clean_mean, clean_std = load_stats(args.clean_stats)
    lofi_mean, lofi_std = load_stats(args.lofi_stats)

    mel_transform = build_mel_transform(device)

    convert_to_lofi(
        args.input, args.output, G, hifigan,
        clean_mean, clean_std, lofi_mean, lofi_std,
        mel_transform, device,
        args.chunk_sec, args.overlap_sec,
    )
    print(f"Saved lofi output to {args.output}")


if __name__ == "__main__":
    main()
