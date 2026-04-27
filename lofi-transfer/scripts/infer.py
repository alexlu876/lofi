#!/usr/bin/env python3
"""Lofi style transfer inference: clean WAV + lofi style ref → styled WAV.

Pipeline:
  1. Load clean + style WAVs, resample to 24 kHz mono.
  2. Chunk clean into 131072-sample windows with 50% overlap.
  3. Compute CQT per chunk, concatenate + global min-max normalize, interpolate to latent frame rate (128 frames).
  4. Encode style ref once → fixed zsem reused for all chunks.
  5. Draw one seeded noise tensor spanning the full output latent; slice per chunk so overlap regions share noise → cleaner crossfade.
  6. Heun 40-step diffusion sampling per chunk with CFG=2.0 on time_cond stream.
  7. Overlap-add chunk latents via triangular partition-of-unity window → single full latent.
  8. Decode ONCE through the AE on the full latent (avoids per-chunk conv boundary artifacts).
  9. Trim to original clean length, peak-normalize, save as WAV.

Usage:
  python infer.py --clean clean.wav --style lofi_ref.wav \
      --out styled.wav --ckpt /path/to/checkpoint491668.pt \
      --config /path/to/config.gin --ae /path/to/AE_real_instruments.pt

Notes:
  - Prefers checkpoint{step}_EMA.pt over checkpoint{step}.pt when available.
  - `--device` auto-detects cuda > mps > cpu.
  - Uses global CQT normalization (different from the notebook's per-clip normalize).
"""
import argparse
import math
import sys
from pathlib import Path

import gin
import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from cqt_pytorch import CQT

# === constants that match training / paper config ===
SR = 24000
X_LENGTH = 131072         # samples per diffusion window (5.46 s)
AE_RATIO = 1024           # AE compression: 1024 samples → 1 latent frame
Z_LENGTH = X_LENGTH // AE_RATIO  # 128 latent frames per window
CQT_OCT = 8
CQT_BPO = 32              # bins per octave → total 256 bins
N_HEUN_STEPS = 40
GUIDANCE = 2.0
GUIDANCE_TYPE = "time_cond"  # CFG on content only, not style
CFG_EPS = 1e-4            # for min-max normalize denominator


def pick_device(requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def load_wav_mono_24k(path, peak_normalize=True):
    """Load + mono-downmix + resample to 24 kHz. If peak_normalize, scale to peak=1.0 (matches AE training distribution)."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    if peak_normalize:
        peak = wav.abs().max()
        # guard against silence; notebook's load_audio has the same bug without guard
        wav = wav / max(peak.item(), 1e-8)
    return wav  # shape [1, T], float32 in ~[-1, 1]


def load_model(config_path, ckpt_path, ae_path, device):
    """Instantiate EDM_ADV from gin config, load weights (prefer EMA), attach frozen AE."""
    # ensure repo is importable
    repo = Path(__file__).resolve().parent.parent / "ctd_repo"
    sys.path.insert(0, str(repo))

    gin.parse_config_file(config_path)
    from diffusion.model import EDM_ADV  # noqa: E402

    ae = torch.jit.load(ae_path, map_location=device).eval()

    model = EDM_ADV()
    model.emb_model = ae  # attach so model.emb_model.encode/decode work

    # Prefer EMA ckpt
    ema_path = Path(str(ckpt_path).replace(".pt", "_EMA.pt"))
    which = "EMA" if ema_path.exists() else "raw"
    path = ema_path if ema_path.exists() else Path(ckpt_path)
    print(f"[model] loading {which} weights: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    # File format differs: EMA ckpts often are plain state_dict; non-EMA wraps in model_state
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[model] loaded with {len(missing)} missing, {len(unexpected)} unexpected keys")
    model.to(device).eval()
    return model


def chunk_with_overlap(wav, chunk_len=X_LENGTH, hop_len=X_LENGTH // 2):
    """Split [1, T] waveform into [N, 1, chunk_len] with 50%-overlap zero-padded last chunk.

    Returns (chunks_tensor, n_chunks, total_chunks_len_samples) where total_chunks_len_samples
    is the effective length after padding (= hop_len * (N-1) + chunk_len).
    """
    T = wav.shape[-1]
    if T <= chunk_len:
        # Single chunk, zero-pad to chunk_len
        padded = F.pad(wav, (0, chunk_len - T))
        return padded.unsqueeze(0), 1, chunk_len  # [1, 1, chunk_len]

    # Stride chunks, last chunk may extend past audio end → zero-pad it
    n_full = (T - chunk_len) // hop_len + 1
    last_start = (n_full - 1) * hop_len
    if last_start + chunk_len < T:
        # Need one more chunk that extends past T
        n_chunks = n_full + 1
    else:
        n_chunks = n_full

    total_len = (n_chunks - 1) * hop_len + chunk_len
    padded = F.pad(wav, (0, max(0, total_len - T)))
    chunks = []
    for i in range(n_chunks):
        start = i * hop_len
        chunks.append(padded[:, start:start + chunk_len])
    return torch.stack(chunks, dim=0), n_chunks, total_len


def compute_time_conds_global_norm(clean_wav, device, cqt_module, model):
    """Compute per-chunk time_cond inputs from clean audio, with GLOBAL CQT normalization.

    Flow:  waveform chunk → CQT (256 bins) → log1p(|·|²) → global min-max → interp to Z_LENGTH
           → encoder_time (structure encoder) → per-chunk time_cond [1, ZS_CHANNELS=16, Z_LENGTH]

    Critical: per-clip normalization (what the notebook does) causes content-encoder drift
    at chunk boundaries. For long audio we need global min/max across the whole source.

    Returns list of per-chunk time_cond tensors [1, 16, Z_LENGTH], count n_chunks, total_len.
    """
    chunks, n_chunks, total_len = chunk_with_overlap(clean_wav, X_LENGTH, X_LENGTH // 2)
    chunks = chunks.to(device)  # [N, 1, X_LENGTH]

    # CQT per chunk (cqt_pytorch is fixed-length)
    cqt_mags = []
    for i in range(n_chunks):
        c = cqt_module.encode(chunks[i:i+1])  # [1, 1, 256, T_cqt]
        mag = torch.abs(c).squeeze(1)  # [1, 256, T_cqt]
        cqt_mags.append(mag)

    # Log-compress then concatenate along time for global normalization
    log_cqt_chunks = [torch.log1p(m ** 2) for m in cqt_mags]
    # Global min/max over all chunks
    global_min = min(lc.min() for lc in log_cqt_chunks)
    global_max = max(lc.max() for lc in log_cqt_chunks)

    # Per-chunk: normalize with global, interpolate to Z_LENGTH frames, then encoder_time
    time_conds = []
    with torch.no_grad():
        for lc in log_cqt_chunks:
            normed = (lc - global_min) / (global_max - global_min + CFG_EPS)
            interp = F.interpolate(normed, size=Z_LENGTH, mode="nearest")  # [1, 256, 128]
            tc = model.encoder_time(interp)  # [1, 16, 128]  via structure encoder
            time_conds.append(tc)

    return time_conds, n_chunks, total_len


def compute_zsem(style_wav, model, device):
    """Encode style reference into a single [1, ZT_CHANNELS] vector, reused across all chunks."""
    T = style_wav.shape[-1]
    if T > X_LENGTH:
        # Use the first X_LENGTH samples; could average a few windows but that changes behavior from paper
        style_wav = style_wav[:, :X_LENGTH]
    elif T < X_LENGTH:
        style_wav = F.pad(style_wav, (0, X_LENGTH - T))
    style_wav = style_wav.unsqueeze(0).to(device)  # [1, 1, X_LENGTH]

    with torch.no_grad():
        z_style = model.emb_model.encode(style_wav)  # [1, AE_EMBSIZE, Z_LENGTH]
        zsem = model.encoder(z_style)                # [1, ZT_CHANNELS]
    return zsem


def sample_per_chunk(model, time_conds, zsem, n_chunks, total_latent_frames, device, seed, guidance=GUIDANCE):
    """Run diffusion per chunk, using a single seeded noise tensor sliced per chunk.

    Each chunk i uses noise[ : , : , i*Z_LENGTH/2 : i*Z_LENGTH/2 + Z_LENGTH ]. Adjacent
    chunks share 50% of their input noise → denoising trajectories in overlap region
    are more similar → smoother crossfade.
    """
    torch.manual_seed(seed)
    full_noise = torch.randn(1, 32, total_latent_frames, device=device)

    hop_frames = Z_LENGTH // 2  # 64
    latents = []
    for i in range(n_chunks):
        start = i * hop_frames
        x0 = full_noise[..., start:start + Z_LENGTH]  # [1, 32, 128]
        with torch.no_grad():
            xS = model.sample(
                x0,
                time_cond=time_conds[i],
                zsem=zsem,
                nb_step=N_HEUN_STEPS,
                guidance=guidance,
                guidance_type=GUIDANCE_TYPE,
            )
        latents.append(xS)
        print(f"[sample] chunk {i+1}/{n_chunks} complete")
    return latents


def triangular_window(length):
    """Partition-of-unity window: shifted-by-half sum = exactly 1.0.

    Asymmetric: w[0]=0, w[half-1]=(half-1)/half, w[half]=1.0, w[length-1]=1/half.
    POU proof:  w[i] + w[i+half] = i/half + (half-i)/half = 1   for all i in [0, half).
    """
    half = length // 2
    ramp_up = torch.linspace(0, 1, half + 1)[:-1]         # [0, 1/half, ..., (half-1)/half]
    ramp_down = torch.linspace(1, 0, half + 1)[:-1]       # [1, (half-1)/half, ..., 1/half]
    return torch.cat([ramp_up, ramp_down])                # length=length


def overlap_add_latents(latents, n_chunks, total_latent_frames, device):
    """Combine chunk latents into a single [1, 32, total_latent_frames] tensor via triangular crossfade.

    Weights sum to 1 in the interior; edge regions (first/last Z_LENGTH/2 frames of overall
    output) are covered by only one chunk and normalize correctly via the weights tensor.
    """
    hop_frames = Z_LENGTH // 2
    full_latent = torch.zeros(1, 32, total_latent_frames, device=device)
    weights = torch.zeros(1, 1, total_latent_frames, device=device)
    window = triangular_window(Z_LENGTH).to(device).view(1, 1, Z_LENGTH)

    for i in range(n_chunks):
        start = i * hop_frames
        end = start + Z_LENGTH
        if i == 0:
            # First chunk: full weight on leading half (no prior overlap)
            w = window.clone()
            w[..., :hop_frames] = 1.0
        elif i == n_chunks - 1:
            # Last chunk: full weight on trailing half (no subsequent overlap)
            w = window.clone()
            w[..., hop_frames:] = 1.0
        else:
            w = window
        full_latent[..., start:end] += latents[i] * w
        weights[..., start:end] += w

    # Normalize (weights should be ~1 everywhere, but safety)
    full_latent = full_latent / weights.clamp(min=1e-6)
    return full_latent


def run(args):
    device = pick_device(args.device)
    print(f"[device] {device}")

    model = load_model(args.config, args.ckpt, args.ae, device)

    # CQT module — fixed at X_LENGTH
    cqt_module = CQT(
        num_octaves=CQT_OCT,
        num_bins_per_octave=CQT_BPO,
        sample_rate=SR,
        block_length=X_LENGTH,
    ).to(device)

    # Load inputs
    print(f"[input] clean: {args.clean}")
    clean = load_wav_mono_24k(args.clean)
    orig_samples = clean.shape[-1]
    print(f"[input] clean duration: {orig_samples / SR:.2f}s, {orig_samples} samples @ 24 kHz")

    print(f"[input] style: {args.style}")
    style = load_wav_mono_24k(args.style)
    print(f"[input] style duration: {style.shape[-1] / SR:.2f}s (using first {X_LENGTH} samples for zsem)")

    # Compute content conditioning
    print("[cqt ] computing per-chunk time_conds with global normalization...")
    time_conds, n_chunks, total_len = compute_time_conds_global_norm(clean, device, cqt_module, model)
    total_latent_frames = total_len // AE_RATIO
    print(f"[cqt ] {n_chunks} chunks, total_samples={total_len}, total_latent_frames={total_latent_frames}")

    # Compute style conditioning (once)
    print("[style] encoding style reference → zsem...")
    zsem = compute_zsem(style, model, device)
    print(f"[style] zsem shape: {zsem.shape}")

    # Diffusion sampling per chunk
    guidance = getattr(args, "guidance", GUIDANCE)
    print(f"[diff] running {n_chunks} chunks × {N_HEUN_STEPS} Heun steps (guidance={guidance})...")
    latents = sample_per_chunk(model, time_conds, zsem, n_chunks, total_latent_frames, device, args.seed, guidance=guidance)

    # Overlap-add stitching
    print("[stitch] triangular crossfade over 50% overlap regions...")
    full_latent = overlap_add_latents(latents, n_chunks, total_latent_frames, device)
    print(f"[stitch] stitched latent: {full_latent.shape}")

    # AE decode in one pass (conv receptive field handles boundaries)
    print("[decode] AE.decode on full latent...")
    with torch.no_grad():
        wav_out = model.emb_model.decode(full_latent)  # [1, 1, total_len]
    print(f"[decode] output: {wav_out.shape}")

    # Trim to original clean length
    wav_out = wav_out[..., :orig_samples]

    # Short linear fade-out on the last 20 ms — the final chunk was partly denoised from
    # zero-padded CQT input and can have brief tail artifacts; fade-out hides any seam.
    fade_samples = min(int(0.02 * SR), wav_out.shape[-1])  # 20 ms
    if fade_samples > 0:
        fade = torch.linspace(1.0, 0.0, fade_samples, device=wav_out.device)
        wav_out[..., -fade_samples:] = wav_out[..., -fade_samples:] * fade

    # Peak-normalize to -1 dBFS
    peak = wav_out.abs().max()
    if peak > 0:
        wav_out = wav_out / peak * 0.891
    else:
        print("[warn] output peak is 0 (silent)")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), wav_out.squeeze(0).cpu(), SR)
    print(f"[save] wrote {out_path} ({wav_out.shape[-1] / SR:.2f}s, {SR} Hz, mono)")


def main():
    p = argparse.ArgumentParser(description="Lofi style-transfer inference (clean + style ref → styled WAV)")
    p.add_argument("--clean", required=True, help="clean source audio WAV (any SR/channels)")
    p.add_argument("--style", required=True, help="lofi style reference WAV (any SR/channels)")
    p.add_argument("--out", required=True, help="output WAV path")
    p.add_argument("--ckpt", required=True, help="path to checkpoint{step}.pt (EMA version used if _EMA.pt exists)")
    p.add_argument("--config", required=True, help="path to config.gin for the training run")
    p.add_argument("--ae", required=True, help="path to AE torchscript (AE_real_instruments.pt)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for diffusion noise")
    p.add_argument("--guidance", type=float, default=GUIDANCE, help="Classifier-free guidance strength (default 2.0; >1 pushes harder toward style)")
    p.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
