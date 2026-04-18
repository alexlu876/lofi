# Lofi Music Style Transfer — Architecture Proposal

**Target reader:** an experienced ML engineer implementing this directly (via Claude Code).
**Framework:** PyTorch 2.x
**Primary compute:** single NVIDIA A100 (40GB or 80GB) on Vast.ai; local M4 Mac Minis (16GB) for preprocessing.

This document specifies a three-component pipeline:

1. **CycleGAN on mel spectrograms** — unpaired clean↔lofi domain translation. Architecture is a U-Net encoder-decoder generator with a residual bottleneck + PatchGAN discriminator. MelGAN-VC-style tile training on mel spectrograms, trained from scratch.
2. **HiFi-GAN V1 vocoder** — mel→waveform. Initialized from the public `UNIVERSAL_V1` checkpoint, then fine-tuned on the clean-music corpus and (briefly) on the lofi corpus.
3. **Inference pipeline** — WAV → log-mel → CycleGAN G(clean→lofi) → lofi log-mel → HiFi-GAN → WAV.

All tensor shapes, channel widths, kernel sizes, strides, and loss weights are given as exact integers. Deviations should only come from explicit hyperparameter sweeps.

---

## 1. Audio Specification

All audio is normalized to these parameters before any training or inference:

| Parameter        | Value  | Rationale |
|------------------|--------|-----------|
| Sample rate      | 22050  | Matches HiFi-GAN UNIVERSAL_V1; good quality/cost tradeoff for music |
| Bit depth        | 16-bit PCM (float32 in memory, range [-1, 1]) | Standard |
| Channels         | Mono (downmix stereo by averaging) | Simpler model, lofi is often effectively mono anyway |
| Clip length      | 5.0 seconds (110,250 samples) | Long enough for rhythmic context, short enough to batch on A100 |

### Mel Spectrogram Parameters

| Parameter   | Value | Notes |
|-------------|-------|-------|
| `n_fft`     | 1024  | STFT window size |
| `hop_length`| 256   | 22050/256 ≈ 86.13 frames/sec → ~431 frames for 5s |
| `win_length`| 1024  | Hann window |
| `n_mels`    | 80    | Matches HiFi-GAN UNIVERSAL_V1 (non-negotiable) |
| `f_min`     | 0     | Matches HiFi-GAN UNIVERSAL_V1 |
| `f_max`     | 8000  | Matches HiFi-GAN UNIVERSAL_V1 |
| Mel scale   | Slaney-normalized (as in `librosa`, `torchaudio`) | Matches HiFi-GAN UNIVERSAL_V1 |
| Log compression | `log(clamp(mel, min=1e-5))` | Natural log, not log10 |

**Crop to exact length.** Always crop/pad log-mels to **`T=432` frames** (a multiple of 16 to simplify 4-step downsampling in the generator). Pad with the minimum log-mel value (i.e., `log(1e-5) ≈ -11.512`) on the right if short; center-crop if long.

**Normalization.** Compute per-domain mean and std of the log-mel across the training corpus, then:

```
x_norm = (x_log_mel - mean) / std
```

Store `mean` and `std` as `domain_stats.json`. At inference, apply the **clean** domain stats to the input before the CycleGAN, and **de-normalize with the lofi stats** before the vocoder. (The vocoder also gets its own stats; see §5.)

Resulting CycleGAN input tensor: `[B, 1, 80, 432]`.

---

## 2. Data Directory Layout

```
data/
├── raw/
│   ├── clean/                   # source clean-music corpus, any format
│   └── lofi/                    # source lofi corpus, any format
├── wav_22k_mono/
│   ├── clean/*.wav              # resampled, mono, 22050 Hz
│   └── lofi/*.wav
├── clips_5s/
│   ├── clean/
│   │   ├── 000000.wav ...       # 5-second non-overlapping or 2.5s-stride clips
│   └── lofi/
├── mel/
│   ├── clean/
│   │   ├── 000000.npy           # shape [80, 432], float32, log-compressed, UNnormalized
│   └── lofi/
├── stats/
│   ├── clean_mel_stats.json     # {"mean": ..., "std": ...}
│   └── lofi_mel_stats.json
├── splits/
│   ├── clean_train.txt          # one .npy path per line
│   ├── clean_val.txt
│   ├── lofi_train.txt
│   └── lofi_val.txt
└── manifests/
    └── ...                      # any source/licensing tracking
```

Aim for **20–50 hours per domain**, which yields roughly 14k–36k clips at 5s with no overlap. Use a **90/10 train/val split** by *source track*, not by clip, to avoid leakage.

---

## 3. CycleGAN Architecture

Two generators and two discriminators:

- **G: clean → lofi** (the one we actually care about at inference)
- **F: lofi → clean**
- **D_L**: distinguishes real lofi mels from `G(clean)`
- **D_C**: distinguishes real clean mels from `F(lofi)`

All four networks process 2D tensors `[B, 1, 80, 432]`.

### 3.1 Generator (U-Net with 9 residual blocks)

Channel schedule: `1 → 64 → 128 → 256 → 512 → 512` (encoder), 9 × ResBlock(512), `512 → 512 → 256 → 128 → 64 → 1` (decoder).

All convs use reflection padding where possible (i.e., for the stem and head). Use **InstanceNorm2d** throughout (not BatchNorm — critical for small-batch style transfer). Use **LeakyReLU(0.2)** in the encoder and discriminator, **ReLU** in the decoder and residual bottleneck.

**Exact layer spec:**

| # | Layer | In | Out | Kernel | Stride | Padding | Norm | Act | Output shape |
|---|-------|----|----|--------|--------|---------|------|-----|--------------|
| 0 | ReflectionPad2d(3) + Conv2d | 1 | 64 | 7×7 | 1 | 0 | IN | LReLU(0.2) | [B, 64, 80, 432] |
| 1 | Conv2d (Down1) | 64 | 128 | 3×3 | 2 | 1 | IN | LReLU(0.2) | [B, 128, 40, 216] |
| 2 | Conv2d (Down2) | 128 | 256 | 3×3 | 2 | 1 | IN | LReLU(0.2) | [B, 256, 20, 108] |
| 3 | Conv2d (Down3) | 256 | 512 | 3×3 | 2 | 1 | IN | LReLU(0.2) | [B, 512, 10, 54] |
| 4 | Conv2d (Down4) | 512 | 512 | 3×3 | 2 | 1 | IN | LReLU(0.2) | [B, 512, 5, 27] |
| 5 | 9 × ResBlock(512) | 512 | 512 | 3×3 | 1 | 1 | IN | ReLU | [B, 512, 5, 27] |
| 6 | ConvTranspose2d (Up4) + skip concat with (4) | 512 | 512 | 4×4 | 2 | 1 | IN | ReLU | [B, 512+512, 10, 54] |
| 7 | Conv2d (post-concat) | 1024 | 512 | 3×3 | 1 | 1 | IN | ReLU | [B, 512, 10, 54] |
| 8 | ConvTranspose2d (Up3) + skip with (3) | 512 | 256 | 4×4 | 2 | 1 | IN | ReLU | [B, 256+256, 20, 108] |
| 9 | Conv2d (post-concat) | 512 | 256 | 3×3 | 1 | 1 | IN | ReLU | [B, 256, 20, 108] |
| 10 | ConvTranspose2d (Up2) + skip with (2) | 256 | 128 | 4×4 | 2 | 1 | IN | ReLU | [B, 128+128, 40, 216] |
| 11 | Conv2d (post-concat) | 256 | 128 | 3×3 | 1 | 1 | IN | ReLU | [B, 128, 40, 216] |
| 12 | ConvTranspose2d (Up1) + skip with (1) | 128 | 64 | 4×4 | 2 | 1 | IN | ReLU | [B, 64+64, 80, 432] |
| 13 | Conv2d (post-concat) | 128 | 64 | 3×3 | 1 | 1 | IN | ReLU | [B, 64, 80, 432] |
| 14 | ReflectionPad2d(3) + Conv2d | 64 | 1 | 7×7 | 1 | 0 | — | Tanh | [B, 1, 80, 432] |

**ResBlock(C):**
```
x → ReflectionPad2d(1) → Conv2d(C, C, 3, 1, 0) → IN → ReLU
  → ReflectionPad2d(1) → Conv2d(C, C, 3, 1, 0) → IN
  → add input (residual)
```
No activation after the add.

**Why `tanh` at the output:** the network produces normalized log-mels in `~[-3, 3]` during training; `tanh` keeps outputs bounded and stable for the discriminator. When moving to the vocoder, we invert the normalization (multiply by `std`, add `mean`) to get absolute log-mels back.

**Total params (approx):** ~45M per generator (two generators → ~90M).

### 3.2 Discriminator (PatchGAN, 70×70-equivalent)

| # | Layer | In | Out | Kernel | Stride | Padding | Norm | Act | Output shape |
|---|-------|----|----|--------|--------|---------|------|-----|--------------|
| 0 | Conv2d | 1 | 64 | 4×4 | 2 | 1 | — | LReLU(0.2) | [B, 64, 40, 216] |
| 1 | Conv2d | 64 | 128 | 4×4 | 2 | 1 | IN | LReLU(0.2) | [B, 128, 20, 108] |
| 2 | Conv2d | 128 | 256 | 4×4 | 2 | 1 | IN | LReLU(0.2) | [B, 256, 10, 54] |
| 3 | Conv2d | 256 | 512 | 4×4 | 1 | 1 | IN | LReLU(0.2) | [B, 512, 9, 53] |
| 4 | Conv2d | 512 | 1 | 4×4 | 1 | 1 | — | — | [B, 1, 8, 52] |

LSGAN uses raw (linear) outputs — no sigmoid.

**Total params (approx):** ~2.8M per discriminator.

### 3.3 Weight initialization

- Conv / ConvTranspose: Normal(0, 0.02) (as in pix2pix/CycleGAN)
- Bias: 0
- InstanceNorm affine params: weight=Normal(1.0, 0.02), bias=0

---

## 4. CycleGAN Loss Functions

For a real clean mel `c` and real lofi mel `l`:

**LSGAN adversarial losses** (the `(x-1)²` and `x²` form):
```
L_adv(D_L) = E_l[(D_L(l) - 1)²] + E_c[D_L(G(c))²]
L_adv(D_C) = E_c[(D_C(c) - 1)²] + E_l[D_C(F(l))²]

L_adv(G)   = E_c[(D_L(G(c)) - 1)²]
L_adv(F)   = E_l[(D_C(F(l)) - 1)²]
```

**Cycle consistency** (L1):
```
L_cyc = E_c ||F(G(c)) - c||_1  +  E_l ||G(F(l)) - l||_1
```

**Identity loss** (L1 — encourages G to be identity on lofi, F identity on clean):
```
L_id = E_l ||G(l) - l||_1  +  E_c ||F(c) - c||_1
```

**Total generator objective:**
```
L_G_total = L_adv(G) + L_adv(F) + λ_cyc * L_cyc + λ_id * L_id
```

**Lambda weights:**

| Term | λ |
|------|---|
| Adversarial (G, F) | 1.0 |
| Cycle consistency  | **10.0** |
| Identity           | **5.0** |

Train discriminators with `L_adv(D_L)` and `L_adv(D_C)` only. The CycleGAN paper uses a buffer of the **50 most recent generated samples** to update discriminators — implement this (`ImagePool(50)`), it noticeably improves stability.

---

## 5. HiFi-GAN Vocoder — V1 Architecture

We use HiFi-GAN **V1** (the largest/highest-quality variant), initialized from the `UNIVERSAL_V1` checkpoint at https://github.com/jik876/hifi-gan, then fine-tuned on our music corpora.

### 5.1 HiFi-GAN Generator

Input: log-mel `[B, 80, T_mel]` (the **absolute**, un-normalized log-mel — the same scale the checkpoint was trained on; see §5.4).
Output: waveform `[B, 1, T_mel * 256]`.

Upsampling ratios: `[8, 8, 2, 2]`. Product = 256 = hop_length.

Kernel sizes for each transposed conv: `[16, 16, 4, 4]` (each = 2x its stride).

Initial channels `h_u = 512`. Channels halve at each upsample stage: 512 -> 256 -> 128 -> 64 -> 32.

**MRF (Multi-Receptive Field fusion) config for V1:**
- `resblock_kernel_sizes = [3, 7, 11]`
- `resblock_dilation_sizes = [[1,3,5], [1,3,5], [1,3,5]]`
- Use **ResBlock type 1** (the original ResBlock1, not ResBlock2)

**Generator structure (do not deviate — this matches the checkpoint exactly):**

```
Conv1d(80, 512, kernel=7, stride=1, padding=3)
for i in range(4):
    LeakyReLU(0.1)
    ConvTranspose1d(in=512//(2**i), out=512//(2**(i+1)),
                    kernel=ks[i], stride=ss[i], padding=(ks[i]-ss[i])//2)
    # MRF: sum of ResBlock1 outputs over the 3 kernel sizes, then divide by 3
    x = sum(ResBlock1(c, k, d) for k, d in zip([3,7,11], [[1,3,5]]*3)) / 3
LeakyReLU(0.1)
Conv1d(32, 1, kernel=7, stride=1, padding=3)
Tanh
```

where `ks = [16, 16, 4, 4]`, `ss = [8, 8, 2, 2]`.

**ResBlock1(C, k, dilations=[d1,d2,d3]):**
```
for d in dilations:
    x = x + Conv1d(C, C, k, 1, padding=(k-1)*d//2, dilation=d)(LeakyReLU(0.1)(
            Conv1d(C, C, k, 1, padding=(k-1)*d//2, dilation=d)(LeakyReLU(0.1)(x))
        ))
return x
```

Use `weight_norm` on every conv in the generator.

### 5.2 Multi-Period Discriminator (MPD)

Five sub-discriminators for periods `p in {2, 3, 5, 7, 11}`. Each reshapes the 1-D waveform `[B, 1, T]` into `[B, 1, T/p, p]` (pad with zeros if `T` isn't divisible by `p`) and runs a 2-D CNN:

| # | Layer | In | Out | Kernel | Stride | Padding |
|---|-------|----|----|--------|--------|---------|
| 1 | Conv2d | 1 | 32 | (5,1) | (3,1) | (2,0) |
| 2 | Conv2d | 32 | 128 | (5,1) | (3,1) | (2,0) |
| 3 | Conv2d | 128 | 512 | (5,1) | (3,1) | (2,0) |
| 4 | Conv2d | 512 | 1024 | (5,1) | (3,1) | (2,0) |
| 5 | Conv2d | 1024 | 1024 | (5,1) | 1 | (2,0) |
| 6 | Conv2d | 1024 | 1 | (3,1) | 1 | (1,0) |

LeakyReLU(0.1) after layers 1-5. Use `weight_norm` on all convs. Return intermediate feature maps for feature matching.

### 5.3 Multi-Scale Discriminator (MSD)

Three sub-discriminators: at full scale, at 2x avg-pooled, at 4x avg-pooled. The first uses `spectral_norm`; the other two use `weight_norm`. Each sub-discriminator:

| # | Layer | In | Out | Kernel | Stride | Padding | Groups |
|---|-------|----|----|--------|--------|---------|--------|
| 1 | Conv1d | 1 | 128 | 15 | 1 | 7 | 1 |
| 2 | Conv1d | 128 | 128 | 41 | 2 | 20 | 4 |
| 3 | Conv1d | 128 | 256 | 41 | 2 | 20 | 16 |
| 4 | Conv1d | 256 | 512 | 41 | 4 | 20 | 16 |
| 5 | Conv1d | 512 | 1024 | 41 | 4 | 20 | 16 |
| 6 | Conv1d | 1024 | 1024 | 41 | 1 | 20 | 16 |
| 7 | Conv1d | 1024 | 1024 | 5 | 1 | 2 | 1 |
| 8 | Conv1d | 1024 | 1 | 3 | 1 | 1 | 1 |

LeakyReLU(0.1) after layers 1-7. Return intermediate feature maps.

Average pooling for the 2x and 4x branches: `AvgPool1d(kernel=4, stride=2, padding=2)` applied repeatedly (once for 2x, twice for 4x).

### 5.4 HiFi-GAN Loss Functions

Given real waveform `y`, generated `y_hat = G(mel(y))`, and the set of discriminators `D = {MPD_p} union {MSD_s}`:

**LSGAN adversarial:**
```
L_adv(D)  = sum_D [ E_y[(D(y) - 1)^2] + E_y_hat[D(y_hat)^2] ]
L_adv(G)  = sum_D   E_y_hat[(D(y_hat) - 1)^2]
```

**Feature matching:**
```
L_fm(G)   = sum_D sum_i (1/N_i) * || D_i(y) - D_i(y_hat) ||_1
```
where `D_i` is the i-th intermediate feature map and `N_i` is its element count.

**Mel reconstruction (L1 in log-mel space):**
```
L_mel(G)  = || mel(y) - mel(y_hat) ||_1
```
Use the **same mel config** as section 1 for this loss.

**Total generator objective:**
```
L_G_total = L_adv(G) + lambda_fm * L_fm(G) + lambda_mel * L_mel(G)
```

**Lambda weights (from HiFi-GAN paper — do not change on first pass):**

| Term               | lambda |
|--------------------|--------|
| Adversarial        | 1.0    |
| Feature matching   | **2.0**  |
| Mel reconstruction | **45.0** |

---

## 6. Training Procedure

Four sequential phases. Each has an exit criterion; do not proceed until it's met.

### Phase 0 — Environment & Data Preprocessing (local, M4 Macs)

1. `pip install torch torchaudio librosa soundfile numpy tqdm pyyaml tensorboard`. Use PyTorch MPS for any local smoke tests; the actual training is on A100 CUDA.
2. Resample raw -> `data/wav_22k_mono/` using `sox` or `ffmpeg` (`-ar 22050 -ac 1`).
3. Run `scripts/clip.py` — slices each WAV into 5-second clips with **2.5-second hop** (50% overlap) for training, non-overlapping for val. Drop clips whose RMS is below `1e-3` (silence).
4. Run `scripts/compute_mel.py` — computes `[80, 432]` log-mels (un-normalized) and saves as `.npy`.
5. Run `scripts/compute_stats.py` — compute mean/std across the train split only. Save to `data/stats/{clean,lofi}_mel_stats.json`.
6. Run `scripts/make_splits.py` — 90/10 by source track.

**Exit criterion:** >=15 hours of train data per domain, `stats.json` written, sanity-check `.npy` shapes are exactly `[80, 432]`.

### Phase 1 — CycleGAN Training (A100)

**Config:**

| Hyperparameter | Value |
|----------------|-------|
| Batch size     | 16    |
| Optimizer      | Adam  |
| Learning rate  | 2e-4  |
| Betas          | (0.5, 0.999) |
| LR schedule    | Constant for first `N/2` epochs, linear decay to 0 for the second `N/2` |
| Total epochs   | 200   |
| Image pool     | 50    |
| Mixed precision| bf16 (not fp16 — CycleGAN is sensitive to fp16 grad underflow) |
| Grad clipping  | None  |
| EMA on G       | Yes, decay 0.999 (use EMA weights for all inference and FID-style eval) |

**Dataloader:** sample one clean and one lofi mel independently per step (they do not need to be paired). Normalize each using its own domain stats. Apply `torch.nn.functional.pad` only if shapes are off; otherwise no augmentation on the first pass. Once baseline is working, add SpecAugment-light (2 time-masks of width <=16, 2 freq-masks of width <=8) **only on the inputs**, not on cycle targets.

**Step schedule per batch:**
```
1. forward G(c), F(l)
2. forward F(G(c)), G(F(l))  # for cycle
3. forward G(l), F(c)        # for identity
4. compute L_G_total, backward, step opt_G (over params of G and F)
5. forward D_L(l), D_L(G(c).detach() or from pool)
6. forward D_C(c), D_C(F(l).detach() or from pool)
7. compute L_adv(D_L) + L_adv(D_C), backward, step opt_D (over params of D_L and D_C)
```

**Logging (TensorBoard):** losses every 50 steps; sample mel images (input / G(input) / F(G(input))) every 500 steps; audio samples (use a *frozen pretrained* HiFi-GAN for audio previews — don't wait until Phase 2) every 2000 steps.

**Checkpointing:** every 2 epochs + best by val cycle-consistency L1.

**Expected wall-clock on A100:** ~45-60 minutes per epoch at 30k training clips, ~150-200 hours for the full 200 epochs. In practice, stop early at ~120 epochs if val cycle loss plateaus.

**Exit criterion:** audio previews of `G(clean)` clearly sound "lofi-like" and `F(G(clean))` is recognizably close to the original input. There's no FID-equivalent for audio style transfer worth pursuing — judge by ear on a held-out set of 20 clips.

### Phase 2 — HiFi-GAN Fine-Tuning (A100)

**Start from the `UNIVERSAL_V1` checkpoint** (`generator_v1`, `do_v1` for the discriminators + optimizer states). This is trained on mixed data at exactly our mel config, which is the whole point of matching section 1 to their setup.

**Config:**

| Hyperparameter | Value |
|----------------|-------|
| Batch size     | 16    |
| Segment length | 8192 samples (~0.37s) per training step (random crop from each 5s clip) |
| Optimizer      | AdamW |
| Learning rate  | 2e-4  |
| Betas          | (0.8, 0.99) |
| Weight decay   | 0.01  |
| LR schedule    | Exponential decay, gamma = 0.999 per epoch |
| Total steps    | 200k for clean-FT, +50k for lofi-FT |
| Mixed precision| fp32 (HiFi-GAN's `weight_norm` + bf16 can be flaky; default to fp32 on A100) |
| Grad clipping  | None  |

**Fine-tuning data:** mel-waveform pairs derived from the **original real audio** in each domain (no CycleGAN outputs). Each step: load a 5s WAV, random-crop 8192 samples, mel-encode to get the target mel (absolute, un-normalized log-mel), feed the mel to the generator, and compute losses against the ground-truth waveform.

**Two sub-phases:**

1. **Clean-FT (200k steps):** fine-tune on `data/wav_22k_mono/clean/*.wav`. This adapts the vocoder to music specifically (the universal checkpoint is trained on mixed speech + music and often sounds a bit degraded on pure music).
2. **Lofi-FT (50k steps, optional):** branch from the Clean-FT checkpoint and fine-tune on `data/wav_22k_mono/lofi/*.wav`. This gives a vocoder that reconstructs lofi character a little more faithfully. If quality is acceptable without it, skip — using the Clean-FT vocoder for both domains is fine.

**Exit criterion:** on a held-out set of 20 clips, `PESQ(mel-reconstructed, original) >= 3.8` and no audible artifacts on listen-through. Anecdotally the universal checkpoint already passes — fine-tuning usually just nudges quality up.

### Phase 3 — End-to-End Integration

1. Load G (clean->lofi) EMA weights from Phase 1.
2. Load HiFi-GAN generator from Phase 2 (Clean-FT or Lofi-FT).
3. Build `infer.py` per section 7.
4. Run smoke tests on 5 clips that were **not** in either training set.

**Exit criterion:** qualitative listening test on 10 held-out clips sounds convincingly lofi. If not, iterate on Phase 1 (usually more training, sometimes tuning lambda_cyc / lambda_id).

---

## 7. Inference Pipeline

Input: a WAV file (any sample rate, any channels, any length). Output: a lofi WAV at 22050 Hz mono.

```python
def convert_to_lofi(in_wav_path, out_wav_path, G, hifigan,
                    clean_stats, lofi_stats, chunk_sec=5.0, overlap_sec=0.5):
    # 1. Load + normalize audio
    y, sr = torchaudio.load(in_wav_path)
    y = y.mean(0, keepdim=True)                     # mono
    if sr != 22050:
        y = torchaudio.functional.resample(y, sr, 22050)

    # 2. Chunk into overlapping 5s windows (overlap = 0.5s)
    chunk = int(chunk_sec * 22050)                   # 110250
    hop   = chunk - int(overlap_sec * 22050)         # 99225
    chunks = frame(y, chunk, hop)                    # pad last with zeros

    # 3. For each chunk: compute log-mel, normalize with clean stats
    mels = mel_transform(chunks)                     # [N, 80, 432]
    mels_norm = (mels - clean_stats.mean) / clean_stats.std

    # 4. Forward through G
    with torch.inference_mode():
        lofi_norm = G(mels_norm.unsqueeze(1)).squeeze(1)

    # 5. De-normalize with LOFI stats -> absolute log-mel
    lofi_mels = lofi_norm * lofi_stats.std + lofi_stats.mean

    # 6. Vocoder: mel -> waveform (per chunk)
    with torch.inference_mode():
        wavs = hifigan(lofi_mels)

    # 7. Overlap-add with equal-power crossfade in the 0.5s overlap regions
    out = overlap_add(wavs, hop, crossfade='cosine')

    # 8. Trim to original length, peak-normalize to -1 dBFS
    out = out[:, :y.shape[-1]]
    out = out / out.abs().max() * 0.891              # -1 dBFS

    torchaudio.save(out_wav_path, out.cpu(), 22050)
```

**Notes on overlap-add:** chunk boundaries in CycleGAN outputs are a real source of audible seams. Two mitigations:

1. **Mel-domain stitching (preferred)**: resolve the overlap on the log-mels using a cosine crossfade across the overlapping frames (~43 frames for 0.5s), then vocode the stitched mel as one contiguous piece.
2. **Waveform crossfade**: equal-power cosine crossfade on the waveform. Simpler, but can introduce phase cancellation.

Default to option 1.

---

## 8. Hyperparameter Summary

```yaml
audio:
  sample_rate: 22050
  n_fft: 1024
  hop_length: 256
  win_length: 1024
  n_mels: 80
  f_min: 0
  f_max: 8000
  clip_seconds: 5.0
  n_frames: 432

cyclegan:
  generator:
    base_channels: 64
    n_downsample: 4
    bottleneck_resblocks: 9
    norm: instance
    activation_enc: leaky_relu_0.2
    activation_dec: relu
    output_activation: tanh
  discriminator:
    base_channels: 64
    n_layers: 3
    norm: instance
  loss:
    adv_type: lsgan
    lambda_cyc: 10.0
    lambda_id: 5.0
  training:
    batch_size: 16
    optimizer: adam
    lr: 2.0e-4
    betas: [0.5, 0.999]
    total_epochs: 200
    lr_decay_start_epoch: 100
    image_pool_size: 50
    ema_decay: 0.999
    precision: bf16

hifigan:
  variant: V1
  init_from: UNIVERSAL_V1
  upsample_rates: [8, 8, 2, 2]
  upsample_kernel_sizes: [16, 16, 4, 4]
  upsample_initial_channel: 512
  resblock_kernel_sizes: [3, 7, 11]
  resblock_dilation_sizes: [[1,3,5], [1,3,5], [1,3,5]]
  resblock_type: 1
  loss:
    adv_type: lsgan
    lambda_fm: 2.0
    lambda_mel: 45.0
  training:
    batch_size: 16
    segment_size: 8192
    optimizer: adamw
    lr: 2.0e-4
    betas: [0.8, 0.99]
    weight_decay: 0.01
    lr_decay: 0.999
    clean_ft_steps: 200000
    lofi_ft_steps: 50000
    precision: fp32
```

---

## 9. Repository Structure

```
lofi-transfer/
├── configs/
│   └── default.yaml
├── data/                           # see section 2
├── scripts/
│   ├── resample.sh
│   ├── clip.py
│   ├── compute_mel.py
│   ├── compute_stats.py
│   └── make_splits.py
├── src/
│   ├── data/
│   │   ├── mel_dataset.py
│   │   └── wav_dataset.py
│   ├── models/
│   │   ├── cyclegan_generator.py
│   │   ├── patchgan.py
│   │   ├── hifigan/
│   │   │   ├── generator.py
│   │   │   ├── discriminators.py
│   │   │   └── utils.py
│   │   └── ema.py
│   ├── losses/
│   │   ├── cyclegan_losses.py
│   │   └── hifigan_losses.py
│   ├── train_cyclegan.py
│   ├── train_hifigan.py
│   └── infer.py
├── checkpoints/
│   ├── cyclegan/
│   └── hifigan/
├── logs/
│   └── tensorboard/
├── requirements.txt
└── README.md
```

---

## 10. Known Risks and Likely Failure Modes

1. **Mode collapse on the lofi side.** Symptom: `G(c)` for different inputs all sound the same. Fix: confirm lambda_id > 0, check image pool, verify InstanceNorm (not BatchNorm).
2. **Cycle loss dominates; generators become near-identity.** Symptom: `G(c) ~ c` audibly. Fix: reduce lambda_cyc from 10 to 5, or increase adversarial weight to 2.0.
3. **Chunk boundary artifacts at inference.** Do mel-domain stitching before the vocoder. If still present, increase overlap_sec from 0.5 to 1.0.
4. **HiFi-GAN UNIVERSAL_V1 mismatch.** If the checkpoint mel config differs from section 1, fine-tuning will diverge immediately. Verify before Phase 2.
5. **Training corpus contamination.** If the lofi corpus contains ambient noise (rain, cafe sounds), the model will learn to add those sounds. Filter aggressively.
6. **Phase 1 never converges** — usually means the data pipeline normalization is wrong. Sanity-check: normalized mels should be approximately zero-mean, unit-std per batch (roughly [-3, 3] range).

---

## 11. Minimum Viable Milestone

If the scope above is too large for an initial pass:

1. Skip HiFi-GAN fine-tuning; use `UNIVERSAL_V1` as-is.
2. Cap the CycleGAN at 60 epochs with 10 hours of data per domain.
3. Single fixed `chunk_sec=5.0`, no overlap-add (just concatenate — accept seams for now).

This should take ~30 hours of A100 time end-to-end and produce something audibly lofi-ish. Use it as a pipeline sanity check before committing to the full 200-epoch run.

---

**Implementation order:** section 2 -> section 8/9 (set up repo + configs) -> Phase 0 -> Phase 1 -> Phase 2 -> Phase 3.
