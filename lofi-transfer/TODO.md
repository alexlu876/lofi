# Post-training-launch cleanup + refactor

Accumulated during the CycleGAN → diffusion pivot. Execute after the
lofi_v1 diffusion fine-tune is stable and running on vast.ai.

## Disk reclaim (Mac local)

All regenerable from `data/raw/` if ever needed. Approximate savings ~290 GB.

- [ ] `data/clips_5s/` — 121 GB, CycleGAN-era 5s clip cache; unused by diffusion pipeline
- [ ] `data/wav_22k_mono/` — 61 GB, 22.05 kHz resampled cache; unused by diffusion
- [ ] `data/mel/` — 76 GB, mel-spectrogram cache; unused by diffusion
- [ ] `data/stats/` — 8 KB (negligible), normalization stats; regenerable
- [ ] `checkpoints/cyclegan_backup/best.pt` duplicate — verify md5 vs `checkpoints/cyclegan/best.pt`, delete one

## Repo reorganization

- [ ] Move `src/train_cyclegan.py`, `src/infer.py`, `src/models/cyclegan_generator.py`, `src/models/patchgan.py`, `src/models/hifigan/`, `src/losses/cyclegan_losses.py`, `src/models/ema.py`, `src/data/mel_dataset.py`, `src/data/wav_dataset.py`, `configs/default.yaml`, `configs/local_m5.yaml` into `src/cyclegan_baseline/` — mark as archived baseline, prevent accidental `python src/train_cyclegan.py` invocation
- [ ] Rename `runs/` → `experiments/` (clearer that these are experiment directories, not model weights)
- [ ] Consider moving `scripts/auto_launch.sh` + `scripts/measure_z_std.py` to `scripts/diffusion/` and the CycleGAN-specific scripts (`run_phase0.sh`, the clip/mel/stats/splits trio) to `scripts/cyclegan_baseline/`

## Documentation

- [ ] **Update CLAUDE.md** — currently documents the CycleGAN architecture as if it's active. Add an addendum or rewrite: "Pivoted to diffusion autoencoder approach (Demerlé et al., ISMIR 2024) — see `ctd_repo/`. CycleGAN preserved as baseline for A/B comparison; model at `checkpoints/cyclegan_backup/best.pt`, training code at `src/cyclegan_baseline/`."
- [ ] **Write README.md at project root** — overview, data flow, launch command for the diffusion pipeline, how to run CycleGAN baseline inference for A/B.
- [ ] **Update requirements.txt** — add ctd_repo deps or point to `ctd_repo/requirements.txt`. Right now a fresh clone can't run anything without also knowing to `pip install -r ctd_repo/requirements.txt`.
- [ ] **Write `pretrained/README.md`** — source links (IRCAM/Demerlé bundle), what each file is, license (CC-BY-NC 4.0).

## Git hygiene

- [ ] Commit (then review): `.gitignore` additions (`wav_24k_mono/`, `lmdb/`, `ctd_repo/`, `pretrained/`), new scripts (`scripts/auto_launch.sh`, `scripts/measure_z_std.py`)
- [ ] Decide: track `runs/lofi_v1/config.gin` (the patched config is meaningful project state) but not `runs/lofi_v1/checkpoint*.pt` (weights, too big)
- [ ] `ctd_repo/` is a third-party clone with local patches. Options: (a) add as git submodule pinned to specific commit, (b) keep as untracked dir + maintain a `patches/` dir of `.patch` files we can re-apply to a fresh clone, (c) fork the repo publicly.

## Code-level TODOs identified during audit

- [ ] `split_to_lmdb.py` `DEFINE_string('device')` fix — our local patch. Consider upstream PR to NilsDem/control-transfer-diffusion.
- [ ] `train_diffusion.py` train/val split swap — our local patch. Same as above, upstream worthy.
- [ ] `runs/lofi_v1/config.gin` classifier-channel correction `[32,128,128,128,32]` — deviates from `main.gin`; document in-file comment explaining the pretrained-checkpoint reason.

## Inference pipeline (new code to write post-training)

- [ ] `scripts/infer.py` — takes clean WAV + lofi style reference WAV, outputs styled WAV. Implements chunking + overlap-add for long audio; global CQT normalization; reuses single `zsem` across chunks.
- [ ] `scripts/eval_harness.py` — runs the 6-clean × 5-style-ref eval grid, computes FAD vs held-out val lofi, writes results + audio to the webapp.
