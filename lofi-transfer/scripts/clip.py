#!/usr/bin/env python3
"""Slice WAVs into 5-second clips.

Training clips use 2.5s hop (50% overlap).
Clips with RMS below 1e-3 are dropped (silence).
Also writes a manifest mapping clip index -> source track for split-by-track.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

SAMPLE_RATE = 22050
CLIP_SAMPLES = int(5.0 * SAMPLE_RATE)  # 110250
TRAIN_HOP = int(2.5 * SAMPLE_RATE)     # 55125
RMS_THRESHOLD = 1e-3


def clip_file(wav_path: Path, out_dir: Path, hop: int, start_idx: int) -> tuple[int, list[tuple[str, str]]]:
    audio, sr = sf.read(wav_path, dtype="float32")
    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr} for {wav_path}"
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    written = 0
    manifest_entries = []
    pos = 0
    while pos + CLIP_SAMPLES <= len(audio):
        clip = audio[pos : pos + CLIP_SAMPLES]
        rms = np.sqrt(np.mean(clip ** 2))
        if rms >= RMS_THRESHOLD:
            clip_name = f"{start_idx + written:06d}"
            out_path = out_dir / f"{clip_name}.wav"
            sf.write(out_path, clip, SAMPLE_RATE, subtype="PCM_16")
            manifest_entries.append((clip_name, wav_path.stem))
            written += 1
        pos += hop

    return written, manifest_entries


def main():
    parser = argparse.ArgumentParser(description="Clip WAVs into 5-second segments")
    parser.add_argument("--domain", required=True, choices=["clean", "lofi"])
    parser.add_argument("--data-dir", default=None, help="Base data directory")
    args = parser.parse_args()

    base = Path(args.data_dir) if args.data_dir else Path(__file__).resolve().parent.parent / "data"
    src_dir = base / "wav_22k_mono" / args.domain
    out_dir = base / "clips_5s" / args.domain
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = base / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        print(f"Source directory not found: {src_dir}", file=sys.stderr)
        sys.exit(1)

    wav_files = sorted(src_dir.glob("*.wav"))
    if not wav_files:
        print(f"No WAV files found in {src_dir}", file=sys.stderr)
        sys.exit(1)

    total = 0
    all_manifest = []
    for wav_path in tqdm(wav_files, desc=f"Clipping {args.domain}"):
        written, entries = clip_file(wav_path, out_dir, TRAIN_HOP, start_idx=total)
        all_manifest.extend(entries)
        total += written

    manifest_path = manifest_dir / f"{args.domain}_clip_sources.txt"
    with open(manifest_path, "w") as f:
        for clip_name, source_name in all_manifest:
            f.write(f"{clip_name}\t{source_name}\n")

    print(f"Wrote {total} clips to {out_dir}")
    print(f"Manifest: {manifest_path}")
    duration_hrs = total * 5.0 / 3600
    print(f"Total duration: {duration_hrs:.1f} hours")


if __name__ == "__main__":
    main()
