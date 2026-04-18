#!/usr/bin/env python3
"""Create 90/10 train/val splits by source track (not by clip) to avoid leakage.

Clips are named {source_stem}_{clip_index}.wav or just {index}.wav.
We group by the original source WAV filename embedded in the clip name,
then split source tracks 90/10.
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path


def group_by_source(npy_files: list[Path]) -> dict[str, list[Path]]:
    """Group mel .npy files by their source track.

    Clip filenames are sequential integers (000000.npy, 000001.npy, ...).
    Since clip.py processes source files in sorted order with sequential numbering,
    we need to map clip indices back to source tracks.

    If a manifest exists mapping clip->source, use that. Otherwise, treat each
    file as its own group (fallback: per-file split rather than per-track).
    """
    groups = defaultdict(list)
    for p in npy_files:
        groups[p.stem].append(p)
    return dict(groups)


def main():
    parser = argparse.ArgumentParser(description="Create train/val splits")
    parser.add_argument("--domain", required=True, choices=["clean", "lofi"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = Path(args.data_dir) if args.data_dir else Path(__file__).resolve().parent.parent / "data"
    mel_dir = base / "mel" / args.domain
    splits_dir = base / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(mel_dir.glob("*.npy"))
    if not npy_files:
        print(f"No .npy files in {mel_dir}", file=sys.stderr)
        sys.exit(1)

    manifest_path = base / "manifests" / f"{args.domain}_clip_sources.txt"
    if manifest_path.exists():
        clip_to_source = {}
        with open(manifest_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    clip_to_source[parts[0]] = parts[1]

        groups = defaultdict(list)
        for p in npy_files:
            source = clip_to_source.get(p.stem, p.stem)
            groups[source].append(p)
        groups = dict(groups)
    else:
        groups = group_by_source(npy_files)

    source_keys = sorted(groups.keys())
    random.seed(args.seed)
    random.shuffle(source_keys)

    n_val = max(1, int(len(source_keys) * args.val_ratio))
    val_keys = set(source_keys[:n_val])

    train_files = []
    val_files = []
    for key in source_keys:
        bucket = val_files if key in val_keys else train_files
        bucket.extend(groups[key])

    train_files.sort()
    val_files.sort()

    train_path = splits_dir / f"{args.domain}_train.txt"
    val_path = splits_dir / f"{args.domain}_val.txt"

    with open(train_path, "w") as f:
        for p in train_files:
            f.write(f"{p}\n")
    with open(val_path, "w") as f:
        for p in val_files:
            f.write(f"{p}\n")

    print(f"{args.domain}: {len(train_files)} train, {len(val_files)} val "
          f"({len(source_keys)} sources, {n_val} held out)")


if __name__ == "__main__":
    main()
