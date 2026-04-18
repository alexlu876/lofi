#!/usr/bin/env python3
"""Compute per-domain mean and std of log-mel spectrograms (train split only).

Outputs: data/stats/{domain}_mel_stats.json with {"mean": float, "std": float}.
Uses Welford's online algorithm to avoid loading all mels into memory.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Compute mel stats")
    parser.add_argument("--domain", required=True, choices=["clean", "lofi"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--split-file", required=True,
                        help="Path to train split .txt (must be train-only to avoid val leakage)")
    args = parser.parse_args()

    base = Path(args.data_dir) if args.data_dir else Path(__file__).resolve().parent.parent / "data"
    mel_dir = base / "mel" / args.domain
    stats_dir = base / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    with open(args.split_file) as f:
        npy_files = [Path(line.strip()) for line in f if line.strip()]

    if not npy_files:
        print(f"No mel files found for {args.domain}", file=sys.stderr)
        sys.exit(1)

    n = 0
    mean = 0.0
    m2 = 0.0

    for npy_path in tqdm(npy_files, desc=f"Stats {args.domain}"):
        mel = np.load(npy_path).astype(np.float64)
        batch_n = mel.size
        batch_mean = mel.mean()
        batch_var = mel.var()
        delta = batch_mean - mean
        tot = n + batch_n
        if tot > 0:
            mean = (n * mean + batch_n * batch_mean) / tot
            m2 += batch_var * batch_n + delta ** 2 * n * batch_n / tot
        n = tot

    std = np.sqrt(m2 / (n - 1)) if n > 1 else 0.0
    stats = {"mean": float(mean), "std": float(std)}

    out_path = stats_dir / f"{args.domain}_mel_stats.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved {out_path}: mean={mean:.4f}, std={std:.4f} (from {n} values across {len(npy_files)} files)")


if __name__ == "__main__":
    main()
