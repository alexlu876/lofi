#!/usr/bin/env bash
set -euo pipefail

# Setup script for A100 on Vast.ai
# Run this AFTER transferring the data:
#   bash scripts/setup_remote.sh

echo "=== Setting up training environment ==="

# Install Python deps
pip install torch torchaudio numpy pyyaml tensorboard tqdm

# Create directory structure
mkdir -p checkpoints/cyclegan checkpoints/hifigan logs/tensorboard/cyclegan

# Fix split file paths (rewrite from Mac absolute paths to current directory)
echo "Fixing split file paths..."
for f in data/splits/*.txt; do
    # Replace the Mac-specific prefix with the current working directory
    sed -i "s|/Users/alex/lu/git/lofi/lofi-transfer/||g" "$f"
    # Prepend current directory
    WORKDIR="$(pwd)"
    sed -i "s|^data/|${WORKDIR}/data/|g" "$f"
    echo "  Fixed: $f ($(wc -l < "$f" | tr -d ' ') entries)"
done

# Verify data
echo ""
echo "=== Verifying data ==="
python3 -c "
import numpy as np
from pathlib import Path

for domain in ['clean', 'lofi']:
    mels = list(Path(f'data/mel/{domain}').glob('*.npy'))
    print(f'{domain}: {len(mels)} mel files')
    # Spot check
    m = np.load(mels[0])
    print(f'  shape={m.shape}, dtype={m.dtype}')

    # Check split files
    train = open(f'data/splits/{domain}_train.txt').readlines()
    val = open(f'data/splits/{domain}_val.txt').readlines()
    print(f'  train={len(train)}, val={len(val)}')

    # Verify first path exists
    first = train[0].strip()
    assert Path(first).exists(), f'Split path not found: {first}'
    print(f'  paths OK')

import json
for domain in ['clean', 'lofi']:
    with open(f'data/stats/{domain}_mel_stats.json') as f:
        s = json.load(f)
    print(f'{domain} stats: mean={s[\"mean\"]:.4f}, std={s[\"std\"]:.4f}')

print()
print('All checks passed!')
"

echo ""
echo "=== Ready to train ==="
echo "Run:"
echo "  python3 src/train_cyclegan.py --config configs/default.yaml --data-dir data --log-dir logs/tensorboard/cyclegan --ckpt-dir checkpoints/cyclegan"
