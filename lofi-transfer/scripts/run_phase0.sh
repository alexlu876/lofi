#!/usr/bin/env bash
set -uo pipefail
# Note: not using `set -e` — we handle errors per-step with explicit checks

# Phase 0: Full preprocessing pipeline with validation
# Run from the lofi-transfer/ directory:
#   caffeinate -s bash scripts/run_phase0.sh 2>&1 | tee logs/phase0.log

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$BASE_DIR"

# Use venv python directly (source activate breaks under set -u)
PYTHON="$BASE_DIR/../lofi/bin/python3"
export PATH="/opt/homebrew/bin:$BASE_DIR/../lofi/bin:$PATH"

if [ ! -x "$PYTHON" ]; then
    echo "FATAL: venv python not found at $PYTHON"
    exit 1
fi

echo "Using Python: $PYTHON"
echo "Python version: $($PYTHON --version)"

mkdir -p logs

FAIL=0
warn() { echo "WARNING: $1"; }
fail() { echo "FATAL: $1"; FAIL=1; }
check_min_files() {
    local dir="$1" min="$2" label="$3"
    local count
    count=$(find "$dir" -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')
    if [ "$count" -lt "$min" ]; then
        fail "$label: only $count files (expected >= $min)"
        return 1
    fi
    echo "  CHECK PASSED: $label has $count files (>= $min)"
    return 0
}

echo "============================================"
echo "Phase 0: Data Preprocessing Pipeline"
echo "Started: $(date)"
echo "============================================"

# =============================================
# Step 1: Strip vocals from tracks that need it
# =============================================
echo ""
echo "[Step 1/6] Separating vocal vs instrumental tracks, then running Demucs..."

# Split clean files into vocal (needs demucs) and instrumental (copy directly)
mkdir -p data/raw/clean_vocal_only data/raw/clean_instrumental data/raw/clean_merged

INSTRUMENTAL_KEYWORDS="pokemon|pok_|_ost_|zelda|mario|hollow_knight|celeste|undertale|stardew|minecraft|animal_cross|chrono|persona|nier|ori_and|journey_game|hades|cuphead|dark_souls|skyrim|fire_emblem|kirby|mega_man|metroid|splatoon|final_fantasy|kingdom_hearts|sonic_racing|nintendo|champion_pokemon|gym_leader|route_|rival_battle"

n_vocal=0
n_instrumental_skip=0
for f in data/raw/clean/*.wav; do
    [ -f "$f" ] || continue
    base="$(basename "$f")"
    name_lower="$(echo "$base" | tr '[:upper:]' '[:lower:]')"
    if echo "$name_lower" | grep -qE "$INSTRUMENTAL_KEYWORDS"; then
        # Instrumental — copy directly to merged, skip demucs
        if [ ! -f "data/raw/clean_merged/$base" ]; then
            cp "$f" "data/raw/clean_merged/$base"
        fi
        n_instrumental_skip=$((n_instrumental_skip + 1))
    else
        # Has vocals — stage for demucs
        if [ ! -f "data/raw/clean_vocal_only/$base" ]; then
            cp "$f" "data/raw/clean_vocal_only/$base"
        fi
        n_vocal=$((n_vocal + 1))
    fi
done
echo "  Instrumental (copied directly): $n_instrumental_skip"
echo "  Vocal (need Demucs): $n_vocal"

# Run Demucs only on vocal tracks
echo "  Running Demucs on vocal tracks..."
$PYTHON scripts/strip_vocals.py \
    --src-dir data/raw/clean_vocal_only \
    --out-dir data/raw/clean_instrumental \
    --model htdemucs \
    --device cpu \
    --workers 1

# --- Validation 1 ---
echo ""
echo "[Validate 1] Checking vocal stripping output..."
n_demucs_out=$(find data/raw/clean_instrumental -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')
echo "  Demucs output files: $n_demucs_out"

# Merge: instrumental tracks already copied; now add demucs output, then originals as fallback
if [ "$n_demucs_out" -gt 0 ]; then
    for f in data/raw/clean_instrumental/*.wav; do
        [ -f "$f" ] || continue
        base="$(basename "$f")"
        if [ ! -f "data/raw/clean_merged/$base" ]; then
            cp "$f" "data/raw/clean_merged/$base"
        fi
    done
fi
# Fallback: any remaining files not yet in merged
for f in data/raw/clean/*.wav; do
    [ -f "$f" ] || continue
    base="$(basename "$f")"
    if [ ! -f "data/raw/clean_merged/$base" ]; then
        cp "$f" "data/raw/clean_merged/$base"
    fi
done
n_merged=$(find data/raw/clean_merged -name '*.wav' | wc -l | tr -d ' ')
echo "  Merged total: $n_merged files"

# =============================================
# Step 2: Resample to 22050 Hz mono
# =============================================
echo ""
echo "[Step 2/6] Resampling to 22050 Hz mono..."

for domain in clean lofi; do
    if [ "$domain" = "clean" ]; then
        src="data/raw/clean_merged"
    else
        src="data/raw/lofi"
    fi
    dst="data/wav_22k_mono/$domain"
    mkdir -p "$dst"
    echo "  Resampling $domain..."
    count=0
    for f in "$src"/*.wav; do
        [ -f "$f" ] || continue
        base="$(basename "$f")"
        out="$dst/$base"
        if [ -f "$out" ]; then continue; fi
        if ffmpeg -y -hide_banner -loglevel error -i "$f" -ar 22050 -ac 1 -sample_fmt s16 "$out"; then
            count=$((count + 1))
        else
            warn "ffmpeg failed on $f"
            rm -f "$out"
        fi
    done
    echo "  $domain: resampled $count new files"
done

# --- Validation 2 ---
echo ""
echo "[Validate 2] Checking resampled files..."
for domain in clean lofi; do
    check_min_files "data/wav_22k_mono/$domain" 10 "wav_22k_mono/$domain"
done
$PYTHON -c "
import soundfile as sf
from pathlib import Path
for domain in ['clean', 'lofi']:
    wavs = list(Path(f'data/wav_22k_mono/{domain}').glob('*.wav'))
    bad = 0
    total_sec = 0
    for w in wavs[:5]:
        info = sf.info(w)
        if info.samplerate != 22050:
            print(f'  ERROR: {w.name} has sr={info.samplerate}, expected 22050')
            bad += 1
        if info.channels != 1:
            print(f'  ERROR: {w.name} has channels={info.channels}, expected 1')
            bad += 1
    for w in wavs:
        total_sec += sf.info(w).duration
    print(f'  {domain}: {len(wavs)} files, {total_sec/3600:.1f} hrs, spot-check errors: {bad}')
"
if [ $? -ne 0 ]; then fail "Resampled file validation failed"; fi

# =============================================
# Step 3: Clip into 5-second segments
# =============================================
echo ""
echo "[Step 3/6] Clipping into 5-second segments..."
$PYTHON scripts/clip.py --domain clean
$PYTHON scripts/clip.py --domain lofi

# --- Validation 3 ---
echo ""
echo "[Validate 3] Checking clips and manifests..."
for domain in clean lofi; do
    check_min_files "data/clips_5s/$domain" 100 "clips_5s/$domain"
    manifest="data/manifests/${domain}_clip_sources.txt"
    if [ -f "$manifest" ]; then
        n_manifest=$(wc -l < "$manifest" | tr -d ' ')
        n_clips=$(find "data/clips_5s/$domain" -name '*.wav' | wc -l | tr -d ' ')
        echo "  Manifest entries: $n_manifest, Clip files: $n_clips"
        if [ "$n_manifest" -ne "$n_clips" ]; then
            warn "$domain: manifest ($n_manifest) != clip count ($n_clips)"
        fi
    else
        fail "$domain: manifest not found at $manifest"
    fi
done

$PYTHON -c "
import soundfile as sf
from pathlib import Path
import numpy as np
for domain in ['clean', 'lofi']:
    clips = list(Path(f'data/clips_5s/{domain}').glob('*.wav'))
    bad = 0
    for c in clips[:20]:
        info = sf.info(c)
        if abs(info.duration - 5.0) > 0.1:
            print(f'  ERROR: {c.name} duration={info.duration:.2f}s, expected ~5.0s')
            bad += 1
        if info.samplerate != 22050:
            bad += 1
    print(f'  {domain}: {len(clips)} clips, spot-check errors: {bad}')
"

# =============================================
# Step 4: Compute mel spectrograms
# =============================================
echo ""
echo "[Step 4/6] Computing mel spectrograms..."
$PYTHON scripts/compute_mel.py --domain clean --device mps
$PYTHON scripts/compute_mel.py --domain lofi --device mps

# --- Validation 4 ---
echo ""
echo "[Validate 4] Checking mel spectrograms..."
$PYTHON -c "
import numpy as np
from pathlib import Path
for domain in ['clean', 'lofi']:
    mels = list(Path(f'data/mel/{domain}').glob('*.npy'))
    bad = 0
    for m in mels[:20]:
        arr = np.load(m)
        if arr.shape != (80, 432):
            print(f'  ERROR: {m.name} shape={arr.shape}, expected (80, 432)')
            bad += 1
        if arr.dtype != np.float32:
            print(f'  ERROR: {m.name} dtype={arr.dtype}, expected float32')
            bad += 1
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            print(f'  ERROR: {m.name} contains NaN/Inf')
            bad += 1
    print(f'  {domain}: {len(mels)} mels, spot-check errors: {bad}')
    if bad > 0:
        exit(1)
"
if [ $? -ne 0 ]; then fail "Mel spectrogram validation failed"; fi

# =============================================
# Step 5: Create train/val splits
# =============================================
echo ""
echo "[Step 5/6] Creating train/val splits..."
$PYTHON scripts/make_splits.py --domain clean
$PYTHON scripts/make_splits.py --domain lofi

# --- Validation 5 ---
echo ""
echo "[Validate 5] Checking splits..."
for domain in clean lofi; do
    train="data/splits/${domain}_train.txt"
    val="data/splits/${domain}_val.txt"
    if [ ! -f "$train" ] || [ ! -f "$val" ]; then
        fail "Split files missing for $domain"
        continue
    fi
    n_train=$(wc -l < "$train" | tr -d ' ')
    n_val=$(wc -l < "$val" | tr -d ' ')
    n_total=$((n_train + n_val))
    pct_val=$((n_val * 100 / n_total))
    echo "  $domain: $n_train train, $n_val val ($pct_val% val)"
    if [ "$pct_val" -lt 5 ] || [ "$pct_val" -gt 20 ]; then
        warn "$domain: val split is ${pct_val}%, expected ~10%"
    fi
    # Verify all paths in split files exist
    n_missing=$(while read -r p; do [ -f "$p" ] || echo missing; done < "$train" | wc -l | tr -d ' ')
    if [ "$n_missing" -gt 0 ]; then
        fail "$domain train split has $n_missing missing files"
    fi
done

# =============================================
# Step 6: Compute normalization stats
# =============================================
echo ""
echo "[Step 6/6] Computing normalization stats..."
$PYTHON scripts/compute_stats.py --domain clean --split-file data/splits/clean_train.txt
$PYTHON scripts/compute_stats.py --domain lofi --split-file data/splits/lofi_train.txt

# --- Validation 6 ---
echo ""
echo "[Validate 6] Checking stats..."
$PYTHON -c "
import json
for domain in ['clean', 'lofi']:
    path = f'data/stats/{domain}_mel_stats.json'
    with open(path) as f:
        stats = json.load(f)
    mean, std = stats['mean'], stats['std']
    print(f'  {domain}: mean={mean:.4f}, std={std:.4f}')
    if abs(mean) > 20:
        print(f'  WARNING: {domain} mean looks unusual (expected roughly -5 to -8)')
    if std < 0.5 or std > 10:
        print(f'  WARNING: {domain} std looks unusual (expected roughly 1-5)')
"

# =============================================
# Final Summary
# =============================================
echo ""
echo "============================================"
echo "Phase 0 Complete: $(date)"
echo "============================================"
echo ""

for domain in clean lofi; do
    n_train=$(wc -l < "data/splits/${domain}_train.txt" | tr -d ' ')
    n_val=$(wc -l < "data/splits/${domain}_val.txt" | tr -d ' ')
    echo "$domain: $n_train train clips, $n_val val clips"
    cat "data/stats/${domain}_mel_stats.json"
    echo ""
done

if [ "$FAIL" -ne 0 ]; then
    echo ""
    echo "*** THERE WERE FAILURES — review the log above ***"
    exit 1
fi

echo ""
echo "All validations passed. Ready for Phase 1 (CycleGAN training)."
