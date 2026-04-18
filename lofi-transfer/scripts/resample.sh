#!/usr/bin/env bash
set -euo pipefail

# Resample all audio in data/raw/{clean,lofi}/ to 22050 Hz mono WAV
# Output goes to data/wav_22k_mono/{clean,lofi}/

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

for domain in clean lofi; do
    src="$BASE_DIR/data/raw/$domain"
    dst="$BASE_DIR/data/wav_22k_mono/$domain"
    mkdir -p "$dst"

    if [ ! -d "$src" ] || [ -z "$(ls -A "$src" 2>/dev/null)" ]; then
        echo "Skipping $domain: no files in $src"
        continue
    fi

    echo "Resampling $domain..."
    count=0
    while IFS= read -r f; do
        base="$(basename "${f%.*}")"
        out="$dst/${base}.wav"
        if [ -f "$out" ]; then
            continue
        fi
        ffmpeg -y -hide_banner -loglevel error -i "$f" -ar 22050 -ac 1 -sample_fmt s16 "$out"
        count=$((count + 1))
        if [ $((count % 100)) -eq 0 ]; then
            echo "  $domain: resampled $count files..."
        fi
    done < <(find "$src" -type f \( -iname '*.wav' -o -iname '*.mp3' -o -iname '*.flac' \
        -o -iname '*.ogg' -o -iname '*.m4a' -o -iname '*.aac' -o -iname '*.opus' \))
    echo "Done: $domain ($count files resampled)"
done
