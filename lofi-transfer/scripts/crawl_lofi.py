#!/usr/bin/env python3
"""Download lofi music samples from YouTube using yt-dlp search.

Searches for lofi hip hop beats, lofi jazz, lofi chill playlists, etc.
Downloads audio-only, converts to WAV.
"""

import argparse
import os
import subprocess
import sys
import json
from pathlib import Path

BREW_BIN = "/opt/homebrew/bin"
ENV = {**os.environ, "PATH": f"{BREW_BIN}:{os.environ.get('PATH', '')}"}

SEARCH_QUERIES = [
    "ytsearch20:lofi hip hop beats to study to",
    "ytsearch20:lofi chill beats instrumental",
    "ytsearch20:lofi jazz hop instrumental",
    "ytsearch20:lofi anime beats",
    "ytsearch20:chillhop instrumental beats",
    "ytsearch20:lofi bedroom producer beats",
    "ytsearch20:lofi rain beats instrumental",
    "ytsearch20:lofi vinyl beats",
    "ytsearch20:lofi piano beats instrumental",
    "ytsearch20:lofi guitar beats chill",
    "ytsearch15:lofi ambient study music",
    "ytsearch15:lofi night drive beats",
    "ytsearch15:lofi city pop beats instrumental",
    "ytsearch15:lofi soul hop beats",
    # Long compilations (1-5 hours each — massive data per download)
    "ytsearch10:lofi hip hop radio beats to relax study to",
    "ytsearch10:lofi chill mix 1 hour",
    "ytsearch10:lofi beats 2 hours",
    "ytsearch10:lofi music mix 3 hours",
    "ytsearch5:lofi hip hop mix compilation",
    "ytsearch5:lofi jazz cafe 1 hour",
    "ytsearch5:lofi rain 2 hours",
    "ytsearch5:lofi sleep music mix",
    "ytsearch5:chillhop essentials mix",
    "ytsearch5:lofi coding music mix",
    "ytsearch5:lofi aesthetic music compilation",
    "ytsearch5:lofi beats playlist mix 2024",
    "ytsearch5:japanese lofi hip hop mix",
    "ytsearch5:lofi piano study mix long",
]

PLAYLIST_URLS = [
    # Add specific playlist URLs here if known, e.g.:
    # "https://www.youtube.com/playlist?list=PLxxxxxxx",
]

MAX_DURATION = 18000  # 5 hours — lofi compilations are often 1-3 hrs
MIN_DURATION = 30


def download_query(query: str, out_dir: Path, archive_file: Path, dry_run: bool = False):
    """Download audio from a yt-dlp search query or URL."""
    cmd = [
        "/opt/homebrew/bin/yt-dlp",
        "--ffmpeg-location", "/opt/homebrew/bin/ffmpeg",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--cookies-from-browser", "firefox",
        "--output", str(out_dir / "%(title)s__%(id)s.%(ext)s"),
        "--match-filter", f"duration>{MIN_DURATION} & duration<{MAX_DURATION}",
        "--download-archive", str(archive_file),
        "--no-playlist" if not query.startswith("http") else "--yes-playlist",
        "--ignore-errors",
        "--no-overwrites",
        "--restrict-filenames",
        "--print-json" if not dry_run else "--simulate",
        query,
    ]

    if dry_run:
        print(f"  [DRY RUN] Would run: {' '.join(cmd[:8])}... {query}")
        return []

    print(f"  Downloading: {query}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=ENV)

    downloaded = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            info = json.loads(line)
            downloaded.append({
                "id": info.get("id"),
                "title": info.get("title"),
                "duration": info.get("duration"),
                "url": info.get("webpage_url"),
            })
        except json.JSONDecodeError:
            pass

    if result.returncode != 0 and result.stderr:
        for line in result.stderr.strip().split("\n"):
            if "ERROR" in line:
                print(f"    Warning: {line.strip()}")

    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Crawl YouTube for lofi music")
    parser.add_argument("--out-dir", default=None, help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without downloading")
    parser.add_argument("--extra-queries", nargs="*", default=[], help="Additional search queries")
    args = parser.parse_args()

    base = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent.parent / "data" / "raw" / "lofi"
    base.mkdir(parents=True, exist_ok=True)
    archive_file = base / ".yt_archive.txt"
    manifest_file = base / "download_manifest.jsonl"

    queries = SEARCH_QUERIES + [f"ytsearch10:{q}" for q in args.extra_queries] + PLAYLIST_URLS

    print(f"Lofi crawler: {len(queries)} queries, output to {base}")
    total = 0

    with open(manifest_file, "a") as mf:
        for query in queries:
            try:
                results = download_query(query, base, archive_file, args.dry_run)
                for r in results:
                    mf.write(json.dumps(r) + "\n")
                    total += 1
            except subprocess.TimeoutExpired:
                print(f"    Timeout on: {query}")
            except Exception as e:
                print(f"    Error on {query}: {e}")

    print(f"\nDone. Downloaded {total} tracks to {base}")
    print(f"Manifest: {manifest_file}")
    print(f"Archive (dedup): {archive_file}")


if __name__ == "__main__":
    main()
